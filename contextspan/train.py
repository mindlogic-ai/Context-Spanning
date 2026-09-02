"""Data preparation and training.

Data format (one JSON per dialogue, next to a stereo wav with L=user, R=agent):
  {"system_prompt": "...", "voice": "voices/f0.pt",
   "turns": [{"speaker": "user"|"agent", "words": [{"w": "hello", "t": 4.21}, ...],
              "ret": true, "reference": "The cafe opens at seven thirty AM."}]}
Word times are seconds; `ret`/`reference` mark an agent turn whose answer is grounded on the
reference. `prepare` encodes audio with Mimi and writes <name>.npz (codes, ch0, audio_mask,
spans); `train` fine-tunes the model on a directory of such files.
"""
import glob
import json
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .model import BASE_REPO, _hf, load_model, load_voice
from .spans import (FRAME_RATE, N_AUDIO_CB, RET_TOKEN_ID, TEXT_PAD, assemble_training_sequence,
                    persona_prefix, sample_rag_delay)


@torch.no_grad()
def prepare(in_dir, out_dir, device="cuda"):
    import sentencepiece
    from moshi.models import loaders
    mimi = loaders.get_mimi(_hf(BASE_REPO, loaders.MIMI_NAME), device)
    spm = sentencepiece.SentencePieceProcessor(_hf(BASE_REPO, loaders.TEXT_TOKENIZER_NAME))
    os.makedirs(out_dir, exist_ok=True)
    fr, sr = FRAME_RATE, int(mimi.sample_rate)
    for jp in sorted(glob.glob(f"{in_dir}/*.json")):
        d = json.load(open(jp))
        wav, s = sf.read(jp[:-5] + ".wav", dtype="float32", always_2d=True)
        if s != sr:
            import librosa
            wav = np.stack([librosa.resample(wav[:, c], orig_sr=s, target_sr=sr) for c in range(2)], 1)
        T = int(len(wav) * fr / sr)
        x = torch.tensor(wav[:T * int(sr / fr)].T, device=device)[:, None]        # [2,1,S]
        user = mimi.encode(x[0:1])[0, :N_AUDIO_CB, :T].cpu()
        agent = mimi.encode(x[1:2])[0, :N_AUDIO_CB, :T].cpu()
        ch0 = torch.full((T,), TEXT_PAD, dtype=torch.long)
        spans = []
        for turn in d["turns"]:
            if turn["speaker"] != "agent" or not turn.get("words"):
                continue
            f0 = int(turn["words"][0]["t"] * fr)
            for w in turn["words"]:
                for k, tid in enumerate(spm.encode(" " + w["w"])):
                    f = int(w["t"] * fr) + k
                    if 0 <= f < T and ch0[f] == TEXT_PAD:
                        ch0[f] = tid
            if turn.get("ret") and turn.get("reference"):
                rf = max(0, f0 - 1)
                ch0[rf] = RET_TOKEN_ID                                            # <ret> right before the turn
                body = turn.get("body_word_index", len(turn["words"]) // 3)       # lead ends here
                cap = int(turn["words"][min(body, len(turn["words"]) - 1)]["t"] * fr) - 2
                spans.append({"ret_frame": rf, "cap_frame": cap, "reference": turn["reference"]})
        np.savez(f"{out_dir}/{os.path.basename(jp)[:-5]}.npz", agent=agent.numpy(), user=user.numpy(),
                 ch0=ch0.numpy(), spans=json.dumps(spans), system_prompt=d.get("system_prompt", ""),
                 voice=d.get("voice", ""))
        print("prepared", jp)


def _sample(path, spm, rng):
    z = np.load(path, allow_pickle=True)
    codes = torch.cat([torch.tensor(z["ch0"])[None], torch.tensor(z["agent"]), torch.tensor(z["user"])], 0)
    spans = []
    for s in json.loads(str(z["spans"])):
        d_lead = max(1, s["cap_frame"] - s["ret_frame"])
        spans.append({"inject_frame": s["ret_frame"] + max(1, sample_rag_delay(d_lead, rng)),
                      "reference": s["reference"]})
    codes, tmask, amask = assemble_training_sequence(codes, torch.ones(codes.shape[1], dtype=torch.bool),
                                                     spans, spm, rng)
    voice = load_voice(str(z["voice"])) if str(z["voice"]) else None
    prefix = persona_prefix(str(z["system_prompt"]), voice, spm)
    P = prefix.shape[1]
    codes = torch.cat([prefix, codes], 1)
    tmask = torch.cat([torch.zeros(P, dtype=torch.bool), tmask])
    amask = torch.cat([torch.zeros(P, dtype=torch.bool), amask])
    return codes, tmask, amask


def loss_fn(out, codes, tmask, amask, lm, w_ret=5.0, pad_w=0.5, acoustic_w=0.02):
    tl = torch.nan_to_num(out.text_logits[:, 0], nan=0.0)                    # [B,T,V]
    tgt = codes[:, 0]
    keep = out.text_mask[:, 0] & tmask
    ce = F.cross_entropy(tl.reshape(-1, tl.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    w = keep.float()
    w = torch.where(tgt == TEXT_PAD, w * pad_w, w)
    w = torch.where(tgt == RET_TOKEN_ID, w * w_ret, w)
    text_loss = (ce * w).sum() / w.sum().clamp_min(1)
    al = torch.nan_to_num(out.logits, nan=0.0)                                # [B,dep_q,T,card]
    ao = lm.audio_offset
    tga = codes[:, ao:ao + al.shape[1]]
    cea = F.cross_entropy(al.reshape(-1, al.shape[-1]), tga.reshape(-1), reduction="none").reshape(tga.shape)
    cbw = torch.full((al.shape[1],), acoustic_w, device=codes.device)
    cbw[0] = 1.0
    aw = (out.mask & amask[:, None, :]).float() * cbw[None, :, None]
    audio_loss = (cea * aw).sum() / aw.sum().clamp_min(1)
    return text_loss + audio_loss, text_loss.item(), audio_loss.item()


def train(data_dir, out_dir, checkpoint=None, steps=1000, lr=2e-6, accum=8, context=3000,
          ckpt_every=250, w_ret=5.0, seed=0, device="cuda"):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lm, mimi, spm = load_model(checkpoint, device) if checkpoint else load_model(None, device)
    del mimi
    lm.train()
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    os.makedirs(out_dir, exist_ok=True)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        tl_acc = al_acc = 0.0
        for _ in range(accum):
            codes, tmask, amask = _sample(rng.choice(files), spm, rng)
            if codes.shape[1] > context:
                s0 = rng.randint(0, codes.shape[1] - context)
                codes, tmask, amask = codes[:, s0:s0 + context], tmask[s0:s0 + context], amask[s0:s0 + context]
            codes, tmask, amask = (t.to(device)[None] for t in (codes, tmask, amask))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lm.forward_train(codes)
                loss, tl, al = loss_fn(out, codes, tmask, amask, lm, w_ret=w_ret)
            (loss / accum).backward()
            tl_acc += tl / accum
            al_acc += al / accum
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step} text_loss {tl_acc:.3f} audio_loss {al_acc:.3f}", flush=True)
        if step % ckpt_every == 0 or step == steps:
            torch.save({"model": lm.state_dict()}, f"{out_dir}/context_spanning_step{step}.pt")
