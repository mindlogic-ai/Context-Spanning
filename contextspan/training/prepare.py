"""Data preparation: dialogues (JSON + stereo wav) -> training tensors (.npz).

One JSON per dialogue, next to a stereo wav with L=user, R=agent:
  {"system_prompt": "...", "voice": "voices/f0.pt",
   "turns": [{"speaker": "user"|"agent", "words": [{"w": "hello", "t": 4.21}, ...],
              "ret": true, "reference": "The cafe opens at seven thirty AM."}]}
Word times are seconds; `ret`/`reference` mark an agent turn whose answer is grounded on the
reference. The output <name>.npz holds the Mimi codes of both channels, the text row (ch0), the
span records (ret frame, cap frame, reference), the persona and the voice.
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from ..model.sequence_convention import FRAME_RATE, N_AUDIO_CB, RET_TOKEN_ID, TEXT_PAD
from ..model.weights import load_mimi, load_tokenizer


@torch.no_grad()
def prepare(in_dir, out_dir, device="cuda"):
    mimi, spm = load_mimi(device), load_tokenizer()
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
