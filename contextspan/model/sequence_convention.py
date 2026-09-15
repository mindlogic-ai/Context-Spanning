"""The Context Spanning sequence convention, shared by training and inference.

One 17-row token sequence: text (ch0), 8 agent audio codebooks, 8 user audio codebooks.
Control tokens on the text row: `<ret>` asks the backend; a Context Span block is the reference
text between two span delimiters. Span and prefix columns carry placeholder audio (SILENCE on
the agent rows, SINE on the user rows) and are masked out of the loss. Everything that decides
how such a sequence is laid out lives here, so the assembler that writes the training data and
the engine that reads a span at inference cannot drift apart.
"""
import os
import random

import torch

TEXT_PAD = 3          # zero_text_code between words
RET_TOKEN_ID = 4      # `<ret>`: MoshiRAG rag_token_id (spm '<0x00>')
# Span delimiters. Checkpoints trained since 2026-09-08 (the released v7 weights included) close the
# block with a distinct token so the frame after the block carries an unambiguous "context ended,
# speak" signal: open = 12 (spm '<0x08>'), close = 13 (spm '<0x09>'). Older checkpoints used 12 on
# both sides; run those with CS_SPAN_CLOSE_ID=12. Training (moshicp) reads the same convention.
SPAN_OPEN_ID = 12
SPAN_CLOSE_ID = int(os.environ.get("CS_SPAN_CLOSE_ID", "13"))
SPAN_TOKEN_ID = SPAN_OPEN_ID   # legacy alias for the opening delimiter; never use it for pairing
N_AUDIO_CB = 8        # 8 agent + 8 user codebooks -> 17 rows with ch0
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]         # user audio on prefix/span frames
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]    # agent audio on prefix/span frames
RAG_DELAY = {"start_delay": 1.0, "end_gap": 1.0, "random_sampling_proba": 0.2}  # MoshiRAG Eq.3
FRAME_RATE = 12.5
REF_DROPOUT = 0.0


def persona_prompt(text: str) -> str:
    """Wrap the persona in the `<system>` markers PersonaPlex expects (idempotent)."""
    t = (text or "").strip()
    return t if t.startswith("<system>") else f"<system> {t} <system>"


def persona_text(persona: str, user: dict | None = None) -> str:
    """Persona followed by the user's name and city in the training corpus's exact phrasing.

    Dialogues where the agent addresses the user by name carry
    `<system> {persona} The user's name is {name}. The user is in {city}. <system>`; a prefix that
    names nobody teaches the model to invent a name (#33). `user` uses the ContextProfile keys.
    """
    t = (persona or "").strip()
    user = user or {}
    for key, phrase in (("name", "The user's name is {}."), ("city", "The user is in {}.")):
        v = str(user.get(key) or "").strip()
        if v:
            t = f"{t} {phrase.format(v)}" if t else phrase.format(v)
    return t


def context_span_ids(ref: str, spm) -> list:
    """Text-row ids of a Context Span block: delimiter, reference tokens, delimiter."""
    return [SPAN_OPEN_ID] + list(spm.encode(ref)) + [SPAN_CLOSE_ID]


def sample_rag_delay(d_lead: int, rng: random.Random, frame_rate: float = FRAME_RATE) -> int:
    """Frames between `<ret>` and the span's arrival, sampled as in MoshiRAG (Eq. 3)."""
    p = RAG_DELAY
    start_f, end_f = p["start_delay"] * frame_rate, p["end_gap"] * frame_rate
    if d_lead < start_f + end_f or rng.random() < p["random_sampling_proba"]:
        d = rng.randint(0, int(d_lead))
    else:
        lo, hi = int(round(start_f)), int(round(d_lead - end_f))
        d = rng.randint(lo, hi) if hi >= lo else rng.randint(0, int(d_lead))
    return min(d, int(d_lead))


def transcript(spm, tokens) -> str:
    """Render a text-row token stream as words with `<ret>` and `<span>` markers kept."""
    out, run = [], []
    for t in tokens:
        if t is None or t <= 2 or t == TEXT_PAD:
            continue
        if t in (RET_TOKEN_ID, SPAN_OPEN_ID, SPAN_CLOSE_ID):
            if run:
                out.append(spm.decode(run))
                run = []
            out.append("<ret>" if t == RET_TOKEN_ID else "<span>")
        else:
            run.append(int(t))
    if run:
        out.append(spm.decode(run))
    return " ".join(out)


def _col(tokens):
    return torch.tensor(tokens, dtype=torch.long)[:, None]


def persona_prefix(system_prompt: str, voice_codes, spm):
    """Masked PersonaPlex prefix [17, P]: voice codes -> silence -> persona text -> silence."""
    ag, us = _col(SILENCE_TOKENS), _col(SINE_TOKENS)
    parts = []

    def sil():
        return torch.cat([torch.full((1, 1), TEXT_PAD, dtype=torch.long), ag, us], 0)

    if voice_codes is not None:
        pv = voice_codes.long().shape[1]
        parts += [torch.cat([torch.full((1, pv), TEXT_PAD, dtype=torch.long), voice_codes.long(),
                             us.repeat(1, pv)], 0), sil()]
    ids = spm.encode(persona_prompt(system_prompt))
    if ids:
        n = len(ids)
        parts += [torch.cat([torch.tensor(ids, dtype=torch.long)[None], ag.repeat(1, n), us.repeat(1, n)], 0),
                  sil()]
    return torch.cat(parts, 1) if parts else torch.zeros(1 + 2 * N_AUDIO_CB, 0, dtype=torch.long)


def assemble_training_sequence(codes, audio_mask, spans, spm, rng=None, ref_dropout=REF_DROPOUT):
    """Splice Context Span blocks into clean codes at their inject frames.

    codes [17, T] (ch0 text, agent, user), audio_mask [T] bool, spans = [{inject_frame, reference}].
    Returns (codes [17, T'], text_mask [T'], audio_mask [T']). Span blocks are masked
    (text_mask=False) and carry SILENCE/SINE audio; the acoustic codebooks of frame f-1 MOVE to the
    block's last column so the model reads real audio exactly once around the block.
    """
    rng = rng or random.Random()
    codes = codes.to(torch.int64).clone()
    tmask = torch.ones(codes.shape[1], dtype=torch.bool)
    amask = audio_mask.clone()
    ag, us = torch.tensor(SILENCE_TOKENS), torch.tensor(SINE_TOKENS)
    for s in sorted(spans, key=lambda x: x["inject_frame"], reverse=True):
        ref = "" if (ref_dropout > 0 and rng.random() < ref_dropout) else s.get("reference", "")
        ids = context_span_ids(ref, spm)
        n, f = len(ids), max(0, min(int(s["inject_frame"]), codes.shape[1] - 1))
        block = torch.cat([torch.tensor(ids)[None], ag[:, None].repeat(1, n), us[:, None].repeat(1, n)], 0)
        if f > 0:
            ac = list(range(2, 9)) + list(range(10, 17))
            block[ac, n - 1] = codes[ac, f - 1]
            codes[2:9, f - 1], codes[10:17, f - 1] = ag[1:], us[1:]
            amask[f - 1] = False
        codes = torch.cat([codes[:, :f], block, codes[:, f:]], 1)
        tmask = torch.cat([tmask[:f], torch.zeros(n, dtype=torch.bool), tmask[f:]])
        amask = torch.cat([amask[:f], torch.zeros(n, dtype=torch.bool), amask[f:]])
    return codes, tmask, amask
