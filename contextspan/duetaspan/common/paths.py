"""Single configuration point for every on-disk location the pipeline uses.

No module in this repository hardcodes an absolute path; they all resolve through
here, so the same code runs on any machine once a handful of environment
variables point at that machine's storage.

| variable                | holds                                        | default             |
| ----------------------- | -------------------------------------------- | ------------------- |
| `DUETASPAN_ROOT`        | this repository                              | auto-detected       |
| `DUETASPAN_DATA`        | corpora, shards, voice banks, tensors        | `$ROOT/datasets`    |
| `DUETASPAN_MODELS`      | base weights, tokenizers, ASR/aligner ckpts  | `$ROOT/models`      |
| `DUETASPAN_WORK`        | bulk scratch: production parts, logs, pages  | `$ROOT/work`        |
| `DUETASPAN_CACHE`       | HuggingFace / torch download cache           | `$ROOT/.cache`      |
| `DUETASPAN_VENDOR`      | vendored `moshi` forks (obtained separately) | `$ROOT/vendor`      |
| `DUETASPAN_CHECKPOINTS` | training checkpoints                         | `$WORK/checkpoints` |
| `DUETASPAN_CORPUS`      | the vectorized corpus the trainer reads      | `$DATA/duetaspan_tensors` |

`HF_HOME` is honoured when it is already exported; otherwise `export_hf_home()`
points it at `CACHE`. A handful of corpora live outside these roots on some
machines and take a per-asset override through `env_path()`.

Example — one export block per machine::

    export DUETASPAN_ROOT=/srv/duetaspan
    export DUETASPAN_DATA=/mnt/corpora/duetaspan
    export DUETASPAN_MODELS=/mnt/weights
    export DUETASPAN_WORK=/mnt/scratch/duetaspan
    export DUETASPAN_VENDOR=/mnt/weights/forks
    export PYTHONPATH=$DUETASPAN_ROOT:$DUETASPAN_VENDOR/personaplex/moshi
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ROOT", "DATA", "MODELS", "WORK", "CACHE", "VENDOR", "CHECKPOINTS", "SPM",
    "CORPUS", "VOICE_PROMPTS",
    "PERSONAPLEX_MOSHI", "KYUTAI_MOSHI", "MOSHIRAG_MOSHI",
    "env_path", "export_hf_home",
]


def _env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


#: Repository root — `duetaspan/common/paths.py` sits two levels below it.
ROOT: Path = _env("DUETASPAN_ROOT", Path(__file__).resolve().parents[2])

#: Corpora and generated data (dialogue shards, audio parts, tensors, voice banks).
DATA: Path = _env("DUETASPAN_DATA", ROOT / "datasets")

#: Model weights and tokenizers obtained separately (PersonaPlex, Kyutai TTS, ASR).
MODELS: Path = _env("DUETASPAN_MODELS", ROOT / "models")

#: Bulk scratch: production part trees, run logs, the review site, exports.
WORK: Path = _env("DUETASPAN_WORK", ROOT / "work")

#: HuggingFace / torch download cache.
CACHE: Path = _env("DUETASPAN_CACHE", ROOT / ".cache")

#: Vendored `moshi` forks — obtained separately, not part of this repository.
#: See the quick start in `README.md` for which fork must be importable as `moshi`.
VENDOR: Path = _env("DUETASPAN_VENDOR", ROOT / "vendor")

#: Training checkpoints.
CHECKPOINTS: Path = _env("DUETASPAN_CHECKPOINTS", WORK / "checkpoints")

#: The vectorized corpus: `train_*.context.jsonl` shards and the `codes_all/` blobs beside
#: them. It is what `vectorize` writes, what `prepare_manifest` pins a split over, and what
#: the trainer reads — one name, so those three cannot be pointed at different generations
#: of the same data by accident.
CORPUS: Path = _env("DUETASPAN_CORPUS", DATA / "duetaspan_tensors")

#: Voice-prompt bank: one `<voice-id>.pt` of agent-voice Mimi codes per identity, plus the
#: `<voice-id>.aug1.pt` augmented variant. The PersonaPlex conditioning prefix is built from it.
VOICE_PROMPTS: Path = DATA / "moshicp" / "voice_prompts"

#: SentencePiece model shared by the LM, the vectorizer and every QA gate.
SPM: Path = MODELS / "personaplex-7b-v1" / "tokenizer_spm_32k_3.model"

#: The PersonaPlex fork of `moshi` (dep_q=16, voice prompts) — the importable `moshi`.
PERSONAPLEX_MOSHI: Path = VENDOR / "personaplex" / "moshi"
#: The Kyutai TTS fork of `moshi` (ships `moshi.models.tts`).
KYUTAI_MOSHI: Path = VENDOR / "kyutai_tts" / "moshi"
#: The MoshiRAG reference fork (kyutai STT used for alignment backfill).
MOSHIRAG_MOSHI: Path = VENDOR / "reference" / "moshirag" / "moshi"


def env_path(name: str, default: Path | str) -> str:
    """Per-asset override: ``env_path("XSTEST_CSV", DATA / "xstest.csv")``."""
    return os.environ.get(name) or str(default)


def export_hf_home() -> str:
    """Point `HF_HOME` at `CACHE` unless the caller already exported one."""
    os.environ.setdefault("HF_HOME", str(CACHE))
    return os.environ["HF_HOME"]
