"""The math benchmark build without a GPU: meta.json rows, TTS text normalisation, voice assignment and the
stereo layout of audios/ (lead silence, noise channel, resume that reproduces the same files)."""
import importlib.util
import json
import os

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("build_audio", os.path.join(_HERE, "..", "benchmark", "math", "build_audio.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _questions(root):
    os.makedirs(os.path.join(root, "questions"))
    rows = {"GSM8K": [("GSM8K-0", "How many eggs?", "18")],
            "AddSub": [("AddSub-1", "Joan found 70 seashells .", "43"), ("AddSub-2", "Tim has *two* “big” hats— why?", "26")]}
    for name in B.SETS:                                                        # every set gets a file; three stay empty
        rows.setdefault(name, [])
    for name, items in rows.items():
        with open(os.path.join(root, "questions", f"{name}.jsonl"), "w") as f:
            for i, q, a in items:
                f.write(json.dumps({"id": i, "dataset": name, "question": q, "answer": a}) + "\n")


def test_meta_follows_set_order(tmp_path):
    _questions(str(tmp_path))
    meta = B.load_meta(str(tmp_path))
    assert [r["id"] for r in meta] == ["AddSub-1", "AddSub-2", "GSM8K-0"]      # benchmark set order, AddSub before GSM8K
    assert meta[0] == {"id": "AddSub-1", "dataset": "AddSub", "text": "Joan found 70 seashells .",
                       "answer": "43", "knowledge": None}


def test_normalize():
    assert B.normalize("Tim has *two* “big” hats— why?") == "Tim has two big hats, why?"
    assert B.normalize(' Sara’s "red"  ball ') == "Sara's red ball"


def test_voice_assignment_is_deterministic_and_distinct():
    pool = [f"voice-donations/v{i:03d}{B.VOICE_SUFFIX}" for i in range(50)]
    for item_id in ("AddSub-1", "SVAMP-chal-7", "GSM8K-1318"):
        va, vu = B.assign_voices(item_id, pool)
        assert va != vu and va in pool and vu in pool
        assert B.assign_voices(item_id, list(pool)) == (va, vu)


def test_voice_pool_excludes_enhanced_copies(tmp_path):
    d = tmp_path / "voice-donations"
    d.mkdir()
    for n in ("b", "a", "a_enhanced", "c"):
        (d / (n + B.VOICE_SUFFIX)).write_bytes(b"")
    (d / "a.wav").write_bytes(b"")
    assert B.voice_pool(str(tmp_path)) == [f"voice-donations/{n}{B.VOICE_SUFFIX}" for n in ("a", "b", "c")]


def test_assemble_layout_and_resume(tmp_path):
    out = str(tmp_path)
    _questions(out)
    meta = B.load_meta(out)
    os.makedirs(os.path.join(out, "speech"))
    rng = np.random.default_rng(0)
    for k, r in enumerate(meta):
        sf.write(os.path.join(out, "speech", f"{r['id']}.wav"), rng.uniform(-0.5, 0.5, B.SR // 2 + k * 100), B.SR, subtype="PCM_16")
    assert B.assemble(meta, out) == 3
    lead = int(B.LEAD_S * B.SR)
    first = {}
    for r in meta:
        path = os.path.join(out, "audios", f"{r['id']}.wav")
        x, sr = sf.read(path, dtype="int16")
        speech, _ = sf.read(os.path.join(out, "speech", f"{r['id']}.wav"), dtype="int16")
        assert sr == B.SR and x.shape == (lead + len(speech), 2) and sf.info(path).subtype == "PCM_16"
        assert not x[:lead, 0].any() and np.array_equal(x[lead:, 0], speech)
        assert 5 < x[:, 1].std() < 9                                           # noise std 7 in int16 units
        first[r["id"]] = open(path, "rb").read()
    os.remove(os.path.join(out, "audios", "AddSub-2.wav"))                     # the middle item: noise must line up
    assert B.assemble(meta, out) == 1
    for r in meta:
        assert open(os.path.join(out, "audios", f"{r['id']}.wav"), "rb").read() == first[r["id"]]
