"""Full-Duplex-Bench v1 / v1.5 renderer: every sample's input.wav goes through the real stack at the
1.0x frame clock and the agent audio is written in the official layout ({task}/{id}/output.wav, plus
clean_output.wav for v1.5 overlap tasks), ready for the benchmark's own scorers.

    python -m eval.fdb.v1_render <data_root> <out_root> [--checkpoint ckpt.pt] [--limit N] [--tasks a,b]

Prompts are the ones the benchmark specifies for PersonaPlex: the teacher prompt for interruption
tasks, "You enjoy having a good conversation." otherwise.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import soundfile as sf

from eval.common import Stack, load_mono, transcript

P_TEACHER = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
P_CONV = "You enjoy having a good conversation."


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root"); ap.add_argument("out_root")
    ap.add_argument("--checkpoint", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default="", help="comma-separated task dirs to run (default all)")
    ap.add_argument("--voice", default=None)
    a = ap.parse_args(argv)
    samples = sorted(os.path.dirname(p) for p in glob.glob(f"{a.data_root}/**/input.wav", recursive=True))
    only = {t for t in a.tasks.split(",") if t}
    bytask = {}
    for s in samples:
        task = os.path.relpath(s, a.data_root).split(os.sep)[0]
        if not only or task in only:
            bytask.setdefault(task, []).append(s)
    print(f"[fdb] {len(samples)} samples / {len(bytask)} tasks: { {k: len(v) for k, v in bytask.items()} }", flush=True)
    st = Stack(a.checkpoint, voice=a.voice)
    done = 0
    for task, dirs in bytask.items():
        for si, sdir in enumerate(dirs[: a.limit] if a.limit else dirs):
            rel = os.path.relpath(sdir, a.data_root); odir = f"{a.out_root}/{rel}"
            if os.path.exists(f"{odir}/output.wav"):
                continue
            os.makedirs(odir, exist_ok=True)
            prompt = P_TEACHER if "interrupt" in rel.lower() else P_CONV

            def render(src, dst):
                pcm = np.concatenate([load_mono(f"{sdir}/{src}", st.sr), np.zeros(2 * st.sr, np.float32)])
                res = st.run(prompt, rel, pcm)
                sf.write(f"{odir}/{dst}", res["agent"], st.sr)
                return res, len(pcm) // st.fs

            if os.path.exists(f"{sdir}/clean_input.wav") and not os.path.exists(f"{odir}/clean_output.wav"):
                render("clean_input.wav", "clean_output.wav")
            res, n = render("input.wav", "output.wav")
            for f in os.listdir(sdir):                      # the scorers want the inputs next to the outputs
                if f != "output.wav" and not os.path.exists(f"{odir}/{f}"):
                    shutil.copy(f"{sdir}/{f}", f"{odir}/{f}")
            json.dump({"transcript": transcript(st.eng.spm, res["tokens"]),
                       "events": [{k: e.get(k) for k in ("t_ret", "t_inj", "question", "src", "inject")} for e in res["events"]]},
                      open(f"{odir}/run_events.json", "w"), ensure_ascii=False)
            done += 1
            print(f"[fdb] {task} {si + 1}/{len(dirs)} {rel}: {n / st.eng.frame_rate:.0f}s ret={len(res['events'])}", flush=True)
    print(f"[fdb] done total={done}", flush=True)


if __name__ == "__main__":
    main()
