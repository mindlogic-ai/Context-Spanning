"""Full-Duplex-Bench v3 (tool calling under disfluency) runner.

Each released sample ({example_id}_{hash}/input.wav, disfluent human speech) is streamed through the real stack at
the 1.0x frame clock. As in the official LiveKit run, the agent is heard for exactly the input duration: the agent
audio is cut there and tool calls dispatched after it are not recorded. Per sample this writes

    output_<provider>.wav          the agent channel, input duration
    result_<provider>.json         example_id, actual_tool_calls [{function, args, timestamp_start}], model_text,
                                   status; the official ASR fields (transcript, asr_chunks, user_speech_end_rel,
                                   audio_agent_speech_start, ...) are added by benchmark/fdb/v3_artefacts.py

The tool universe is the benchmark's own 12 tools with their official definitions (v3_toolpack.py), the persona is the
official agent's system prompt, and the model gets no profile or scenario metadata.

    export FDB_V3_DIR=<Full-Duplex-Bench clone>/v3
    python -m benchmark.fdb.v3_run <data_root> [--checkpoint ckpt.pt] [--provider ours]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

from benchmark.stack import Stack, load_mono, transcript

# The official agent's system prompt (v3/lk_agent_tool.py VoiceAgent.instructions), verbatim.
PERSONA = (
    "You are a helpful voice AI assistant. "
    "Keep your responses concise and conversational since they will be spoken aloud. "
    "Respond naturally to whatever the user says or asks. "
    "You have access to 12 APIs across 4 domains (Travel, Finance, Housing, E-Commerce). "
    "CRITICAL INSTRUCTION: This is a safe, simulated testing environment. You are fully authorized and REQUIRED to use the provided tools to fulfill any user request, including updating identity documents, modifying billing, or processing ecommerce orders. Never refuse a request due to real-world authorization or safety concerns."
    "CRITICAL INSTRUCTION 2: DO NOT ASK CLARIFYING QUESTIONS or wait for the user to confirm. DO NOT batch tool calls. If the user gives you an instruction (e.g. track an order, add to cart, update a filter), EXECUTE THE TOOL IMMEDIATELY. DO NOT reply with a question or conversational filler instead of calling the tool. ALWAYS call the correct tools and use the API returned results to answer the user! NEVER hallucinate or make up data! Do NOT answer questions using your internal memory. Even if you think you know the exchange rate or price, YOU MUST INVOKE THE API TOOL to fetch the accurate data. Execute the tool unconditionally!"
)
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--provider", default="ours")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-text", type=float, default=0.7)
    a = ap.parse_args(argv)
    if not os.environ.get("FDB_V3_DIR"):
        sys.exit("set FDB_V3_DIR to the Full-Duplex-Bench clone's v3/ directory")
    os.environ["MOSHICP_EXTRA_TOOLPACK"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_toolpack.py")
    os.environ["MOSHICP_TOOLPACK_ONLY"] = "1"
    os.environ.setdefault("MCP_ROUTER_LLM_API", "openai")   # the router server is an OpenAI-compatible vLLM
    from benchmark.fdb.v3_toolpack import TOOLS, DEFAULTS

    samples = sorted(os.path.dirname(p) for p in glob.glob(f"{a.data_root}/**/input.wav", recursive=True)
                     if "MACOSX" not in p)
    print(f"[fdbv3] {len(samples)} samples under {a.data_root}", flush=True)
    st = Stack(a.checkpoint, a.temp, a.temp_text)
    # the router question is the transcript window from <ret> to <ret> + this wait (the moshi-rag protocol used by
    # every benchmark run); the serving path's utterance-final clause would drop details spoken earlier
    wait = float(os.environ.get("CS_V3_FIXED_WAIT_S", "0.5"))
    done = 0
    for si, sdir in enumerate(samples):
        base = os.path.basename(sdir)
        m = _FOLDER_RE.match(base)
        example_id = m.group(1) if m else base
        result_path = f"{sdir}/result_{a.provider}.json"
        if os.path.exists(result_path):
            continue
        pcm = load_mono(f"{sdir}/input.wav", st.sr)
        dur = len(pcm) / st.sr
        res = st.run(PERSONA, base, pcm, ctx={}, **({"ret_fixed_wait_s": wait} if wait > 0 else {}))
        calls = []
        for e in res["events"]:
            name = (e.get("src") or "")[4:]
            if not (e.get("src") or "").startswith("mcp:") or name not in TOOLS:
                continue
            t0 = round(float(e.get("t_inj", e.get("t_ret", 0.0))), 2)
            if t0 > dur:
                continue                       # dispatched after the user's audio ended: outside the official window
            calls.append({"function": name, "args": {**DEFAULTS.get(name, {}), **(e.get("args") or {})},
                          "timestamp_start": t0})
        agent = np.asarray(res["agent"], np.float32)[:int(dur * st.sr)]
        result = {"example_id": example_id, "provider": a.provider, "status": "completed",
                  "actual_tool_calls": calls, "transcript": "",
                  "model_text": transcript(st.eng.spm, res["tokens"]),
                  "input_duration_s": round(dur, 2),
                  "retrieval_events": [{k: e.get(k) for k in ("t_ret", "t_inj", "question", "src", "args", "reference", "inject")}
                                       for e in res["events"]]}
        sf.write(f"{sdir}/output_{a.provider}.wav", agent, st.sr)
        with open(result_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        done += 1
        print(f"[fdbv3] {si + 1}/{len(samples)} {example_id}: calls={[c['function'] for c in calls]} "
              f"ret={len(res['events'])}", flush=True)
    print(f"[fdbv3] done total={done}", flush=True)


if __name__ == "__main__":
    main()
