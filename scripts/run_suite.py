"""Sequential GPU runs with a fixed recipe and separate logs for every seed."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--seeds", nargs="+", type=int, default=[23, 42, 71])
p.add_argument("--steps", type=int, default=480)
p.add_argument("--arax-snapshot")
p.add_argument("--prefix", default="replication")
args = p.parse_args()
for seed in args.seeds:
    out = Path("runs") / f"{args.prefix}-seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    cmd = [
        sys.executable,
        "-m",
        "engram_poc.cli",
        "experiment",
        "--seed",
        str(seed),
        "--steps",
        str(args.steps),
        "--dimension",
        "1",
        "--eval-every",
        "120",
        "--output",
        str(out),
    ]
    if args.arax_snapshot:
        cmd += ["--arax-snapshot", args.arax_snapshot]
    print(json.dumps({"starting": str(out), "command": cmd}), flush=True)
    with (out / "console.log").open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        print((out / "console.log").read_text()[-5000:], flush=True)
        raise SystemExit(proc.returncode)
    results = json.loads((out / "results.json").read_text())
    print(
        json.dumps(
            {
                "completed": str(out),
                "accuracy": {
                    k: v["accuracy"] for k, v in results["conditions"].items()
                },
                "seconds": results["runtime"]["seconds"],
            }
        ),
        flush=True,
    )
