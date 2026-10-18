from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description="MLX conditional memory research prototype")
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("experiment", help="fit memory and evaluate held-out sign composition")
    train.add_argument("--model", default="Qwen/Qwen3-4B-MLX-bf16")
    train.add_argument("--output", default="runs/pilot-seed17")
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--steps", type=int, default=240)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--eval-every", type=int, default=60)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--slots", type=int, default=8192)
    train.add_argument("--dimension", type=int, default=64)
    train.add_argument("--layer", type=int)
    train.add_argument("--train-components", type=int, default=24)
    train.add_argument("--valid-components", type=int, default=8)
    train.add_argument("--test-components", type=int, default=16)
    train.add_argument("--arax-snapshot", help="directory containing the three versioned ARAX case snapshots")
    fetch = sub.add_parser("arax", help="fetch a bounded rare-disease neighborhood and retain raw receipts")
    fetch.add_argument("--output", default="data/processed/arax")
    fetch.add_argument("--max-results", type=int, default=30)
    fetch.add_argument("--timeout", type=int, default=120)
    fetch.add_argument("--endpoint", default="https://arax.ncats.io/api/arax/v1.4")
    fetch.add_argument("--diseases", nargs="+", default=["Fabry disease", "Gaucher disease", "Rett syndrome"])
    args = p.parse_args()
    if args.command == "experiment":
        from .experiment import run
        result = run(args)
        print(json.dumps({"output": args.output, "conditions": {
            k: v["accuracy"] for k, v in result["conditions"].items()}}))
    elif args.command == "arax":
        from .arax import fetch_cases
        print(json.dumps(fetch_cases(args), indent=2))


if __name__ == "__main__":
    main()
