"""Real-module smoke: participant 1 policy -> participant 4 logs -> participant 2 IPS."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explorekit.integration import run_smoke              
from explorekit.logging import write_logs_parquet              


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ExploreKit integration smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=1000)
    parser.add_argument("--write-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(seed=args.seed, n_rounds=args.n_rounds)
    print(f"logs={result.n_logs}")
    print(f"mean_reward={result.mean_reward:.6f}")
    print(f"IPS={result.ips_estimate:.6f}")
    print(f"ESS={result.ess:.1f}")
    print(f"reliability={result.reliability}")
    if args.write_logs:
        paths = write_logs_parquet(result.logs, "outputs/logs")
        print("saved_logs=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
