import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig
from .datasets import classification_split, regression_split
from .metrics import LOWER_IS_BETTER, winner
from .plots import plot_comparison
from .runners import run_autogluon, run_fedot

PROBLEMS = {
    "classification": (classification_split, ["accuracy", "f1", "roc_auc"]),
    "regression": (regression_split, ["mae", "rmse", "r2"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="automl_benchmark", description="AutoGluon vs FEDOT")
    parser.add_argument("--time-limit", type=int, default=BenchmarkConfig.time_limit,
                        help="бюджет времени на задачу и фреймворк, секунд")
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument("--output", type=Path, default=Path("results/latest"),
                        help="каталог для results.json и comparison_results.png")
    return parser.parse_args()


def print_table(problem: str, metrics: list[str], res: dict) -> None:
    print(f"\n{problem}")
    print(f"{'metric':<10}{'AutoGluon':>12}{'FEDOT':>12}  best")
    for m in metrics:
        ag, fd = res["autogluon"][m], res["fedot"][m]
        print(f"{m:<10}{ag:>12.4f}{fd:>12.4f}  {winner(m, ag, fd)}")
    print(f"{'time, s':<10}{res['autogluon']['time']:>12.1f}{res['fedot']['time']:>12.1f}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    cfg = BenchmarkConfig(time_limit=args.time_limit, seed=args.seed)
    np.random.seed(cfg.seed)

    results = {}
    for problem, (make_split, metrics) in PROBLEMS.items():
        split = make_split(cfg.test_size, cfg.seed)
        print(f"{problem}: train={len(split.X_train)} test={len(split.X_test)}")
        results[problem] = {
            "autogluon": run_autogluon(problem, split, cfg),
            "fedot": run_fedot(problem, split, cfg),
        }
        print_table(problem, metrics, results[problem])

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"time_limit": cfg.time_limit, "seed": cfg.seed, "results": results}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    plot_comparison(results, cfg.time_limit, args.output / "comparison_results.png")
    print(f"\nСохранено в {args.output}")


if __name__ == "__main__":
    main()
