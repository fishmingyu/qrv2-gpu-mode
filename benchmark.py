from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from reference import check_implementation, generate_input


@dataclass(frozen=True)
class Case:
    batch: int
    n: int
    cond: int
    profile: str
    seed: int

    @property
    def label(self) -> str:
        return f"b{self.batch}_n{self.n}_cond{self.cond}_{self.profile}_s{self.seed}"


LEADERBOARD_CASES = (
    Case(20, 32, 1, "dense", 43214),
    Case(40, 176, 1, "dense", 423011),
    Case(40, 352, 1, "dense", 123456),
    Case(640, 512, 2, "dense", 1029),
    Case(60, 1024, 2, "dense", 75342),
    Case(8, 2048, 1, "dense", 224466),
    Case(2, 4096, 1, "dense", 32412),
    Case(640, 512, 2, "mixed", 770001),
    Case(60, 1024, 2, "mixed", 770002),
    Case(640, 512, 0, "rankdef", 770003),
    Case(640, 512, 0, "clustered", 770004),
    Case(60, 1024, 0, "nearrank", 770005),
)

_L2_FLUSH_BYTES = 256 * 1024 * 1024
_l2_buffer: torch.Tensor | None = None


def _load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)
    return module


def _parse_case(value: str) -> Case:
    parts = value.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "case must be batch,n,cond,profile,seed"
        )
    batch, n, cond, profile, seed = parts
    return Case(int(batch), int(n), int(cond), profile, int(seed))


def _input(case: Case, seed: int | None = None) -> torch.Tensor:
    return generate_input(
        batch=case.batch,
        n=case.n,
        cond=case.cond,
        case=case.profile,
        seed=case.seed if seed is None else seed,
    )


def _clone(value):
    return value.clone() if isinstance(value, torch.Tensor) else value


def _flush_l2() -> None:
    global _l2_buffer
    device = torch.device("cuda", torch.cuda.current_device())
    if _l2_buffer is None or _l2_buffer.device != device:
        _l2_buffer = torch.empty(
            (_L2_FLUSH_BYTES,), device=device, dtype=torch.uint8
        )
    _l2_buffer.zero_()


def _input_count(case: Case) -> int:
    bytes_per_input = case.batch * case.n * case.n * 4
    return max(1, min(50, (256 * 1024 * 1024) // bytes_per_input))


def _check_case(case: Case, kernel) -> dict[str, object]:
    data = _input(case)
    torch.cuda.synchronize()
    output = kernel(_clone(data))
    torch.cuda.synchronize()
    good, message = check_implementation(data, output)
    if not good:
        raise RuntimeError(message)
    return {
        "label": case.label,
        **asdict(case),
        "status": "pass",
        "correctness": True,
        "message": message,
    }


def _benchmark_case(
    case: Case,
    kernel,
    *,
    max_repeats: int,
    max_time_ns: float,
    recheck: bool,
) -> dict[str, object]:
    count = _input_count(case)
    inputs = [_input(case, case.seed + 42 * (i + 1)) for i in range(count)]
    originals = [_clone(data) for data in inputs]

    outputs = [kernel(_clone(data)) for data in inputs]
    for data, output in zip(originals, outputs):
        good, message = check_implementation(data, output)
        if not good:
            raise RuntimeError(message)

    durations: list[float] = []
    wall_start = time.perf_counter_ns()
    for iteration in range(max_repeats):
        torch.cuda.synchronize()
        _flush_l2()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        outputs = [kernel(data) for data in inputs]
        end.record()
        torch.cuda.synchronize()
        durations.append(begin.elapsed_time(end) * 1.0e6 / count)

        if recheck:
            for data, output in zip(originals, outputs):
                good, message = check_implementation(data, output)
                if not good:
                    raise RuntimeError(message)

        elapsed = time.perf_counter_ns() - wall_start
        if iteration >= 2 and elapsed > 1.0e8:
            mean = sum(durations) / len(durations)
            if len(durations) > 1:
                variance = sum((value - mean) ** 2 for value in durations)
                std = math.sqrt(variance / (len(durations) - 1))
                err = std / math.sqrt(len(durations))
            else:
                std = err = 0.0
            if (
                err / mean < 0.001
                or mean * len(durations) > max_time_ns
                or elapsed > 120.0e9
            ):
                break

    mean_ns = sum(durations) / len(durations)
    variance = sum((value - mean_ns) ** 2 for value in durations)
    std_ns = (
        math.sqrt(variance / (len(durations) - 1))
        if len(durations) > 1
        else 0.0
    )
    mean_ms = mean_ns / 1.0e6
    return {
        "label": case.label,
        **asdict(case),
        "status": "pass",
        "correctness": True,
        "runs": len(durations),
        "mean_ns": mean_ns,
        "std_ns": std_ns,
        "best_ns": min(durations),
        "worst_ns": max(durations),
        "mean_ms": mean_ms,
        "best_ms": min(durations) / 1.0e6,
        "tflops_est_mean": (
            case.batch * (4.0 / 3.0) * case.n**3 / mean_ms / 1.0e9
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correctness and leaderboard harness for QR2 mega-kernels"
    )
    parser.add_argument(
        "--mode", choices=("test", "benchmark", "leaderboard"), default="test"
    )
    parser.add_argument("--case", action="append", type=_parse_case)
    parser.add_argument("--submission", type=Path, default=Path("submission.py"))
    parser.add_argument("--max-repeats", type=int, default=200)
    parser.add_argument("--max-time-ns", type=float, default=10.0e9)
    parser.add_argument(
        "--recheck", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.mode == "leaderboard":
        args.max_repeats = max(args.max_repeats, 1000)
        args.max_time_ns = max(args.max_time_ns, 30.0e9)

    kernel = _load_submission(args.submission).custom_kernel
    cases = tuple(args.case) if args.case else LEADERBOARD_CASES
    rows: list[dict[str, object]] = []
    for case in cases:
        try:
            if args.mode == "test":
                row = _check_case(case, kernel)
                print(f"{case.label:36s} pass {row['message']}")
            else:
                row = _benchmark_case(
                    case,
                    kernel,
                    max_repeats=args.max_repeats,
                    max_time_ns=args.max_time_ns,
                    recheck=args.recheck,
                )
                print(
                    f"{case.label:36s} mean={row['mean_ms']:9.4f} ms "
                    f"best={row['best_ms']:9.4f} ms runs={row['runs']:3d} "
                    f"{row['tflops_est_mean']:8.2f} TF/s"
                )
            rows.append(row)
        except Exception as error:
            if not args.continue_on_error:
                raise
            row = {
                "label": case.label,
                **asdict(case),
                "status": "fail",
                "correctness": False,
                "error": f"{type(error).__name__}: {error}",
            }
            rows.append(row)
            print(f"{case.label:36s} failed: {row['error']}")
        finally:
            torch.cuda.empty_cache()

    timed = [float(row["mean_ms"]) for row in rows if "mean_ms" in row]
    if timed:
        geomean = math.exp(sum(math.log(value) for value in timed) / len(timed))
        print(f"geomean_mean_ms={geomean:.4f} over {len(timed)} cases")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
