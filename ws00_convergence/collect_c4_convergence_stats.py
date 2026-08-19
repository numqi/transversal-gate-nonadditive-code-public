#!/usr/bin/env python3
"""Collect per-initialization convergence statistics for the ((6,2,3)) C4 search.

Unlike ``draft00.py``, this script retains every optimization result instead of
returning only the best of many random starts.  It is intentionally standalone
so that figure-generation and manuscript files remain untouched.

The experiment uses 100 deterministic random initializations for each target
``[lambda*_target]^2`` in ``{2/3, 0.8, 1.0}``.  Each initialization is first
optimized with L-BFGS-B at ``tol=1e-8``.  If that result has loss below ``1e-5``,
the same result is refined with ``tol=1e-20``.  Scientific success is defined
only after this procedure as final total loss below ``1e-7``.

The raw CSV is checkpointed after every run and can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

# Keep each optimization laptop-safe and reproducible.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numqi
import numpy as np
import opt_einsum
import scipy
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / "c4_convergence_stats_raw.csv"
SUMMARY_JSON = SCRIPT_DIR / "c4_convergence_stats_summary.json"

# numqi's default cache is under macOS Application Support, which may be
# unavailable in sandboxed/non-interactive runs.  Redirect only its disposable
# HDF5 cache; experimental outputs remain beside this script.
NUMQI_DATA_DIR = Path(
    os.environ.get(
        "NUMQI_DATA_DIR", str(Path(tempfile.gettempdir()) / "codex-numqi-data")
    )
)


def _numqi_savepath() -> str:
    NUMQI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(NUMQI_DATA_DIR / "data.hdf5")


numqi._internal.get_savepath = _numqi_savepath

TARGETS = (
    (2.0 / 3.0, "lower_boundary"),
    (0.8, "interior"),
    (1.0, "upper_boundary"),
)
RUNS_PER_TARGET = 100
BASE_SEED = 2026072200
PRELIMINARY_TOL = 1e-8
REFINEMENT_TRIGGER = 1e-5
REFINEMENT_TOL = 1e-20
SUCCESS_THRESHOLD = 1e-7

FIELDNAMES = [
    "target_lambda2",
    "target_region",
    "run_index",
    "seed",
    "num_parameters",
    "preliminary_loss",
    "preliminary_optimizer_success",
    "preliminary_status",
    "preliminary_message",
    "preliminary_nit",
    "preliminary_nfev",
    "preliminary_njev",
    "refined",
    "refinement_loss",
    "refinement_optimizer_success",
    "refinement_status",
    "refinement_message",
    "refinement_nit",
    "refinement_nfev",
    "refinement_njev",
    "final_total_loss",
    "scientific_success",
    "total_nit",
    "total_nfev",
    "total_njev",
    "final_gradient_inf_norm",
    "wall_time_s",
]


class QECC623TransversalGroupModel(torch.nn.Module):
    """Exact C4 model used by ``draft00.py``, reproduced to avoid imports."""

    def __init__(self) -> None:
        super().__init__()
        group = "C4"
        num_qubit = 6
        _, error_torch = numqi.qec.make_pauli_error_list_sparse(
            num_qubit, distance=3, kind="torch-csr01"
        )
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        self.manifold = numqi.manifold.Stiefel(
            2**num_qubit, rank=2, dtype=torch.complex128
        )
        group_order = int(group[1:])
        assert group_order % 2 == 0
        m = group_order // 2
        self.logical_gate = torch.tensor(
            numqi.qec.get_su2_finite_subgroup_generator(f"C{2 * m}")[0],
            dtype=torch.complex128,
        )
        self.manifold_su2 = numqi.manifold.SpecialOrthogonal(
            2, batch_size=num_qubit, dtype=torch.complex128
        )
        self.lambda2_target: torch.Tensor | None = None

        n = num_qubit
        contraction_args = [y for x in range(n) for y in [(2, 2), (n + x, x)]]
        self.contract_expr = opt_einsum.contract_expression(
            [2] * (n + 1),
            list(range(n)) + [2 * n],
            [2] * (n + 1),
            list(range(n, 2 * n)) + [2 * n + 1],
            *contraction_args,
            [2 * n + 1, 2 * n],
        )

    def set_lambda2_target(self, value: float) -> None:
        self.lambda2_target = torch.tensor(float(value), dtype=torch.float64)

    def forward(self) -> torch.Tensor:
        coeff = self.manifold().to(torch.complex128)
        q0 = coeff
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        su2 = self.manifold_su2()
        reshaped = q0.reshape([2] * (self.num_qubit + 1))
        logical_u = self.contract_expr(reshaped, reshaped.conj(), *su2)
        constraints = [
            torch.vdot(q0[:, 0], q0[:, 1]),
            lambda_aij[:, 0, 1],
            lambda_aij[:, 0, 0].real - lambda_aij[:, 1, 1].real,
            logical_u - self.logical_gate,
        ]
        if self.lambda2_target is not None:
            signature_coefficients = (
                lambda_aij[:, 0, 0] + lambda_aij[:, 1, 1]
            ).real / 2
            norm2 = torch.dot(signature_coefficients, signature_coefficients)
            constraints.append(norm2 - self.lambda2_target)
        return sum(
            torch.vdot(value.reshape(-1), value.reshape(-1)).real
            for value in constraints
        )


def package_version(distribution: str, module: object) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scalar_attr(result: object, name: str, default: int = 0) -> int:
    value = getattr(result, name, default)
    return default if value is None else int(value)


def result_message(result: object) -> str:
    return str(getattr(result, "message", ""))


def gradient_inf_norm(result: object) -> float:
    jac = getattr(result, "jac", None)
    if jac is None:
        return math.nan
    values = np.asarray(jac, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else math.nan


def seed_for(target_index: int, run_index: int) -> int:
    return BASE_SEED + 1000 * target_index + run_index


def _worker_init() -> None:
    """Limit numerical-library thread pools inside each spawned worker."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_one(
    target_index: int, target_lambda2: float, region: str, run_index: int
) -> dict[str, object]:
    seed = seed_for(target_index, run_index)
    rng = np.random.default_rng(seed)
    model = QECC623TransversalGroupModel()
    model.set_lambda2_target(target_lambda2)
    num_parameters = int(numqi.optimize.get_model_flat_parameter(model).size)
    theta0 = rng.uniform(-1.0, 1.0, size=num_parameters)

    started = time.perf_counter()
    preliminary = numqi.optimize.minimize(
        model,
        theta0=theta0,
        num_repeat=1,
        tol=PRELIMINARY_TOL,
        early_stop_threshold=REFINEMENT_TRIGGER,
        print_freq=0,
        print_every_round=0,
    )
    if preliminary is None:
        raise RuntimeError("Unexpected empty preliminary optimization result")

    refined = float(preliminary.fun) < REFINEMENT_TRIGGER
    refinement = None
    final = preliminary
    if refined:
        refinement = numqi.optimize.minimize(
            model,
            theta0=np.asarray(preliminary.x),
            num_repeat=1,
            tol=REFINEMENT_TOL,
            print_freq=0,
            print_every_round=0,
        )
        if refinement is None:
            raise RuntimeError("Unexpected empty refinement optimization result")
        final = refinement
    wall_time_s = time.perf_counter() - started

    pre_nit = scalar_attr(preliminary, "nit")
    pre_nfev = scalar_attr(preliminary, "nfev")
    pre_njev = scalar_attr(preliminary, "njev")
    ref_nit = scalar_attr(refinement, "nit") if refinement is not None else 0
    ref_nfev = scalar_attr(refinement, "nfev") if refinement is not None else 0
    ref_njev = scalar_attr(refinement, "njev") if refinement is not None else 0
    final_loss = float(final.fun)

    return {
        "target_lambda2": format(target_lambda2, ".17g"),
        "target_region": region,
        "run_index": run_index,
        "seed": seed,
        "num_parameters": num_parameters,
        "preliminary_loss": format(float(preliminary.fun), ".17g"),
        "preliminary_optimizer_success": bool(preliminary.success),
        "preliminary_status": scalar_attr(preliminary, "status", -1),
        "preliminary_message": result_message(preliminary),
        "preliminary_nit": pre_nit,
        "preliminary_nfev": pre_nfev,
        "preliminary_njev": pre_njev,
        "refined": refined,
        "refinement_loss": (
            format(float(refinement.fun), ".17g")
            if refinement is not None
            else ""
        ),
        "refinement_optimizer_success": (
            bool(refinement.success) if refinement is not None else ""
        ),
        "refinement_status": (
            scalar_attr(refinement, "status", -1) if refinement is not None else ""
        ),
        "refinement_message": (
            result_message(refinement) if refinement is not None else ""
        ),
        "refinement_nit": ref_nit,
        "refinement_nfev": ref_nfev,
        "refinement_njev": ref_njev,
        "final_total_loss": format(final_loss, ".17g"),
        "scientific_success": final_loss < SUCCESS_THRESHOLD,
        "total_nit": pre_nit + ref_nit,
        "total_nfev": pre_nfev + ref_nfev,
        "total_njev": pre_njev + ref_njev,
        "final_gradient_inf_norm": format(gradient_inf_norm(final), ".17g"),
        "wall_time_s": format(wall_time_s, ".9g"),
    }


def read_existing_rows() -> list[dict[str, str]]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(
                f"Existing CSV schema does not match this script: {reader.fieldnames}"
            )
        return list(reader)


def append_row(row: dict[str, object]) -> None:
    is_new = not RAW_CSV.exists()
    with RAW_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Not a serialized Boolean: {value!r}")


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {
        "min": float(np.min(array)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "max": float(np.max(array)),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {"lower": center - half_width, "upper": center + half_width}


def optimization_subset_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "runs": len(rows),
        "final_total_loss": percentile_summary(
            float(row["final_total_loss"]) for row in rows
        ),
        "total_iterations": percentile_summary(
            float(row["total_nit"]) for row in rows
        ),
        "total_function_evaluations": percentile_summary(
            float(row["total_nfev"]) for row in rows
        ),
        "wall_time_s": percentile_summary(float(row["wall_time_s"]) for row in rows),
    }


def build_summary(
    rows: list[dict[str, str]],
    elapsed_s: float,
    workers: int,
    estimated_prior_wall_s: float,
) -> dict[str, object]:
    expected = len(TARGETS) * RUNS_PER_TARGET
    keys = [
        (float(row["target_lambda2"]), int(row["run_index"])) for row in rows
    ]
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    if len(set(keys)) != expected:
        raise RuntimeError("Duplicate target/run keys found")

    target_summaries = []
    for target, region in TARGETS:
        selected = [
            row
            for row in rows
            if math.isclose(float(row["target_lambda2"]), target, abs_tol=1e-15)
        ]
        if len(selected) != RUNS_PER_TARGET:
            raise RuntimeError(
                f"Target {target:.17g}: expected {RUNS_PER_TARGET}, found {len(selected)}"
            )
        if {int(row["run_index"]) for row in selected} != set(
            range(RUNS_PER_TARGET)
        ):
            raise RuntimeError(f"Target {target:.17g}: run indices are incomplete")
        for row in selected:
            if row["target_region"] != region:
                raise RuntimeError(f"Target {target:.17g}: inconsistent region label")
            expected_seed = seed_for(TARGETS.index((target, region)), int(row["run_index"]))
            if int(row["seed"]) != expected_seed:
                raise RuntimeError(f"Target {target:.17g}: unexpected seed")
            loss = float(row["final_total_loss"])
            if not math.isfinite(loss) or loss < 0:
                raise RuntimeError(f"Target {target:.17g}: invalid final loss {loss}")
            if parse_bool(row["scientific_success"]) != (loss < SUCCESS_THRESHOLD):
                raise RuntimeError(f"Target {target:.17g}: success flag mismatch")

        successes = sum(parse_bool(row["scientific_success"]) for row in selected)
        refined_count = sum(parse_bool(row["refined"]) for row in selected)
        successful_rows = [
            row for row in selected if parse_bool(row["scientific_success"])
        ]
        unsuccessful_rows = [
            row for row in selected if not parse_bool(row["scientific_success"])
        ]
        losses = [float(row["final_total_loss"]) for row in selected]
        total_nit = [float(row["total_nit"]) for row in selected]
        total_nfev = [float(row["total_nfev"]) for row in selected]
        runtimes = [float(row["wall_time_s"]) for row in selected]
        target_summaries.append(
            {
                "target_lambda2": target,
                "region": region,
                "runs": RUNS_PER_TARGET,
                "successes": successes,
                "success_fraction": successes / RUNS_PER_TARGET,
                "success_fraction_wilson_95": wilson_interval(
                    successes, RUNS_PER_TARGET
                ),
                "refined_runs": refined_count,
                "final_total_loss": percentile_summary(losses),
                "total_iterations": percentile_summary(total_nit),
                "total_function_evaluations": percentile_summary(total_nfev),
                "wall_time_s": percentile_summary(runtimes),
                "optimizer_successes": sum(
                    parse_bool(row["preliminary_optimizer_success"])
                    for row in selected
                ),
                "successful_runs": optimization_subset_summary(successful_rows),
                "unsuccessful_runs": optimization_subset_summary(unsuccessful_rows),
                "loss_separation": {
                    "largest_successful_loss": max(
                        float(row["final_total_loss"]) for row in successful_rows
                    ),
                    "smallest_unsuccessful_loss": min(
                        float(row["final_total_loss"]) for row in unsuccessful_rows
                    ),
                },
            }
        )

    return {
        "experiment": "((6,2,3)) C4 manifold-optimization convergence statistics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "procedure": {
            "targets_lambda2": [target for target, _ in TARGETS],
            "target_regions": [region for _, region in TARGETS],
            "runs_per_target": RUNS_PER_TARGET,
            "base_seed": BASE_SEED,
            "seed_formula": "base_seed + 1000 * target_index + run_index",
            "initialization": "numpy Generator(PCG64), uniform[-1, 1)",
            "optimizer": "L-BFGS-B via numqi.optimize.minimize/scipy.optimize.minimize",
            "preliminary_tolerance": PRELIMINARY_TOL,
            "refinement_trigger": REFINEMENT_TRIGGER,
            "refinement_tolerance": REFINEMENT_TOL,
            "scientific_success_threshold": SUCCESS_THRESHOLD,
            "threads_per_process": 1,
            "execution": (
                "sequential" if workers == 1 else f"spawn ProcessPoolExecutor ({workers} workers)"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "numqi": package_version("numqi", numqi),
            "opt_einsum": package_version("opt_einsum", opt_einsum),
        },
        "completed_rows": len(rows),
        "elapsed_this_invocation_s": elapsed_s,
        "estimated_total_collection_wall_s": estimated_prior_wall_s + elapsed_s,
        "estimated_total_collection_wall_note": (
            "For rows completed before the final invocation, this estimate uses the "
            "sum of their per-run optimizer wall times; it excludes earlier process "
            "startup and checkpoint-I/O overhead. The final parallel invocation is "
            "timed directly."
        ),
        "total_recorded_optimization_time_s": sum(
            float(row["wall_time_s"]) for row in rows
        ),
        "targets": target_summaries,
        "validation": {
            "expected_rows": expected,
            "unique_target_run_keys": len(set(keys)),
            "all_losses_finite_nonnegative": True,
            "all_success_flags_recomputed": True,
            "all_run_indices_complete": True,
            "all_seeds_match_formula": True,
        },
        "limitations": [
            "These are empirical success frequencies for one deterministic set of 100 random initializations per target, not probabilities with a known sampling distribution over all possible initializations.",
            "The three targets test the lower boundary, one interior point, and the upper boundary of the analytically realized interval; they do not characterize every target in the interval.",
            "Optimizer termination status and scientific success are different: scientific success is determined only by final total loss < 1e-7.",
            "A failure to reach the threshold is an optimization non-finding, not evidence that a code does not exist.",
            "Per-run wall times are hardware- and load-dependent. The first 18 lower-boundary runs were collected sequentially before the resumable experiment switched to four workers; iteration and evaluation counts are more comparable across targets than elapsed seconds.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=None,
        help="Run at most this many missing target/run pairs (for smoke tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of spawned worker processes; use 1 for sequential execution.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate the completed raw CSV and rebuild JSON without optimizing.",
    )
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 0:
        parser.error("--max-new-runs must be nonnegative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It can only be set once per Python process; this script is normally fresh.
        pass

    existing = read_existing_rows()
    completed = {
        (float(row["target_lambda2"]), int(row["run_index"])) for row in existing
    }
    if len(completed) != len(existing):
        raise RuntimeError("Existing CSV contains duplicate target/run rows")

    if args.summary_only:
        if not SUMMARY_JSON.exists():
            raise RuntimeError("--summary-only requires an existing summary for runtime metadata")
        with SUMMARY_JSON.open(encoding="utf-8") as handle:
            previous_summary = json.load(handle)
        previous_invocation_s = float(previous_summary["elapsed_this_invocation_s"])
        previous_total_wall_s = float(
            previous_summary["estimated_total_collection_wall_s"]
        )
        summary = build_summary(
            existing,
            elapsed_s=previous_invocation_s,
            workers=args.workers,
            estimated_prior_wall_s=previous_total_wall_s - previous_invocation_s,
        )
        with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=False)
            handle.write("\n")
        print(
            f"Validated {len(existing)} existing rows and rebuilt "
            f"{SUMMARY_JSON.name} without new optimizations."
        )
        return

    missing: list[tuple[int, float, str, int]] = []
    for target_index, (target, region) in enumerate(TARGETS):
        for run_index in range(RUNS_PER_TARGET):
            if (target, run_index) in completed:
                continue
            missing.append((target_index, target, region, run_index))
    if args.max_new_runs is not None:
        missing = missing[: args.max_new_runs]

    # Prime numqi's disposable HDF5 cache before workers access it concurrently.
    if missing:
        QECC623TransversalGroupModel()

    estimated_prior_wall_s = sum(float(row["wall_time_s"]) for row in existing)
    started = time.perf_counter()
    new_count = 0

    def record(row: dict[str, object]) -> None:
        nonlocal new_count
        append_row(row)
        new_count += 1
        print(
            f"target={float(row['target_lambda2']):.12g} "
            f"({row['target_region']}) run={int(row['run_index']):03d} "
            f"seed={row['seed']} loss={float(row['final_total_loss']):.6e} "
            f"success={row['scientific_success']} time={float(row['wall_time_s']):.2f}s",
            flush=True,
        )

    if args.workers == 1:
        for job in missing:
            record(run_one(*job))
    else:
        mp_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
            initializer=_worker_init,
        ) as executor:
            future_to_job = {
                executor.submit(run_one, *job): job for job in missing
            }
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    record(future.result())
                except Exception as error:
                    raise RuntimeError(f"Optimization failed for job {job}") from error

    if args.max_new_runs is not None and len(read_existing_rows()) < len(TARGETS) * RUNS_PER_TARGET:
        print(f"Stopped after {new_count} new run(s); CSV checkpoint is resumable.")
        return

    rows = read_existing_rows()
    elapsed_s = time.perf_counter() - started
    summary = build_summary(
        rows,
        elapsed_s,
        workers=args.workers,
        estimated_prior_wall_s=estimated_prior_wall_s,
    )
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(
        f"Validated {len(rows)} rows. Wrote {RAW_CSV.name} and {SUMMARY_JSON.name}."
    )


if __name__ == "__main__":
    main()
