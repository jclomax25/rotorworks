#!/usr/bin/env python3
"""
RotorWorks batch automation utility.

This tool wraps the existing fixed-wing and multicopter simulators and adds:

1) Sensitivity / what-if sweeps
2) Constraint-driven sizing search
3) Repeatable scripted multi-run execution

The script drives the existing CLIs (no simulator code changes required).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


SIM_SCRIPT_DEFAULTS = {
    "fixedwing": "fixedwing-power-sim-gui.py",
    "multicopter": "multicopter-power-sim-gui.py",
}

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class RunResult:
    name: str
    args: Dict[str, Any]
    return_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    metrics: Dict[str, Any]
    command: List[str]
    feasible: Optional[bool] = None
    violation_score: Optional[float] = None


def normalize_key(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", s.strip().lower())
    return out.strip("_")


def parse_first_float(text: str) -> Optional[float]:
    m = FLOAT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def smart_cast(value: str) -> Any:
    s = value.strip()
    if not s:
        return s
    lo = s.lower()
    if lo == "true":
        return True
    if lo == "false":
        return False
    # JSON objects/arrays/strings/numbers can be passed explicitly.
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    try:
        if "." in s or "e" in lo:
            return float(s)
        return int(s)
    except Exception:
        return s


def parse_set_items(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set item '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = smart_cast(v)
    return out


def load_json_object(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_cli_args(arg_map: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in arg_map.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            args.extend([flag, ",".join(str(x) for x in value)])
            continue
        args.extend([flag, str(value)])
    return args


def resolve_sim_script(sim: str, override_path: Optional[str]) -> Path:
    if override_path:
        p = Path(override_path).expanduser().resolve()
    else:
        here = Path(__file__).resolve().parent
        p = (here / SIM_SCRIPT_DEFAULTS[sim]).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Simulator script not found: {p}")
    return p


def parse_metrics(sim: str, stdout: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    lines = stdout.splitlines()

    # Generic raw key:value capture.
    # Keep these around even if strict regex extractors miss, so downstream
    # users can still inspect textual outputs in summary files.
    for ln in lines:
        if ":" not in ln:
            continue
        key, val = ln.split(":", 1)
        nk = normalize_key(key)
        if nk:
            metrics[f"raw_{nk}"] = val.strip()

    text = "\n".join(lines)

    def cap_float(pattern: str, key: str, group: int = 1) -> None:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            return
        try:
            metrics[key] = float(m.group(group))
        except Exception:
            pass

    def cap_str(pattern: str, key: str, group: int = 1) -> None:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            return
        metrics[key] = m.group(group).strip()

    # Regex extractors for headline metrics from each simulator's CLI output.
    # These patterns intentionally target stable human-readable summary lines.
    if sim == "fixedwing":
        cap_float(r"Flight Time\s*:\s*([-+]?\d*\.?\d+)\s*min", "flight_time_min")
        cap_float(r"Flight Range\s*:\s*([-+]?\d*\.?\d+)\s*km", "flight_range_km")
        cap_float(r"Best Endurance Speed\s*:\s*([-+]?\d*\.?\d+)\s*m/s", "best_endurance_speed_mps")
        cap_float(r"Best Range Speed\s*:\s*([-+]?\d*\.?\d+)\s*m/s", "best_range_speed_mps")
        cap_float(r"Stall Speed\s*:\s*([-+]?\d*\.?\d+)\s*m/s", "stall_speed_mps")
        cap_float(r"Rate of Climb\s*:\s*([-+]?\d*\.?\d+)\s*m/min", "rate_of_climb_mpm")
        cap_float(r"Total Elec\.\s*Power\s*:\s*([-+]?\d*\.?\d+)\s*W", "total_power_W")
        cap_float(r"Takeoff Ground Roll\s*:\s*([-+]?\d*\.?\d+)\s*m", "takeoff_dist_m")
        cap_float(r"Landing Distance\s*:\s*([-+]?\d*\.?\d+)\s*m", "landing_dist_m")
        cap_float(r"Service Ceiling\s*:\s*([0-9]+)\s*m", "service_ceiling_m")
        cap_str(r"Reserve Status\s*:\s*([A-Z]+)", "reserve_status")
        cap_str(r"Thermal.*\[\s*([A-Z]+)\s*\]", "thermal_status")
    else:
        cap_float(r"Estimated flight time.*:\s*([-+]?\d*\.?\d+)\s*min", "flight_time_min")
        cap_float(r"Estimated flight distance.*:\s*([-+]?\d*\.?\d+)\s*km", "flight_range_km")
        cap_float(r"Best endurance speed.*:\s*([-+]?\d*\.?\d+)\s*m/s", "best_endurance_speed_mps")
        cap_float(r"Best range speed.*:\s*([-+]?\d*\.?\d+)\s*m/s", "best_range_speed_mps")
        cap_float(r"Hover Efficiency\s*:\s*([-+]?\d*\.?\d+)", "hover_efficiency_gW")
        cap_float(r"Figure of Merit.*:\s*([-+]?\d*\.?\d+)", "figure_of_merit")
        cap_float(r"Disk Loading\s*:\s*([-+]?\d*\.?\d+)", "disk_loading_N_m2")
        cap_float(r"SoC / model source\s*:\s*([-+]?\d*\.?\d+)\s*%", "soc_percent")
        cap_str(r"SoC / model source\s*:\s*[-+]?\d*\.?\d+\s*%\s*/\s*(.+)$", "soc_model_source")
        cap_str(r"Reserve Status\s*:\s*([A-Z]+)", "reserve_status")
        cap_str(r"Motor Thermal Status\s*:\s*([A-Za-z]+)", "thermal_status")
        cap_float(r"Potential Power Term\s*:\s*([-+]?\d*\.?\d+)\s*W", "potential_power_W")

    # Fallback numeric extraction for common raw keys.
    # This improves resilience if output wording shifts slightly but still
    # contains a recognizable "Label: value unit" structure.
    fallback_key_map = {
        "raw_flight_time": "flight_time_min",
        "raw_flight_range": "flight_range_km",
        "raw_total_elec_power": "total_power_W",
        "raw_ground_speed": "groundspeed_mps",
    }
    for raw_key, final_key in fallback_key_map.items():
        if final_key in metrics:
            continue
        if raw_key in metrics:
            f = parse_first_float(str(metrics[raw_key]))
            if f is not None:
                metrics[final_key] = f

    return metrics


def run_simulation(
    sim: str,
    sim_script: Path,
    run_name: str,
    run_args: Dict[str, Any],
    timeout_s: float,
    print_command: bool = False,
) -> RunResult:
    cmd = [sys.executable, str(sim_script)] + build_cli_args(run_args)
    if print_command:
        print(f"CMD[{run_name}]: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.time() - t0
    metrics = parse_metrics(sim, proc.stdout)
    return RunResult(
        name=run_name,
        args=dict(run_args),
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_s=elapsed,
        metrics=metrics,
        command=cmd,
    )


def write_run_report(path: Path, rr: RunResult) -> None:
    write_json(
        path,
        {
            "name": rr.name,
            "command": rr.command,
            "args": rr.args,
            "return_code": rr.return_code,
            "elapsed_s": rr.elapsed_s,
            "metrics": rr.metrics,
            "feasible": rr.feasible,
            "violation_score": rr.violation_score,
            "stdout": rr.stdout,
            "stderr": rr.stderr,
        },
    )


def parse_values_list(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("No values parsed.")
    return vals


def build_range_values(start: float, stop: float, step: float) -> List[float]:
    if step == 0:
        raise ValueError("step must not be 0")
    vals: List[float] = []
    if start <= stop and step < 0:
        step = -step
    if start > stop and step > 0:
        step = -step
    v = start
    if step > 0:
        while v <= stop + 1e-12:
            vals.append(round(v, 12))
            v += step
    else:
        while v >= stop - 1e-12:
            vals.append(round(v, 12))
            v += step
    if not vals:
        raise ValueError("Generated empty value range.")
    return vals


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: List[str] = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def format_short_result(rr: RunResult) -> str:
    m = rr.metrics
    t = m.get("flight_time_min", float("nan"))
    d = m.get("flight_range_km", float("nan"))
    reserve = m.get("reserve_status", "n/a")
    th = m.get("thermal_status", "n/a")
    return (
        f"{rr.name}: rc={rr.return_code}, "
        f"time={t:.2f} min, range={d:.2f} km, reserve={reserve}, thermal={th}"
        if isinstance(t, (int, float)) and isinstance(d, (int, float))
        else f"{rr.name}: rc={rr.return_code}"
    )


def numeric_metric_keys(results: Sequence[RunResult]) -> List[str]:
    keys: set[str] = set()
    for rr in results:
        for k, v in rr.metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                keys.add(k)
    return sorted(keys)


def plot_sweep(
    out_dir: Path,
    x_key: str,
    x_vals: List[float],
    results: Sequence[RunResult],
    requested_metrics: Optional[List[str]] = None,
) -> List[Path]:
    if not HAS_MPL:
        return []
    created: List[Path] = []
    default_metrics = [
        "flight_time_min",
        "flight_range_km",
        "best_endurance_speed_mps",
        "best_range_speed_mps",
        "total_power_W",
        "rate_of_climb_mpm",
    ]
    metric_keys = requested_metrics or default_metrics

    for mk in metric_keys:
        y_vals: List[float] = []
        x_ok: List[float] = []
        for xv, rr in zip(x_vals, results):
            y = rr.metrics.get(mk, None)
            if isinstance(y, (int, float)) and math.isfinite(float(y)):
                x_ok.append(float(xv))
                y_vals.append(float(y))
        if len(x_ok) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_ok, y_vals, marker="o")
        ax.set_xlabel(x_key)
        ax.set_ylabel(mk)
        ax.set_title(f"{mk} vs {x_key}")
        ax.grid(True, alpha=0.3)
        p = out_dir / f"plot_{mk}_vs_{x_key}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(p)
    return created


def parse_design_var_spec(spec: str) -> Tuple[str, List[float]]:
    # Supported:
    #   name:start:stop:step
    #   name=v1,v2,v3
    if "=" in spec:
        name, values_txt = spec.split("=", 1)
        return name.strip(), parse_values_list(values_txt)
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid design-var spec '{spec}'. Use name:start:stop:step or name=v1,v2,v3"
        )
    name = parts[0].strip()
    vals = build_range_values(float(parts[1]), float(parts[2]), float(parts[3]))
    return name, vals


def parse_target_list(specs: Sequence[str], mode: str) -> List[Tuple[str, str, Any]]:
    out: List[Tuple[str, str, Any]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Invalid target spec '{s}', expected metric=value")
        k, v = s.split("=", 1)
        key = k.strip()
        if mode in ("min", "max"):
            out.append((mode, key, float(v)))
        else:
            out.append((mode, key, smart_cast(v)))
    return out


def evaluate_constraints(
    metrics: Dict[str, Any],
    constraints: Sequence[Tuple[str, str, Any]],
) -> Tuple[bool, float]:
    """
    Evaluate hard constraints and compute a normalized violation score.

    Violation score is dimensionless and additive:
      - min/max constraints use relative error against target magnitude
      - missing numeric metrics incur a large fixed penalty
      - eq constraints incur unit penalty when mismatched
    """
    feasible = True
    violation = 0.0
    for mode, key, target in constraints:
        value = metrics.get(key, None)
        if mode in ("min", "max"):
            if value is None or not isinstance(value, (int, float)):
                feasible = False
                violation += 1e6
                continue
            v = float(value)
            t = float(target)
            if mode == "min" and v < t:
                feasible = False
                denom = max(abs(t), 1e-6)
                violation += (t - v) / denom
            elif mode == "max" and v > t:
                feasible = False
                denom = max(abs(t), 1e-6)
                violation += (v - t) / denom
        elif mode == "eq":
            if str(value).strip().lower() != str(target).strip().lower():
                feasible = False
                violation += 1.0
    return feasible, violation


def choose_best_result(
    results: Sequence[RunResult],
    objective: str,
) -> Optional[RunResult]:
    """
    Pick best result using objective spec:
      maximize:<metric>
      minimize:<metric>
      target:<metric>:<value>
    Returns None if objective metric is unavailable in all candidates.
    """
    if not results:
        return None

    if objective.startswith("maximize:"):
        key = objective.split(":", 1)[1]
        best = [r for r in results if isinstance(r.metrics.get(key), (int, float))]
        if not best:
            return None
        return max(best, key=lambda r: float(r.metrics[key]))

    if objective.startswith("minimize:"):
        key = objective.split(":", 1)[1]
        best = [r for r in results if isinstance(r.metrics.get(key), (int, float))]
        if not best:
            return None
        return min(best, key=lambda r: float(r.metrics[key]))

    if objective.startswith("target:"):
        parts = objective.split(":")
        if len(parts) != 3:
            raise ValueError("target objective format: target:metric:value")
        key = parts[1]
        target = float(parts[2])
        best = [r for r in results if isinstance(r.metrics.get(key), (int, float))]
        if not best:
            return None
        return min(best, key=lambda r: abs(float(r.metrics[key]) - target))

    raise ValueError(f"Unsupported objective: {objective}")


def build_base_args(args: argparse.Namespace) -> Dict[str, Any]:
    base = load_json_object(args.base_args_file)
    base.update(parse_set_items(args.set or []))
    return base


def command_sweep(args: argparse.Namespace) -> int:
    base_args = build_base_args(args)
    sim_script = resolve_sim_script(args.sim, args.sim_script)

    if args.values:
        sweep_vals = parse_values_list(args.values)
    else:
        if args.start is None or args.stop is None or args.step is None:
            raise ValueError("Provide --values or all of --start/--stop/--step")
        sweep_vals = build_range_values(args.start, args.stop, args.step)

    out_dir = ensure_dir(Path(args.output_dir))
    results: List[RunResult] = []
    rows: List[Dict[str, Any]] = []

    print(f"Running sweep for {args.sim}: {args.sweep_var} over {len(sweep_vals)} points")
    for i, val in enumerate(sweep_vals, start=1):
        run_args = dict(base_args)
        run_args[args.sweep_var] = val
        run_name = f"{args.sweep_var}={val}"
        rr = run_simulation(args.sim, sim_script, run_name, run_args, args.timeout)
        results.append(rr)
        print(f"[{i}/{len(sweep_vals)}] {format_short_result(rr)}")

        txt_path = out_dir / f"run_{i:03d}_{normalize_key(run_name)}.txt"
        write_text(
            txt_path,
            f"# COMMAND ARGS\n{json.dumps(run_args, indent=2)}\n\n"
            f"# STDOUT\n{rr.stdout}\n\n# STDERR\n{rr.stderr}\n",
        )

        # Keep one flat row per run for easy post-processing in pandas/Excel.
        row = {
            "run_name": run_name,
            "sweep_var": args.sweep_var,
            "sweep_value": val,
            "return_code": rr.return_code,
            "elapsed_s": rr.elapsed_s,
        }
        row.update(rr.metrics)
        rows.append(row)

    summary_csv = out_dir / "sweep_summary.csv"
    summary_json = out_dir / "sweep_summary.json"
    write_summary_csv(summary_csv, rows)
    write_json(summary_json, rows)

    plot_paths = plot_sweep(
        out_dir=out_dir,
        x_key=args.sweep_var,
        x_vals=sweep_vals,
        results=results,
        requested_metrics=args.plot_metric,
    )

    print("\nSweep complete.")
    print(f"- CSV summary : {summary_csv}")
    print(f"- JSON summary: {summary_json}")
    if plot_paths:
        print(f"- Plots       : {len(plot_paths)} files")
    elif not HAS_MPL:
        print("- Plots       : skipped (matplotlib unavailable)")
    else:
        print("- Plots       : no plottable numeric metrics found")

    return 0


def command_size(args: argparse.Namespace) -> int:
    base_args = build_base_args(args)
    sim_script = resolve_sim_script(args.sim, args.sim_script)
    out_dir = ensure_dir(Path(args.output_dir))

    design_specs = args.design_var or []
    if not design_specs:
        raise ValueError("At least one --design-var is required for sizing mode.")

    design_vars: List[Tuple[str, List[float]]] = [parse_design_var_spec(s) for s in design_specs]
    names = [n for n, _ in design_vars]
    value_sets = [vals for _, vals in design_vars]
    combos = list(itertools.product(*value_sets))
    if args.max_runs and len(combos) > args.max_runs:
        combos = combos[: args.max_runs]

    constraints: List[Tuple[str, str, Any]] = []
    constraints.extend(parse_target_list(args.target_min or [], "min"))
    constraints.extend(parse_target_list(args.target_max or [], "max"))
    constraints.extend(parse_target_list(args.target_eq or [], "eq"))

    print(f"Running sizing search for {args.sim}")
    print(f"- design variables: {', '.join(names)}")
    print(f"- combinations    : {len(combos)}")
    print(f"- constraints     : {len(constraints)}")
    print(f"- objective       : {args.objective}")

    all_results: List[RunResult] = []
    rows: List[Dict[str, Any]] = []

    # Grid-search all design-point combinations.
    for i, combo in enumerate(combos, start=1):
        run_args = dict(base_args)
        for name, val in zip(names, combo):
            run_args[name] = val
        run_name = ",".join(f"{n}={v}" for n, v in zip(names, combo))
        rr = run_simulation(args.sim, sim_script, run_name, run_args, args.timeout)
        feasible, viol = evaluate_constraints(rr.metrics, constraints)
        rr.feasible = feasible and rr.return_code == 0
        rr.violation_score = viol + (1e6 if rr.return_code != 0 else 0.0)
        all_results.append(rr)
        print(
            f"[{i}/{len(combos)}] {run_name} -> feasible={rr.feasible}, "
            f"viol={rr.violation_score:.4f}, rc={rr.return_code}"
        )

        row = {
            "run_name": run_name,
            "return_code": rr.return_code,
            "elapsed_s": rr.elapsed_s,
            "feasible": rr.feasible,
            "violation_score": rr.violation_score,
        }
        for n, v in zip(names, combo):
            row[n] = v
        row.update(rr.metrics)
        rows.append(row)

    feasible_runs = [r for r in all_results if r.feasible]
    if feasible_runs:
        # Objective selection happens only across feasible candidates.
        best = choose_best_result(feasible_runs, args.objective)
    else:
        # If no feasible design exists, report least-violating candidate.
        best = min(all_results, key=lambda r: float(r.violation_score or 1e12)) if all_results else None

    summary_csv = out_dir / "size_summary.csv"
    summary_json = out_dir / "size_summary.json"
    write_summary_csv(summary_csv, rows)
    write_json(summary_json, rows)

    if best is not None:
        best_path = out_dir / "size_best_run.json"
        write_json(
            best_path,
            {
                "name": best.name,
                "feasible": best.feasible,
                "violation_score": best.violation_score,
                "args": best.args,
                "metrics": best.metrics,
                "return_code": best.return_code,
            },
        )
        print("\nBest sizing result:")
        print(f"- name        : {best.name}")
        print(f"- feasible    : {best.feasible}")
        print(f"- return_code : {best.return_code}")
        print(f"- violation   : {best.violation_score}")
        if "flight_time_min" in best.metrics:
            print(f"- flight_time : {best.metrics['flight_time_min']:.2f} min")
        if "flight_range_km" in best.metrics:
            print(f"- flight_range: {best.metrics['flight_range_km']:.2f} km")
        print(f"- details     : {best_path}")
    else:
        print("No runs executed.")

    print("\nSizing search complete.")
    print(f"- CSV summary : {summary_csv}")
    print(f"- JSON summary: {summary_json}")
    return 0


def command_batch(args: argparse.Namespace) -> int:
    base_args = build_base_args(args)
    sim_script = resolve_sim_script(args.sim, args.sim_script)
    out_dir = ensure_dir(Path(args.output_dir))

    runs_data = load_json_object(args.runs_file)
    runs = runs_data.get("runs", None)
    if not isinstance(runs, list):
        raise ValueError("Runs JSON must contain {'runs': [...]} with list of run objects.")

    rows: List[Dict[str, Any]] = []
    results: List[RunResult] = []
    print(f"Running batch for {args.sim}: {len(runs)} runs")
    for i, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Run #{i} is not an object.")
        name = str(run.get("name", f"run_{i}"))
        overrides = run.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Run '{name}' overrides must be an object.")
        run_args = dict(base_args)
        run_args.update(overrides)
        rr = run_simulation(args.sim, sim_script, name, run_args, args.timeout)
        results.append(rr)
        print(f"[{i}/{len(runs)}] {format_short_result(rr)}")

        txt_path = out_dir / f"batch_{i:03d}_{normalize_key(name)}.txt"
        write_text(
            txt_path,
            f"# COMMAND ARGS\n{json.dumps(run_args, indent=2)}\n\n"
            f"# STDOUT\n{rr.stdout}\n\n# STDERR\n{rr.stderr}\n",
        )

        # One compact row per scripted run plus parsed metrics.
        row = {
            "run_name": name,
            "return_code": rr.return_code,
            "elapsed_s": rr.elapsed_s,
        }
        row.update(rr.metrics)
        rows.append(row)

    summary_csv = out_dir / "batch_summary.csv"
    summary_json = out_dir / "batch_summary.json"
    write_summary_csv(summary_csv, rows)
    write_json(summary_json, rows)

    # Optional simple bar plots across run index for selected metrics.
    if HAS_MPL:
        plot_metrics = args.plot_metric or ["flight_time_min", "flight_range_km"]
        x_labels = [r.name for r in results]
        for mk in plot_metrics:
            ys: List[float] = []
            labels_ok: List[str] = []
            for rr in results:
                v = rr.metrics.get(mk, None)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    labels_ok.append(rr.name)
                    ys.append(float(v))
            if len(ys) < 1:
                continue
            fig, ax = plt.subplots(figsize=(max(7, len(ys) * 0.7), 4))
            ax.bar(range(len(ys)), ys)
            ax.set_xticks(range(len(ys)))
            ax.set_xticklabels(labels_ok, rotation=30, ha="right")
            ax.set_ylabel(mk)
            ax.set_title(f"{mk} by run")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"batch_plot_{mk}.png", dpi=150)
            plt.close(fig)

    print("\nBatch complete.")
    print(f"- CSV summary : {summary_csv}")
    print(f"- JSON summary: {summary_json}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RotorWorks batch/sweep/sizing driver for fixed-wing and multicopter simulators."
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--sim", choices=["fixedwing", "multicopter"], required=True)
        sp.add_argument("--sim-script", default=None, help="Optional path to simulator script.")
        sp.add_argument("--base-args-file", default=None, help="JSON file with base CLI args object.")
        sp.add_argument(
            "--set",
            action="append",
            default=[],
            help="Override/add base arg as key=value (repeatable).",
        )
        sp.add_argument("--timeout", type=float, default=120.0, help="Per-run timeout seconds.")
        sp.add_argument("--output-dir", default="rotorworks-batch-output", help="Output directory.")
        sp.add_argument(
            "--print-commands",
            action="store_true",
            help="Print full simulator CLI command for each run.",
        )

    # Sweep mode
    sp_sweep = sub.add_parser("sweep", help="One-variable sensitivity / what-if sweep.")
    add_common(sp_sweep)
    sp_sweep.add_argument("--sweep-var", required=True, help="Variable name to vary, e.g. weight")
    sp_sweep.add_argument("--values", default=None, help="Comma-separated values, e.g. 1200,1400,1600")
    sp_sweep.add_argument("--start", type=float, default=None)
    sp_sweep.add_argument("--stop", type=float, default=None)
    sp_sweep.add_argument("--step", type=float, default=None)
    sp_sweep.add_argument(
        "--plot-metric",
        action="append",
        default=[],
        help="Metric to plot (repeatable). If omitted, defaults are used.",
    )

    # Sizing mode
    sp_size = sub.add_parser("size", help="Constraint-driven sizing search (grid).")
    add_common(sp_size)
    sp_size.add_argument(
        "--design-var",
        action="append",
        default=[],
        help="Design variable spec, repeatable: name:start:stop:step OR name=v1,v2,v3",
    )
    sp_size.add_argument("--target-min", action="append", default=[], help="Constraint metric=min_value")
    sp_size.add_argument("--target-max", action="append", default=[], help="Constraint metric=max_value")
    sp_size.add_argument("--target-eq", action="append", default=[], help="Constraint metric=string_or_value")
    sp_size.add_argument(
        "--objective",
        default="maximize:flight_time_min",
        help="Objective: maximize:metric | minimize:metric | target:metric:value",
    )
    sp_size.add_argument("--max-runs", type=int, default=0, help="Optional cap on evaluated combinations.")

    # Scripted batch mode
    sp_batch = sub.add_parser("batch", help="Run explicit scripted run list from JSON.")
    add_common(sp_batch)
    sp_batch.add_argument(
        "--runs-file",
        required=True,
        help="JSON object with {'runs':[{'name':'..','overrides':{...}}, ...]}",
    )
    sp_batch.add_argument(
        "--plot-metric",
        action="append",
        default=[],
        help="Metric to bar-plot across runs (repeatable).",
    )

    return p


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    # Normalize output dir with command-specific timestamp.
    ts = time.strftime("%Y%m%d-%H%M%S")
    args.output_dir = str(Path(args.output_dir) / f"{args.command}-{args.sim}-{ts}")

    try:
        if args.command == "sweep":
            return command_sweep(args)
        if args.command == "size":
            return command_size(args)
        if args.command == "batch":
            return command_batch(args)
        raise ValueError(f"Unknown command: {args.command}")
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: simulator timed out after {e.timeout}s", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

