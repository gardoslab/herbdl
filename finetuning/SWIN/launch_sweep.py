#!/usr/bin/env python3
"""
Launch a sweep of training jobs from a single base config + a list of per-job overrides.

Usage:
  python launch_sweep.py --base configs/swin_base_pretrained_linear.yml --sweep my_sweep.yml
  python launch_sweep.py --base configs/swin_base_pretrained_linear.yml --sweep my_sweep.yml --dry-run

Sweep YAML format:
  qsub:                          # optional — overrides defaults below
    h_rt: "48:00:00"
    gpus: 1
    gpu_c: 7.0
    pe: "omp 8"

  experiments:
    - training.seed: 0
      custom.run_id: my_run_seed0_052026
      training.output_dir: /path/to/output/SEED0
      training.logging_dir: /path/to/output/SEED0

    - training.seed: 1
      custom.run_id: my_run_seed1_052026
      training.output_dir: /path/to/output/SEED1
      training.logging_dir: /path/to/output/SEED1
"""

import argparse
import os
import subprocess
import sys

import yaml

DEFAULTS = {
    "h_rt":  "48:00:00",
    "pe":    "omp 8",
    "P":     "herbdl",
    "gpus":  1,
    "gpu_c": 8.0,
    "M":     "faridkar@bu.edu",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_advanced.sh")


def build_set_args(overrides: dict) -> str:
    """Convert {key: value} overrides to '--set key=value ...' string."""
    parts = []
    for key, value in overrides.items():
        parts.append(f"--set {key}={value}")
    return " ".join(parts)


def submit(base_config: str, overrides: dict, qsub_opts: dict, dry_run: bool) -> None:
    run_id = overrides.get("custom.run_id", "unknown")
    job_name = run_id.upper().replace("-", "_")[:15]   # qsub name limit
    set_args = build_set_args(overrides)

    cmd = [
        "qsub",
        "-l", f"h_rt={qsub_opts['h_rt']}",
        "-pe", qsub_opts["pe"],
        "-P",  qsub_opts["P"],
        "-l",  f"gpus={qsub_opts['gpus']}",
        "-l",  f"gpu_c={qsub_opts['gpu_c']}",
        "-m",  "beas",
        "-M",  qsub_opts["M"],
        "-N",  job_name,
        "-v",  f"CONFIG_FILE={base_config},SET_ARGS={set_args}",
        TRAIN_SCRIPT,
    ]

    print(f"[{'DRY RUN' if dry_run else 'SUBMIT'}] {run_id}")
    print(f"  overrides : {set_args or '(none)'}")
    print(f"  job name  : {job_name}")
    print(f"  command   : {' '.join(cmd)}")
    print()

    if not dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"  {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Launch a sweep of qsub training jobs")
    parser.add_argument("--base",    required=True, help="Base config YAML path")
    parser.add_argument("--sweep",   required=True, help="Sweep spec YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    if not os.path.isfile(args.base):
        sys.exit(f"Base config not found: {args.base}")
    if not os.path.isfile(args.sweep):
        sys.exit(f"Sweep file not found: {args.sweep}")

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)

    experiments = sweep.get("experiments", [])
    if not experiments:
        sys.exit("No experiments found in sweep file.")

    qsub_opts = {**DEFAULTS, **sweep.get("qsub", {})}

    print(f"Base config : {args.base}")
    print(f"Experiments : {len(experiments)}")
    print(f"qsub opts   : {qsub_opts}")
    print()

    for exp in experiments:
        # Convert all values to strings for --set compatibility
        overrides = {str(k): str(v) for k, v in exp.items()}
        submit(args.base, overrides, qsub_opts, args.dry_run)

    print("Done." if not args.dry_run else "Dry run complete — nothing submitted.")


if __name__ == "__main__":
    main()
