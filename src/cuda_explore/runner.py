"""Submit the CUDA test suites to Slurm."""

import argparse
import shlex
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from .paths import LOG_DIR, PROJECT_ROOT
from .run_tests import NO_COLOR, OPTIONS

GPU_CONSTRAINT = "gtx_2080_ti|titan_rtx"
JOB_TIME_HOURS = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", choices=sorted(OPTIONS - {NO_COLOR}), metavar="suite")
    parser.add_argument(NO_COLOR, action="store_true", help=argparse.SUPPRESS)
    parser.epilog = f"Suites: {', '.join(sorted(OPTIONS - {NO_COLOR}))}"
    suites = list(dict.fromkeys(parser.parse_intermixed_args(argv).suites))
    job_name = "check_" + "_".join(suites)
    check_command = shlex.join(["make", "check", *suites, f"CHECK_ARGS={NO_COLOR}"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=LOG_DIR, prefix="check_", suffix=".sh", delete=False
    ) as script:
        script.write(f"#!/bin/bash\nexec {check_command}\n")
        job_script = Path(script.name)

    log_pattern = LOG_DIR / f"{job_name}_%j.out"
    try:
        result = subprocess.run(
            [
                "sbatch", "--parsable", f"--job-name={job_name}", "--partition=gpus",
                "--gres=gpu:1", f"--time={JOB_TIME_HOURS}:00:00", "--mem=16G",
                "--cpus-per-task=8", f"--constraint={GPU_CONSTRAINT}",
                f"--chdir={PROJECT_ROOT}", f"--output={log_pattern}", str(job_script),
            ],
            capture_output=True, text=True,
        )
    except OSError as error:
        print(f"ERROR: unable to submit the job: {error}")
        return 1
    if result.returncode != 0:
        print(f"ERROR: sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
        return 1

    job_id = result.stdout.strip().split(";", 1)[0]
    print(f"Job ID: {job_id}")
    print(f"Job script: {job_script}")
    print(f"Log: {str(log_pattern).replace('%j', job_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
