"""GPU-packing orchestrator for ``evaluate.cli``.

Runs a list of ``evaluate.cli`` command lines as concurrent subprocesses, packing the
single local GPU up to a target memory utilization without overshooting it (a small card
OOMs easily). Heuristic: launch the largest dataset first, let it settle so its GPU
footprint is known, then keep adding smaller jobs that fit in the remaining headroom.

Jobs are read from a Python file exposing ``JOBS: list[str]`` (see ``jobs.example.py``).

    uv run --active python orchestrator.py --jobs-file jobs.py
    uv run --active python orchestrator.py --jobs-file jobs.example.py --dry-run
"""

from __future__ import annotations

import json
import os
import runpy
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

# Approximate row counts per dataset, used as a proxy for GPU memory footprint.
# Keys match the SupportedDatasets enum names in data/__init__.py. Update here when
# datasets change.
DATASET_ROWS: dict[str, int] = {
    "GSK_HEPG2": 13327,
    "GSK_HEPG2_SAMPLED_HALF": 6664,
    "DB_MALARIA": 5972,
    "DB_HEPG2": 5972,
    "SINGLE_TARGET_TBA": 519,
    "DUAL_TARGET_TBA": 519,
    "PK": 190,
}

# Datasets we don't recognise sort last (run them only once there's known headroom).
UNKNOWN_DATASET_ROWS = 0

# Models that don't touch the GPU; their predicted GPU footprint is ~0.
CPU_ONLY_MODELS = {"xgboost"}


@dataclass
class OrchestratorConfig:
    """Configuration for the GPU-packing orchestrator."""

    jobs_file: str
    """Path to a Python file exposing ``JOBS: list[str]`` of evaluate.cli command lines."""

    mem_threshold: float = 0.80
    """Target GPU memory utilization. New jobs launch only while usage stays under this fraction."""

    settle_seconds: float = 600.0
    """Minimum seconds between launches so a new job allocates GPU memory before we measure headroom."""

    poll_interval: float = 30.0
    """Seconds between scheduler ticks (reap finished jobs, probe GPU, maybe launch)."""

    max_concurrent: int = 8
    """Hard cap on concurrently running subprocesses (guards against CPU oversubscription)."""

    max_cpu_concurrent: int = 1
    """Hard cap on concurrent CPU-only jobs (e.g. xgboost). They predict ~0 GPU so they would
    otherwise always 'fit' and crowd the box; each spawns its own Ray + thread pool."""

    mem_margin_mib: float = 300.0
    """Safety pad added to each job's predicted memory before deciding it fits."""

    mem_ema_alpha: float = 0.5
    """EMA weight for updating the per-row memory estimate from each observed launch."""

    log_dir: str = "orchestrator_runs"
    """Parent directory for per-run log folders."""

    dry_run: bool = False
    """Print the size-sorted launch order and exit without running anything."""


@dataclass
class Job:
    """A single evaluate.cli invocation tracked through its lifecycle."""

    idx: int
    command: str
    argv: list[str]
    dataset: str | None
    rows: int
    cpu_only: bool

    # Runtime state.
    status: str = "pending"  # pending | running | done | failed
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log_path: Path | None = None
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    used_before_launch_mib: float | None = None

    @property
    def job_id(self) -> str:
        ds = self.dataset or "unknown"
        return f"job{self.idx:02d}_{ds}"


def parse_command(command: str) -> list[str]:
    """Split a command string into argv, tolerating stray ``\\`` line-continuation tokens."""
    tokens = shlex.split(command)
    return [t for t in tokens if t not in ("\\",)]


def extract_dataset(argv: list[str]) -> str | None:
    """Return the value following ``--dataset`` (or ``--dataset=X``) in argv, if present."""
    for i, tok in enumerate(argv):
        if tok == "--dataset" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--dataset="):
            return tok.split("=", 1)[1]
    return None


def extract_model(argv: list[str]) -> str | None:
    """Return the model subcommand (first token after the ``evaluate.cli`` module token)."""
    for i, tok in enumerate(argv):
        if tok == "evaluate.cli" or tok.endswith("evaluate.cli"):
            if i + 1 < len(argv):
                return argv[i + 1]
    # Fallback: first token that isn't an option or a known launcher word.
    skip = {"uv", "run", "--active", "python", "python3", "-m", "evaluate.cli"}
    for tok in argv:
        if tok in skip or tok.startswith("-"):
            continue
        return tok
    return None


def build_jobs(commands: list[str]) -> list[Job]:
    """Parse command strings into Jobs sorted by dataset size, largest first."""
    jobs: list[Job] = []
    for idx, command in enumerate(commands):
        argv = parse_command(command)
        dataset = extract_dataset(argv)
        model = extract_model(argv)
        rows = (
            DATASET_ROWS.get(dataset, UNKNOWN_DATASET_ROWS)
            if dataset
            else UNKNOWN_DATASET_ROWS
        )
        cpu_only = model in CPU_ONLY_MODELS
        jobs.append(
            Job(
                idx=idx,
                command=command,
                argv=argv,
                dataset=dataset,
                rows=rows,
                cpu_only=cpu_only,
            )
        )
    # Largest dataset first; stable on original order for ties.
    jobs.sort(key=lambda j: (-j.rows, j.idx))
    return jobs


def load_jobs_file(path: str) -> list[str]:
    """Load the ``JOBS`` list from a Python file."""
    namespace = runpy.run_path(path)
    if "JOBS" not in namespace:
        raise SystemExit(f"{path} does not define a JOBS variable")
    jobs = namespace["JOBS"]
    if not isinstance(jobs, list) or not all(isinstance(j, str) for j in jobs):
        raise SystemExit(f"JOBS in {path} must be a list[str]")
    if not jobs:
        raise SystemExit(f"JOBS in {path} is empty")
    return jobs


def gpu_mem() -> tuple[float, float]:
    """Return (used_mib, total_mib) for GPU 0 via nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    )
    used_str, total_str = out.strip().splitlines()[0].split(",")
    return float(used_str), float(total_str)


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig, jobs: list[Job], run_dir: Path):
        self.cfg = cfg
        self.jobs = jobs
        self.run_dir = run_dir
        self.pending = list(jobs)  # already size-sorted
        self.running: list[Job] = []
        self.mem_per_row: float | None = None
        self.last_launch_ts: float = 0.0
        self.baseline_used: float = 0.0
        self.total_mib: float = 0.0
        self.peak_used: float = 0.0

    # --- memory model ---------------------------------------------------------

    def predict_mem(self, job: Job) -> float:
        """Predicted GPU memory (MiB) for a job, including the safety margin."""
        if job.cpu_only or job.rows <= 0:
            return 0.0
        if self.mem_per_row is None:
            # No calibration yet: assume it could be as large as the whole budget so we
            # only ever launch one uncalibrated GPU job at a time.
            return self.total_mib * self.cfg.mem_threshold
        return self.mem_per_row * job.rows + self.cfg.mem_margin_mib

    def update_mem_per_row(self, job: Job, used_now: float) -> None:
        """Refine the per-row estimate from a settled launch's observed memory delta."""
        if job.cpu_only or job.rows <= 0 or job.used_before_launch_mib is None:
            return
        delta = used_now - job.used_before_launch_mib
        if delta <= 0:
            return
        observed = delta / job.rows
        if self.mem_per_row is None:
            self.mem_per_row = observed
        else:
            a = self.cfg.mem_ema_alpha
            self.mem_per_row = a * observed + (1 - a) * self.mem_per_row

    # --- subprocess lifecycle -------------------------------------------------

    def launch(self, job: Job, used_now: float) -> None:
        job.log_path = self.run_dir / f"{job.job_id}.log"
        job_tmp = self.run_dir / f"{job.job_id}_tmp"
        job_tmp.mkdir(exist_ok=True)
        env = os.environ.copy()
        # Isolate each run's Ray instance to avoid temp/port collisions (cli.py ray.init).
        env["TMPDIR"] = str(job_tmp)
        env["RAY_TMPDIR"] = str(job_tmp)
        log_file = open(job.log_path, "w")
        job.proc = subprocess.Popen(
            job.argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job.status = "running"
        job.started_at = time.time()
        job.used_before_launch_mib = used_now
        self.running.append(job)
        self.pending.remove(job)
        self.last_launch_ts = time.time()
        print(
            f"[launch] {job.job_id} (rows={job.rows}, "
            f"predicted={self.predict_mem(job):.0f} MiB) -> {job.log_path}"
        )

    def reap(self) -> None:
        """Collect finished subprocesses and classify their exit."""
        still_running: list[Job] = []
        for job in self.running:
            assert job.proc is not None
            rc = job.proc.poll()
            if rc is None:
                still_running.append(job)
                continue
            job.returncode = rc
            job.finished_at = time.time()
            if rc == 0 and not self._log_has_oom(job):
                job.status = "done"
            else:
                job.status = "failed"
            reason = "OOM" if self._log_has_oom(job) else f"rc={rc}"
            mark = "done" if job.status == "done" else f"FAILED ({reason})"
            print(f"[reap]   {job.job_id} {mark}")
        self.running = still_running

    @staticmethod
    def _log_has_oom(job: Job) -> bool:
        if job.log_path is None or not job.log_path.exists():
            return False
        try:
            text = job.log_path.read_text(errors="replace")
        except OSError:
            return False
        return "CUDA out of memory" in text or "torch.OutOfMemoryError" in text

    # --- scheduling -----------------------------------------------------------

    def pick_job_that_fits(self, used_now: float) -> Job | None:
        """Largest pending job whose predicted memory fits under the threshold.

        CPU-only jobs predict ~0 GPU so they always fit; they're additionally gated by
        ``max_cpu_concurrent`` so they don't crowd the box while the GPU is saturated.
        """
        cap = self.total_mib * self.cfg.mem_threshold
        running_cpu = sum(1 for j in self.running if j.cpu_only)
        for job in self.pending:  # already largest-first
            if job.cpu_only and running_cpu >= self.cfg.max_cpu_concurrent:
                continue
            predicted = self.predict_mem(job)
            if used_now + predicted <= cap:
                return job
        return None

    def run(self) -> None:
        self.baseline_used, self.total_mib = gpu_mem()
        self.peak_used = self.baseline_used
        print(
            f"GPU: {self.baseline_used:.0f}/{self.total_mib:.0f} MiB used at start; "
            f"threshold {self.cfg.mem_threshold:.0%} "
            f"({self.total_mib * self.cfg.mem_threshold:.0f} MiB)"
        )

        try:
            while self.pending or self.running:
                self.reap()
                used_now, self.total_mib = gpu_mem()
                self.peak_used = max(self.peak_used, used_now)

                # Refine the per-row estimate from jobs that have had time to settle.
                for job in self.running:
                    if (
                        job.started_at is not None
                        and time.time() - job.started_at >= self.cfg.settle_seconds
                        and job.used_before_launch_mib is not None
                    ):
                        self.update_mem_per_row(job, used_now)
                        job.used_before_launch_mib = None  # consume once

                settled = time.time() - self.last_launch_ts >= self.cfg.settle_seconds
                if (
                    self.pending
                    and settled
                    and len(self.running) < self.cfg.max_concurrent
                ):
                    job = self.pick_job_that_fits(used_now)
                    if job is not None:
                        self.launch(job, used_now)
                        continue  # re-tick immediately to reflect the new launch

                self._print_status(used_now)
                if not self.pending and not self.running:
                    break
                time.sleep(self.cfg.poll_interval)
        except KeyboardInterrupt:
            print("\n[interrupt] terminating running jobs...")
            self._terminate_all()
        finally:
            self._write_summary()

    def _terminate_all(self) -> None:
        for job in self.running:
            if job.proc is not None and job.proc.poll() is None:
                job.proc.terminate()
        deadline = time.time() + 15
        for job in self.running:
            if job.proc is None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                job.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                job.proc.kill()
            job.status = "failed"
            job.returncode = job.proc.returncode

    def _print_status(self, used_now: float) -> None:
        counts = {s: 0 for s in ("pending", "running", "done", "failed")}
        for job in self.jobs:
            counts[job.status] += 1
        mpr = f"{self.mem_per_row:.3f}" if self.mem_per_row is not None else "n/a"
        print(
            f"[status] GPU {used_now:.0f}/{self.total_mib:.0f} MiB | "
            f"mem/row={mpr} | running={counts['running']} pending={counts['pending']} "
            f"done={counts['done']} failed={counts['failed']}"
        )

    def _write_summary(self) -> None:
        summary = {
            "run_dir": str(self.run_dir),
            "gpu_total_mib": self.total_mib,
            "baseline_used_mib": self.baseline_used,
            "peak_used_mib": self.peak_used,
            "mem_per_row_final": self.mem_per_row,
            "jobs": [
                {
                    "job_id": j.job_id,
                    "command": j.command,
                    "dataset": j.dataset,
                    "rows": j.rows,
                    "cpu_only": j.cpu_only,
                    "status": j.status,
                    "returncode": j.returncode,
                    "wall_seconds": (
                        round(j.finished_at - j.started_at, 1)
                        if j.started_at and j.finished_at
                        else None
                    ),
                    "log": str(j.log_path) if j.log_path else None,
                }
                for j in self.jobs
            ],
        }
        path = self.run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {path}")
        for j in sorted(self.jobs, key=lambda j: j.idx):
            print(f"  {j.job_id:<28} {j.status:<8} rc={j.returncode}")


def print_dry_run(jobs: list[Job]) -> None:
    print("Dry run — launch order (largest dataset first):\n")
    for pos, job in enumerate(jobs, 1):
        ds = job.dataset or "unknown"
        tag = " [cpu-only]" if job.cpu_only else ""
        print(f"  {pos:>2}. {job.job_id:<28} rows={job.rows:<6} {ds}{tag}")
        print(f"      {job.command}")
    print("\nNo processes were launched (dry run).")


def main(cfg: OrchestratorConfig) -> None:
    commands = load_jobs_file(cfg.jobs_file)
    jobs = build_jobs(commands)

    if cfg.dry_run:
        print_dry_run(jobs)
        return

    # Fail fast if we can't read the GPU.
    try:
        gpu_mem()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not query GPU via nvidia-smi: {exc}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}\n")

    orch = Orchestrator(cfg, jobs, run_dir)
    orch.run()


if __name__ == "__main__":
    # Ensure Ctrl-C raises KeyboardInterrupt in the main loop.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main(tyro.cli(OrchestratorConfig))
