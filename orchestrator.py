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
import random
import runpy
import shlex
import shutil
import signal
import subprocess
import tempfile
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

# CLI model token -> the tag evaluate.cli stamps on its wandb run. Note the mapping isn't
# identity: deltaprop runs are tagged "deltaprop-btlh2" (see evaluate/cli.py).
MODEL_WANDB_TAG = {
    "chemprop": "chemprop",
    "deltaprop": "deltaprop-btlh2",
    "xgboost": "xgboost",
}


@dataclass
class OrchestratorConfig:
    """Configuration for the GPU-packing orchestrator."""

    jobs_file: str
    """Path to a Python file exposing ``JOBS: list[str]`` of evaluate.cli command lines."""

    mem_threshold: float = 0.70
    """Target GPU memory utilization. New jobs launch only while usage stays under this fraction.
    Kept well below 1.0 because a job's footprint can swing ~30% up after it's packed (variable
    batch shapes, pairwise sampling); on a 24 GB card 0.70 leaves ~7 GB to absorb those spikes."""

    settle_seconds: float = 2 * 60
    """Minimum seconds between launches so a new job allocates GPU memory before we measure headroom.
    A 4090 ramps fast and has ample headroom, so we pack roughly every 3 min instead of every 10."""

    poll_interval: float = 30.0
    """Seconds between scheduler ticks (reap finished jobs, probe GPU, maybe launch)."""

    max_concurrent: int = 6
    """Hard cap on concurrently running subprocesses. GPU jobs are dataloader-bound, so several
    are packed per card to overlap one job's input-collation gaps with another's compute burst
    (raising both GPU and CPU utilisation). Per-job thread caps (see launch()) keep this from
    oversubscribing the cores; size it with num_workers so total dataloader workers stay near the
    core count, spread across the available GPUs (e.g. 6 jobs x 4 workers ~= 24 cores)."""

    max_cpu_concurrent: int = 2
    """Hard cap on concurrent CPU-only jobs (e.g. xgboost). They predict ~0 GPU so they would
    otherwise always 'fit' and crowd the box; each spawns its own Ray + thread pool."""

    mem_margin_mib: float = 500.0
    """Safety pad added to each job's predicted memory before deciding it fits."""

    mem_ema_alpha: float = 0.5
    """EMA weight for updating the per-row memory estimate from each observed launch."""

    max_retries: int = 2
    """How many times to re-queue a job after it fails (so up to ``max_retries + 1`` total
    launches). A failed job goes back to ``pending`` and is re-scheduled on a freshly chosen
    GPU once headroom allows -- this recovers transient OOMs caused by over-packing (the most
    common failure here), at the cost of repeating genuinely broken jobs a couple of times.
    Set to 0 to disable retries."""

    log_dir: str = "orchestrator_runs"
    """Parent directory for per-run log folders."""

    reconcile_wandb: bool = True
    """Wandb is the source of truth for what's done: before launching, fetch wandb runs and skip
    any job whose run is ``finished``. Only ``finished`` counts -- ``preempted``/``preempting`` are
    treated as not done and re-run. Disable to force a full re-run (nothing is skipped). If wandb is
    unreachable this warns and runs everything rather than guessing."""

    wandb_entity: str | None = None
    """Wandb entity to query for reconciliation. Defaults to the API's default entity."""

    random_seed: int | None = None
    """Seed for the randomized job selection. Set for reproducible launch orders."""

    gpus: tuple[int, ...] | None = None
    """Physical GPU indices (nvidia-smi order) to spread jobs across. None = every detected
    GPU. Each job is pinned to one GPU via CUDA_VISIBLE_DEVICES; jobs run one-per-GPU and are
    packed up to ``mem_threshold`` independently on each card."""

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
    model: str | None = None
    split: str = "butina"  # evaluate.cli default split is BUTINA
    use_feats: bool = False

    # Runtime state.
    status: str = "pending"  # pending | running | done | failed | skipped
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log_path: Path | None = None
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    used_before_launch_mib: float | None = None
    # Physical GPU index this job was pinned to (None for cpu-only jobs).
    gpu: int | None = None
    attempts: int = 0  # number of times this job has been launched (for retry tracking)

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


def extract_split(argv: list[str]) -> str:
    """Return the ``--train-cf.split-type`` value (lowercased), defaulting to butina."""
    for i, tok in enumerate(argv):
        if tok == "--train-cf.split-type" and i + 1 < len(argv):
            return argv[i + 1].lower()
        if tok.startswith("--train-cf.split-type="):
            return tok.split("=", 1)[1].lower()
    return "butina"


def extract_use_feats(argv: list[str]) -> bool:
    """Whether ``--train-cf.use-feats`` is set (tyro boolean flag)."""
    for tok in argv:
        if tok in ("--train-cf.use-feats", "--train-cf.use_feats"):
            return True
        if tok.startswith(("--train-cf.use-feats=", "--train-cf.use_feats=")):
            return tok.split("=", 1)[1].lower() in ("true", "1", "yes")
    return False


def extract_wandb_project(argv: list[str]) -> str | None:
    """Return the ``--wandb-cf.project-name`` value in argv, if present."""
    for i, tok in enumerate(argv):
        if tok == "--wandb-cf.project-name" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--wandb-cf.project-name="):
            return tok.split("=", 1)[1]
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
                model=model,
                split=extract_split(argv),
                use_feats=extract_use_feats(argv),
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


def gpu_mem_all() -> dict[int, tuple[float, float]]:
    """Return {gpu_index: (used_mib, total_mib)} for every GPU via nvidia-smi.

    nvidia-smi reports physical indices and ignores CUDA_VISIBLE_DEVICES, so these indices
    are stable regardless of how individual jobs are pinned.
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, tuple[float, float]] = {}
    for line in out.strip().splitlines():
        idx_str, used_str, total_str = line.split(",")
        result[int(idx_str)] = (float(used_str), float(total_str))
    return result


def fetch_finished_wandb(
    projects: list[str], entity: str | None
) -> list[frozenset[str]]:
    """Tag sets of every wandb run in ``state == 'finished'`` across the given projects.

    Only ``finished`` runs count as complete; ``preempted``/``preempting`` (every run here
    calls ``mark_preempting()``) are deliberately excluded so partial work gets re-run.
    """
    import wandb

    api = wandb.Api()
    ent = entity or api.default_entity
    finished: list[frozenset[str]] = []
    for project in projects:
        for run in api.runs(f"{ent}/{project}"):
            if run.state == "finished":
                finished.append(frozenset(run.tags))
    return finished


def job_finished_on_wandb(job: Job, finished_tag_sets: list[frozenset[str]]) -> bool:
    """True if some finished wandb run matches this job's (model, dataset, split, feats)."""
    model_tag = MODEL_WANDB_TAG.get(job.model or "")
    if model_tag is None or job.dataset is None:
        return False
    want = {model_tag, job.dataset.lower(), job.split.lower()}
    for tags in finished_tag_sets:
        # feats is a separate experiment: require an exact match on the with-feats tag.
        if want <= tags and (("with-feats" in tags) == job.use_feats):
            return True
    return False


def mark_finished_jobs(
    jobs: list[Job], finished_tag_sets: list[frozenset[str]]
) -> None:
    """Mark jobs whose wandb run is ``finished`` as ``skipped`` (wandb is the source of truth)."""
    for job in jobs:
        if job_finished_on_wandb(job, finished_tag_sets):
            job.status = "skipped"


class Orchestrator:
    def __init__(
        self,
        cfg: OrchestratorConfig,
        jobs: list[Job],
        run_dir: Path,
    ):
        self.cfg = cfg
        self.jobs = jobs
        self.run_dir = run_dir
        # Already-complete jobs were marked "skipped" upstream; only queue the rest.
        self.pending = [j for j in jobs if j.status == "pending"]  # already size-sorted
        self.running: list[Job] = []
        self.mem_per_row: float | None = None
        # Row count of the job that last calibrated mem_per_row. The per-row model has no
        # intercept, so we only ever (re)calibrate from the largest job seen -- a small one
        # would inflate per-row and make big jobs look unschedulable. 0 = uncalibrated.
        self.calib_rows: int = 0
        self.last_launch_ts: float = 0.0
        # GPUs to spread across. cfg.gpus restricts the set; otherwise every detected GPU.
        # Per-GPU memory is tracked independently so each card is packed up to mem_threshold
        # on its own. mem_per_row stays global -- it's a workload property, not a per-card one
        # (this assumes the GPUs are homogeneous; mixed-capacity cards are only approximated).
        detected = gpu_mem_all()
        if cfg.gpus is not None:
            missing = [g for g in cfg.gpus if g not in detected]
            if missing:
                raise SystemExit(f"Requested GPUs not found via nvidia-smi: {missing}")
            self.gpu_ids: list[int] = list(cfg.gpus)
        else:
            self.gpu_ids = sorted(detected)
        self.gpu_total: dict[int, float] = {g: detected[g][1] for g in self.gpu_ids}
        self.gpu_used: dict[int, float] = {g: detected[g][0] for g in self.gpu_ids}
        self.baseline_used: dict[int, float] = dict(self.gpu_used)
        self.peak_used: dict[int, float] = dict(self.gpu_used)
        # Short temp root for per-job Ray instances. Must be short: Ray builds AF_UNIX socket
        # paths under here and they cannot exceed 107 bytes (run_dir is far too deep to use).
        self.ray_tmp_root = Path(tempfile.mkdtemp(prefix="orchray_"))
        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)

    # --- memory model ---------------------------------------------------------

    def predict_mem(self, job: Job) -> float | None:
        """Predicted GPU memory (MiB) for a job, including the safety margin.

        CPU-only (and unknown/zero-row) jobs predict 0. A GPU job needs a per-row
        estimate; before any GPU job has settled we have no estimate, so we return None
        and the scheduler launches a single GPU job at a time to calibrate. (Returning the
        full budget here would make the first GPU job never fit alongside a non-zero
        baseline, so only the always-fits CPU jobs would ever launch.)
        """
        if job.cpu_only or job.rows <= 0:
            return 0.0
        if self.mem_per_row is None:
            return None
        return self.mem_per_row * job.rows + self.cfg.mem_margin_mib

    def update_mem_per_row(self, job: Job, used_now: float) -> bool:
        """Refine the per-row estimate from a settled launch's observed memory delta.

        Returns ``True`` when the sample has been handled and can be consumed -- either it
        calibrated the estimate, or it was deliberately skipped because a larger job already
        anchors it. Returns ``False`` when the job has not yet allocated GPU memory
        (``delta <= 0``, e.g. it is still in CPU-side preprocessing); the caller should keep
        the sample and retry on a later tick rather than burning the one measurement.

        Only (re)calibrate from the largest job seen so far. The per-row model has no
        intercept, so a small job (fixed CUDA/model overhead spread over few rows) yields a
        hugely inflated per-row that would make big jobs look unschedulable. Anchoring on the
        largest keeps the estimate representative for heavy jobs and merely conservative
        (padded by ``mem_margin_mib``) for the light ones.
        """
        if job.cpu_only or job.rows <= 0 or job.used_before_launch_mib is None:
            return True
        if self.mem_per_row is not None and job.rows < self.calib_rows:
            return True  # a larger job already anchors the estimate; skip, don't spin
        delta = used_now - job.used_before_launch_mib
        if delta <= 0:
            return False  # GPU not allocated yet (still preprocessing); retry next tick
        observed = delta / job.rows
        if self.mem_per_row is None:
            self.mem_per_row = observed
        else:
            a = self.cfg.mem_ema_alpha
            self.mem_per_row = a * observed + (1 - a) * self.mem_per_row
        self.calib_rows = job.rows
        return True

    # --- subprocess lifecycle -------------------------------------------------

    def launch(self, job: Job, gpu: int | None, used_now: float) -> None:
        job.attempts += 1
        # Suffix retry logs so a failed attempt's output isn't overwritten by the next try.
        suffix = "" if job.attempts == 1 else f".try{job.attempts}"
        job.log_path = self.run_dir / f"{job.job_id}{suffix}.log"
        # Per-job Ray temp dir under the short root (absolute + short to satisfy Ray's
        # AF_UNIX 107-byte socket-path limit). Keyed by idx to keep it tiny.
        job_tmp = self.ray_tmp_root / f"j{job.idx}"
        job_tmp.mkdir(exist_ok=True)
        env = os.environ.copy()
        # Isolate each run's Ray instance to avoid temp/port collisions (cli.py ray.init).
        env["TMPDIR"] = str(job_tmp)
        env["RAY_TMPDIR"] = str(job_tmp)
        # Pin the job to its assigned GPU. With only that card visible, the trainers'
        # accelerator="auto", devices=1 lands on it (no evaluate.cli changes needed).
        # PCI_BUS_ID makes CUDA's enumeration match the nvidia-smi index we assigned.
        # cpu-only jobs (xgboost) get no GPU.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "" if job.cpu_only else str(gpu)
        # Bound BLAS/OpenMP threads so concurrent jobs don't oversubscribe the cores.
        # GPU jobs (chemprop/deltaprop) are dataloader-bound -- their parallelism comes
        # from DataLoader workers, not intra-op math -- so they get a small cap. cpu-only
        # jobs (xgboost) do their real work in-process and get a fair share of cores.
        if job.cpu_only:
            threads = max(1, (os.cpu_count() or 1) // self.cfg.max_cpu_concurrent)
        else:
            threads = 2
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = str(threads)
        # Idle OpenMP threads must sleep, not busy-spin -- spin-waiting under
        # oversubscription is the main driver of the ~500k ctx-switches/sec observed
        # when packing many jobs onto the box.
        env["OMP_WAIT_POLICY"] = "PASSIVE"
        env["KMP_BLOCKTIME"] = "0"
        log_file = open(job.log_path, "w")
        job.proc = subprocess.Popen(
            job.argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job.gpu = gpu
        job.status = "running"
        job.started_at = time.time()
        job.used_before_launch_mib = used_now
        self.running.append(job)
        self.pending.remove(job)
        self.last_launch_ts = time.time()
        predicted = self.predict_mem(job)
        pred_str = f"{predicted:.0f} MiB" if predicted is not None else "uncalibrated"
        gpu_str = "cpu" if job.cpu_only else f"gpu{gpu}"
        attempt_str = "" if job.attempts == 1 else f", attempt {job.attempts}"
        print(
            f"[launch] {job.job_id} ({gpu_str}, rows={job.rows}, "
            f"predicted={pred_str}{attempt_str}) -> {job.log_path}"
        )

    def reap(self) -> None:
        """Collect finished subprocesses and classify their exit.

        Failed jobs are re-queued (back to ``pending``) while they have retries left, so a
        transient over-pack OOM can succeed on a later, emptier launch. Once retries are
        exhausted the job is marked ``failed`` for good.
        """
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
                print(f"[reap]   {job.job_id} done")
                continue
            reason = "OOM" if self._log_has_oom(job) else f"rc={rc}"
            if job.attempts <= self.cfg.max_retries:
                # Re-queue: reset runtime state and let the scheduler place it afresh.
                job.status = "pending"
                job.proc = None
                job.gpu = None
                job.started_at = None
                job.finished_at = None
                job.returncode = None
                job.used_before_launch_mib = None
                self.pending.append(job)
                retries_left = self.cfg.max_retries - job.attempts + 1
                print(
                    f"[reap]   {job.job_id} FAILED ({reason}) -> retrying "
                    f"({retries_left} left)"
                )
            else:
                job.status = "failed"
                print(
                    f"[reap]   {job.job_id} FAILED ({reason}) "
                    f"after {job.attempts} attempt(s)"
                )
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

    def pick_job_and_gpu(
        self, used_by_gpu: dict[int, float], free_only: bool = False
    ) -> tuple[Job, int | None] | None:
        """Pick a ``(job, gpu)`` pair to launch, chosen at random among those that fit.

        A GPU job is eligible iff *some* GPU has room for its predicted footprint under that
        card's ``mem_threshold`` cap; it is pinned to one of the GPUs with room. Among the
        eligible ``(job, gpu)`` pairs we choose *randomly* rather than strictly largest-first,
        so one heavy dataset (e.g. GSK_HEPG2, which contributes many same-size jobs) doesn't
        monopolise every launch slot. CPU-only jobs (gpu=None) respect the CPU-only cap.

        Two exceptions, both about calibration: the intercept-free per-row model must be
        anchored on a big job (see ``update_mem_per_row``), so while ``mem_per_row`` is still
        unknown we (a) run only one uncalibrated GPU job *per card* (placing each on a GPU with
        no running GPU job), and (b) deterministically pick the *largest* such job.

        When ``free_only`` is set the caller is launching onto an idle card without waiting for
        the global settle timer, so every returned pair is pinned to a GPU with no running job
        (at most one new job per card per tick) and CPU-only jobs are not offered -- they stay
        on the settle-paced path.
        """
        running_cpu = sum(1 for j in self.running if j.cpu_only)
        # GPUs that currently have a running GPU job pinned to them.
        busy_gpus = {
            j.gpu for j in self.running if not j.cpu_only and j.gpu is not None
        }
        free_gpus = [g for g in self.gpu_ids if g not in busy_gpus]

        # CPU-only candidates (no GPU assignment). Skipped under free_only: they don't occupy a
        # card, so they belong on the settle-paced path, not the idle-card fast path.
        cpu_candidates: list[Job] = []
        if not free_only and running_cpu < self.cfg.max_cpu_concurrent:
            cpu_candidates = [j for j in self.pending if j.cpu_only]

        # GPU (job, gpu) candidate pairs.
        gpu_pairs: list[tuple[Job, int]] = []
        for job in self.pending:
            if job.cpu_only:
                continue
            predicted = self.predict_mem(job)
            if predicted is None:
                # Uncalibrated GPU job: one per free GPU (one at a time per card).
                for g in free_gpus:
                    gpu_pairs.append((job, g))
                continue
            cand_gpus = free_gpus if free_only else self.gpu_ids
            for g in cand_gpus:
                cap = self.gpu_total[g] * self.cfg.mem_threshold
                if used_by_gpu[g] + predicted <= cap:
                    gpu_pairs.append((job, g))

        # Anchor the first calibration on the largest eligible GPU job (any free GPU).
        if self.mem_per_row is None and gpu_pairs:
            job, _ = max(gpu_pairs, key=lambda pair: pair[0].rows)
            if free_gpus:
                return job, random.choice(free_gpus)

        candidates: list[tuple[Job, int | None]] = [(j, None) for j in cpu_candidates]
        candidates.extend(gpu_pairs)
        if not candidates:
            return None
        return random.choice(candidates)

    def run(self) -> None:
        gpu_list = ", ".join(
            f"gpu{g} {self.gpu_used[g]:.0f}/{self.gpu_total[g]:.0f} MiB"
            for g in self.gpu_ids
        )
        print(
            f"GPUs at start: {gpu_list}; per-GPU threshold {self.cfg.mem_threshold:.0%}"
        )

        try:
            while self.pending or self.running:
                self.reap()
                detected = gpu_mem_all()
                for g in self.gpu_ids:
                    self.gpu_used[g] = detected[g][0]
                    self.gpu_total[g] = detected[g][1]
                    self.peak_used[g] = max(self.peak_used[g], self.gpu_used[g])

                # Refine the per-row estimate from jobs that have had time to settle.
                # Each job's footprint is read from the card it was pinned to.
                for job in self.running:
                    if (
                        job.started_at is not None
                        and time.time() - job.started_at >= self.cfg.settle_seconds
                        and job.used_before_launch_mib is not None
                        and job.gpu is not None
                    ):
                        if self.update_mem_per_row(job, self.gpu_used[job.gpu]):
                            job.used_before_launch_mib = (
                                None  # consume once measured/skipped
                            )
                        else:
                            print(
                                f"[calib]  {job.job_id} no GPU allocation yet "
                                f"(g{job.gpu}={self.gpu_used[job.gpu]:.0f} MiB); "
                                "deferring calibration"
                            )

                # The settle timer paces *packing* (adding a job to a card that already has one,
                # so its footprint is measured before we add more). Launching onto an idle card
                # has no headroom to measure, so we allow it immediately and only restrict the
                # pick to free cards (free_only) when the timer hasn't elapsed.
                busy_gpus = {
                    j.gpu for j in self.running if not j.cpu_only and j.gpu is not None
                }
                has_free_gpu = any(g not in busy_gpus for g in self.gpu_ids)
                settled = time.time() - self.last_launch_ts >= self.cfg.settle_seconds
                if (
                    self.pending
                    and (settled or has_free_gpu)
                    and len(self.running) < self.cfg.max_concurrent
                ):
                    pick = self.pick_job_and_gpu(self.gpu_used, free_only=not settled)
                    if pick is not None:
                        job, gpu = pick
                        used_now = self.gpu_used[gpu] if gpu is not None else 0.0
                        self.launch(job, gpu, used_now)
                        continue  # re-tick immediately to reflect the new launch

                self._print_status()
                if not self.pending and not self.running:
                    break
                time.sleep(self.cfg.poll_interval)
        except KeyboardInterrupt:
            print("\n[interrupt] terminating running jobs...")
            self._terminate_all()
        finally:
            self._write_summary()
            shutil.rmtree(self.ray_tmp_root, ignore_errors=True)

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

    def _print_status(self) -> None:
        counts = {s: 0 for s in ("pending", "running", "done", "failed", "skipped")}
        for job in self.jobs:
            counts[job.status] += 1
        mpr = f"{self.mem_per_row:.3f}" if self.mem_per_row is not None else "n/a"
        gpu_str = " ".join(
            f"g{g}={self.gpu_used[g]:.0f}/{self.gpu_total[g]:.0f}" for g in self.gpu_ids
        )
        print(
            f"[status] {gpu_str} MiB | "
            f"mem/row={mpr} | running={counts['running']} pending={counts['pending']} "
            f"done={counts['done']} failed={counts['failed']} skipped={counts['skipped']}"
        )

    def _write_summary(self) -> None:
        summary = {
            "run_dir": str(self.run_dir),
            "gpus": {
                str(g): {
                    "total_mib": self.gpu_total[g],
                    "baseline_used_mib": self.baseline_used[g],
                    "peak_used_mib": self.peak_used[g],
                }
                for g in self.gpu_ids
            },
            "mem_per_row_final": self.mem_per_row,
            "jobs": [
                {
                    "job_id": j.job_id,
                    "command": j.command,
                    "dataset": j.dataset,
                    "rows": j.rows,
                    "cpu_only": j.cpu_only,
                    "gpu": j.gpu,
                    "attempts": j.attempts,
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
            gpu_str = "" if j.gpu is None else f" gpu={j.gpu}"
            print(f"  {j.job_id:<28} {j.status:<8} rc={j.returncode}{gpu_str}")


def print_dry_run(jobs: list[Job]) -> None:
    print("Dry run — launch order (largest dataset first):\n")
    pos = 0
    for job in jobs:
        ds = job.dataset or "unknown"
        if job.status == "skipped":
            print(
                f"   -- {job.job_id:<28} rows={job.rows:<6} {ds} [skip: already done]"
            )
            continue
        pos += 1
        tag = " [cpu-only]" if job.cpu_only else ""
        feat = " [feats]" if job.use_feats else ""
        print(f"  {pos:>2}. {job.job_id:<28} rows={job.rows:<6} {ds}{tag}{feat}")
        print(f"      {job.command}")
    print("\nNo processes were launched (dry run).")


def reconcile_with_wandb(jobs: list[Job], cfg: OrchestratorConfig) -> None:
    """Mark jobs already ``finished`` on wandb as skipped. Wandb is the source of truth."""
    finished_tag_sets: list[frozenset[str]] = []
    if cfg.reconcile_wandb:
        projects = sorted({p for j in jobs if (p := extract_wandb_project(j.argv))})
        if projects:
            try:
                finished_tag_sets = fetch_finished_wandb(projects, cfg.wandb_entity)
                print(
                    f"Reconciled {len(finished_tag_sets)} finished wandb run(s) "
                    f"across {len(projects)} project(s): {', '.join(projects)}"
                )
            except Exception as exc:  # noqa: BLE001 - network/auth/import all degrade alike
                print(
                    f"[warn] wandb reconciliation failed ({exc}); "
                    "running all jobs (nothing skipped)."
                )

    mark_finished_jobs(jobs, finished_tag_sets)

    done = [j for j in jobs if j.status == "skipped"]
    runnable = sum(1 for j in jobs if j.status == "pending")
    if done:
        print(f"\nAlready completed on wandb ({len(done)}) — skipping:")
        for job in sorted(done, key=lambda j: j.idx):
            ds = job.dataset or "unknown"
            feat = "feats" if job.use_feats else "no-feats"
            print(
                f"  ✓ {job.job_id:<28} "
                f"{job.model or '?':<10} {ds:<18} {job.split:<8} {feat}"
            )
    print(f"\n{len(done)} already finished on wandb (skipped); {runnable} to run.\n")


def main(cfg: OrchestratorConfig) -> None:
    commands = load_jobs_file(cfg.jobs_file)
    jobs = build_jobs(commands)

    reconcile_with_wandb(jobs, cfg)

    if cfg.dry_run:
        print_dry_run(jobs)
        return

    # Fail fast if we can't read the GPUs.
    try:
        gpu_mem_all()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not query GPUs via nvidia-smi: {exc}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(cfg.log_dir) / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}\n")

    orch = Orchestrator(cfg, jobs, run_dir)
    orch.run()


if __name__ == "__main__":
    # Ensure Ctrl-C raises KeyboardInterrupt in the main loop.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main(tyro.cli(OrchestratorConfig))
