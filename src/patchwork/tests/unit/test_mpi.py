"""Patches on their own communicators, under ``mpirun``."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOB = Path(__file__).with_name("_mpi_placement_job.py")
MPIRUN = shutil.which("mpirun")

pytestmark = pytest.mark.skipif(
    MPIRUN is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpirun and mpi4py",
)


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_patches_on_some_ranks_give_what_one_rank_gives(ranks: int) -> None:
    """Placement, gathers, both solvers and the plot (see the job's docstring)."""
    env = os.environ.copy()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run(
        [MPIRUN, "-n", str(ranks), "--oversubscribe", sys.executable, str(JOB)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )

    assert "MPI_JOB_OK" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_halo_exchange_and_restriction_under_mpi(ranks: int) -> None:
    """``test_halos.py`` on every rank, with patches on one, some and all ranks."""
    env = os.environ.copy()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    tests = Path(__file__).with_name("test_halos.py")
    result = subprocess.run(
        [
            MPIRUN,
            "-n",
            str(ranks),
            "--oversubscribe",
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(tests),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
