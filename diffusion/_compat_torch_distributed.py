"""
torch.distributed compatibility shim for Windows ROCm PyTorch builds.

PyTorch on Windows + ROCm (e.g. torch==2.9.1+rocm7.2.1) is compiled with
USE_DISTRIBUTED=0, so torch.distributed.is_available() returns False and most
symbols are missing (is_initialized, get_world_size, group, ReduceOp, ...).

Many libraries (vector_quantize_pytorch >=1.22, parts of diffusers/accelerate)
import these unconditionally at module top-level and crash. This shim adds
safe single-process defaults so those imports succeed without entering any
real distributed code path.

USAGE: import this module BEFORE importing any of those libraries.
       Typically: add `from diffusion import _compat_torch_distributed  # noqa: F401`
       at the very top of main_test.py / main_train.py, before the diffusers/peft imports.

WARNING: do NOT install this as sitecustomize.py — importing torch in sitecustomize
makes every Python subprocess (including pip / accelerate helpers) load torch at
startup, which on Windows ROCm spawns helper processes that then re-trigger
sitecustomize.py recursively. This caused a fork-bomb of 100+ python.exe instances
during development. Always import explicitly from application entry points instead.
"""

import torch.distributed as _dist

if not _dist.is_available():
    if not hasattr(_dist, 'is_initialized'):
        _dist.is_initialized = lambda: False
    if not hasattr(_dist, 'get_world_size'):
        _dist.get_world_size = lambda group=None: 1
    if not hasattr(_dist, 'get_rank'):
        _dist.get_rank = lambda group=None: 0
    if not hasattr(_dist, 'is_nccl_available'):
        _dist.is_nccl_available = lambda: False
    if not hasattr(_dist, 'is_mpi_available'):
        _dist.is_mpi_available = lambda: False
    if not hasattr(_dist, 'is_gloo_available'):
        _dist.is_gloo_available = lambda: False

    if not hasattr(_dist, 'group'):
        class _DummyGroup:
            WORLD = None
            NON_GROUP_MEMBER = -100
        _dist.group = _DummyGroup

    if not hasattr(_dist, 'ReduceOp'):
        class _ReduceOp:
            SUM = 0
            PRODUCT = 1
            MIN = 2
            MAX = 3
            BAND = 4
            BOR = 5
            BXOR = 6
            PREMUL_SUM = 7
            AVG = 8
        _dist.ReduceOp = _ReduceOp

    if not hasattr(_dist, 'GroupMember'):
        class _GroupMember:
            WORLD = None
            NON_GROUP_MEMBER = -100
        _dist.GroupMember = _GroupMember

    def _unavailable(name):
        def _raise(*a, **k):
            raise RuntimeError(
                f"torch.distributed.{name} called but torch.distributed is not "
                f"available on this PyTorch build (Windows ROCm USE_DISTRIBUTED=0)."
            )
        return _raise

    for _op in [
        'init_process_group', 'destroy_process_group', 'new_group',
        'barrier', 'broadcast', 'all_reduce', 'all_gather', 'all_gather_object',
        'gather', 'scatter', 'reduce', 'reduce_scatter',
        'send', 'recv', 'isend', 'irecv', 'monitored_barrier',
    ]:
        if not hasattr(_dist, _op):
            setattr(_dist, _op, _unavailable(_op))
