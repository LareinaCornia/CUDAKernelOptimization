# CUDAKernelOptimization

CUDA implementations of SAXPY, matrix multiplication (SGEMM), and depthwise 1D
convolution, with correctness and performance suites.

## Layout

| Path | Purpose |
| --- | --- |
| `kernels/` | CUDA implementations and drivers grouped by operation |
| `include/` | Shared CUDA headers |
| `tools/` | Device configuration utility |
| `src/cuda_explore/` | Python test runner, benchmark calculations, and Slurm submission |
| `test/run_tests.py`, `scripts/runner.py` | Script entry points for the Python package |
| `scripts/interact.sh` | Interactive Hydra allocation |
| `docs/` | Project notes |

## Build and run

Use a Linux host with an NVIDIA GPU, the CUDA toolkit (including cuBLAS), a
compatible host C++ compiler, GNU Make, and uv. Run from this checkout with the
default editable uv environment. For commands issued elsewhere, use
`make -C /path/to/CUDAKernelOptimization …` or
`uv run --project /path/to/CUDAKernelOptimization …`.

```sh
make -j
make check saxpy_correct conv1d_correct
uv run --locked cuda-explore --help
uv run --locked cuda-explore --no-color sgemm_all
./build/query_config
```

`make check` runs all suites by default, and `make sgemm` builds just SGEMM.
SAXPY and SGEMM performance suites require Nsight Systems (`nsys`) or
`nvprof`. Conv1D performance uses CUDA-enabled PyTorch. To configure a
different GPU architecture or host compiler, clean before rebuilding, for
example `make clean && make CUDA_ARCHS=89 HOST_CXX=g++-13`.

On Hydra, submit a batch job with:

```sh
uv run --locked cuda-explore-slurm saxpy_correct conv1d_correct
```

The job runs from the checkout; its filesystem must be visible to the compute
node. The script and its job-ID-specific output are saved under `logs/`.

Compiled executables live under `build/`. Performance summaries go to stdout,
and each profiler invocation uses a temporary directory inside `build/` that
is removed when parsing finishes. The Conv1D PyTorch extension source is
separate from the standalone binaries. An extension builder needs to add
`include/` to its header search path.
