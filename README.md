# QR2 Shape-Specialized Householder Megakernels

A shape-specialized batched FP32 QR implementation for NVIDIA Blackwell GPUs.
The small n32 and n176 routes use full-matrix megakernels; larger Householder
routes use fused panel megakernels inside blocked or recursive factorizations.
The implementation returns the LAPACK/PyTorch compact Householder
representation `(H, tau)` used by `torch.geqrf`.

## Provenance

The wide-panel Householder leaf kernels use and adapt the CUDA design from
**gau.nernst's QR2 submission** on the
[GPU MODE leaderboard](https://www.gpumode.com/leaderboard/774?tab=rankings).
That provenance is recorded once here and at the top of `submission.py`;
internal symbols use project-local names.

This repository adds and evaluates:

- full-matrix n32 and n176 Householder megakernels;
- recursive Householder panel decomposition for larger shapes;
- terminal-tail specializations, including the n512 active `96 + 96` tail;
- compact-WY construction and Tensor-Core near/far updates;
- per-matrix handling of heterogeneous n512 batches;
- graph replay to amortize a dependent multi-kernel launch schedule;
- a guarded Gram/CholeskyQR-style path for the benchmark's n4096 profile;
- FP32 compact-factor reconstruction and the QR2 conditioning checks.

This is a derivative implementation. It does not claim the original
wide-panel kernel design as independent work.

## What “megakernel” means here

“Megakernel” describes the CUDA kernel that fuses a complete Householder panel:
it loads the panel and performs the sequential reflector chain, intra-panel
updates, `tau` generation, and factor writeback in one launch. For n32 and
n176, that panel is the complete square matrix, so the QR factorization itself
is a full-matrix megakernel. The n176 launch uses a two-CTA cluster. The normal
n32 execution wrapper replays a two-node input-copy-plus-kernel CUDA Graph,
but the numerical factorization is still one kernel.

For n352 and larger Householder routes, the fused panel is only a leaf of the
complete factorization. A recursive node is broader than one leaf: it is
composed from leaf megakernels, small `T` builders, and Tensor-Core
`bmm`/`baddbmm` operations. The complete recursive node and factorization are
therefore not single monolithic CUDA kernels.

The execution boundaries are shape-specific:

| Route | Factorization boundary | CUDA Graph replay | PDL |
|---|---|---:|---:|
| n32 | full-matrix megakernel | input copy + QR kernel | QR kernel |
| n176 | full-matrix two-CTA-cluster megakernel | no | no |
| n352, n512, n1024, n2048 | panel megakernels in a multi-kernel DAG | yes | not generally |
| guarded n4096 | Gram/CholeskyQR-style multi-kernel DAG | yes | no |

This implementation is also **not persistent**: its CTAs factor fixed panels
and exit; they do not remain resident and pull work from a long-lived queue.

CUDA Graph replay reduces host/driver launch gaps. It does not fuse kernels,
remove mathematical dependencies, or by itself prove execution overlap. The
submission creates no auxiliary execution streams.

PDL is not a blanket property of the larger recursive routes. It is enabled
for the n32 QR kernel and for a specialized n1024 near-rank-deficient tail-pack
launch; the main n1024 panel/update DAG, and the n2048 and guarded n4096 routes,
should be described as CUDA Graph schedules rather than uniformly as
“CUDA Graph + PDL.”

The concise description is:

> Shape-specialized Householder QR with full-matrix n32/n176 megakernels and
> larger blocked or recursive factorizations built from fused panel
> megakernels, compact-WY composition, Tensor-Core trailing updates, and CUDA
> Graph replay, plus a guarded CholeskyQR-style n4096 specialization.

See [ALGORITHM.md](ALGORITHM.md) for the mathematical and implementation
boundaries.

## Verified result

On one NVIDIA GB200 with CUDA 13 and PyTorch 2.12, two complete runs of the
current 12-case `leaderboard-current` suite produced **0.948304 ms** and
**0.9503 ms** geometric means. All 12 cases passed the FP32 compact-factor
residual and orthogonality gates against the original FP32 inputs. The table
below records the 0.948304 ms run.

| Case | Mean (ms) |
|---|---:|
| `b20 n32 dense` | 0.0143 |
| `b40 n176 dense` | 0.0755 |
| `b40 n352 dense` | 0.2446 |
| `b640 n512 dense` | 2.6213 |
| `b60 n1024 dense` | 1.7618 |
| `b8 n2048 dense` | 3.0648 |
| `b2 n4096 dense` | 5.5608 |
| `b640 n512 mixed` | 2.8325 |
| `b60 n1024 mixed` | 1.7601 |
| `b640 n512 rank-deficient` | 2.2982 |
| `b640 n512 clustered` | 1.5158 |
| `b60 n1024 near-rank-deficient` | 1.4675 |

The roughly 0.2% run-to-run movement is measurement variability, with the
largest change in the very short n176 case. The score is an
environment-specific measurement, not a portable latency guarantee or a
guarantee that every run is below 0.950 ms.

## Numerical contract

For CUDA FP32 input `A` with shape `(batch, n, n)`, `custom_kernel` returns:

- `H`: FP32, with `R` in the upper triangle and reflector tails below it;
- `tau`: FP32 Householder coefficients.

The checker reconstructs `Q = householder_product(H, tau)` and validates the
factor residual and orthogonality in FP64. Internal FP16 reflector copies are
an update optimization only; the returned representation remains FP32.

The optimized code is intentionally specialized for the seven public batch
and matrix shapes in the QR2 suite. Unsupported shapes use `torch.geqrf` as a
correctness fallback.

## Repository layout

- `submission.py`: standalone submission with embedded CUDA source;
- `ALGORITHM.md`: terminology, derivation, and implementation mapping;
- `benchmark.py`: self-contained 12-case correctness/timing harness;
- `reference.py`: input profiles and FP64 residual checker.

There are no generated candidates or historical tuning experiments here.

## Run

Use a CUDA 13 development image, then install the dependencies from
`pyproject.toml`.

```bash
python benchmark.py --mode test
python benchmark.py --mode benchmark --max-repeats 10
python benchmark.py --mode leaderboard --json-out .build/leaderboard.json
```

`leaderboard` enables correctness rechecks and allows up to 1000 repeats. The
shorter `benchmark` mode is intended for iteration, not final reporting.
