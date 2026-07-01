# QR2 Recursive Householder Mega-Panels

A high-performance batched FP32 QR implementation for NVIDIA GPUs. It returns
the compact `(H, tau)` representation used by `torch.geqrf`.

## Attribution

This implementation uses and adapts the wide-panel Householder CUDA kernels
from **gau.nernst's QR2 submission** on the
[GPU MODE leaderboard](https://www.gpumode.com/leaderboard/774?tab=rankings).
Those kernels are the foundation of the leaf panel factorization. We preserve
that provenance explicitly instead of encoding the original author's name in
every internal symbol.

The main extensions developed in this repository are:

- recursive Householder composition above the wide-panel leaf kernels;
- direct construction of parent compact-WY factors for large panels;
- graph-orchestrated execution of the recursive factorization;
- per-matrix routing for heterogeneous, ill-conditioned batches;
- a guarded communication-avoiding QR path for the `4096 x 4096` case;
- FP32 compact-factor output and robustness against the QR2 numerical gates.

This is therefore a derivative and extended implementation, not a claim that
the wide-panel leaf kernels were developed independently from scratch.

## What is the algorithm?

The implementation is best described as:

> Graph-orchestrated recursive Householder QR with fused mega-panel leaf
> kernels, plus a guarded CAQR path for the largest benchmark shape.

It is not a persistent kernel, and the complete QR factorization is not fused
into one monolithic launch. A leaf kernel is a mega-kernel because one launch
generates and applies many Householder reflectors for a wide panel. Recursive
nodes and trailing updates are composed from several kernels and matrix
multiplications.

See [ALGORITHM.md](ALGORITHM.md) for the compact-WY baseline, recursive
construction, GPU mapping, CAQR path, and the distinction from a persistent
kernel.

## Results

On one NVIDIA GB200 with CUDA 13 and PyTorch 2.12, the 12-case QR2 leaderboard
suite reaches a **0.9913 ms geometric mean**. All cases pass the FP32
compact-QR factor-residual and orthogonality gates measured against the
original FP32 input.

The optimized benchmark shapes are:

- `(20, 32, 32)`
- `(40, 176, 176)`
- `(40, 352, 352)`
- `(640, 512, 512)`
- `(60, 1024, 1024)`
- `(8, 2048, 2048)`
- `(2, 4096, 4096)`

Other square batched FP32 shapes use `torch.geqrf` as a correctness fallback.

## Repository layout

- `submission.py`: the independent submission file with embedded CUDA source;
- `ALGORITHM.md`: algorithm derivation, diagrams, and implementation mapping;
- `benchmark.py`: self-contained 12-case correctness and timing harness;
- `reference.py`: input profiles and FP64 residual checker.

There are no generated candidates, tuning scripts, or historical experiments
in this repository.

## Run

Use a CUDA development image with CUDA 13, then install the dependencies in
`pyproject.toml`.

```bash
python benchmark.py --mode test
python benchmark.py --mode benchmark --max-repeats 10
python benchmark.py --mode leaderboard --json-out .build/leaderboard.json
```

`leaderboard` uses up to 1000 repeats and enables correctness rechecks during
timing. The shorter `benchmark` command is intended for local iteration.
