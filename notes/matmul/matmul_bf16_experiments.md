# Matmul BF16 Ablation Experiments (`tileKernel.cu`)

- Target kernel: `[src/matmul_bf16/tileKernel.cu](../../src/matmul_bf16/tileKernel.cu)`
- Rule: `O` means the stage is included in that experiment. Blank means ablated (skipped).
- Debug skip flags used for ablations:
  - `-DDEBUG_SKIP_G2S_LOAD`
  - `-DDEBUG_SKIP_LDMATRIX`
  - `-DDEBUG_SKIP_MMA`
  - `-DDEBUG_SKIP_R2G_STORE`
- Raw log: `logs/matmul_bf16_16_ablations_20260309_rerun.log`
- cuBLAS-only log: `logs/matmul_bf16_cublas_only_20260309_rerun.log`

## Benchmark Configuration

- `m=8192`
- `warmup=5`, `iters=10`, `kernels=1`
- Git commit (`HEAD`): `978266a0d70b5e4d73451c70ba84acdc025405e0` (`978266a`)
- Tile config: `BM=128`, `BN=128`, `BK=32`, `WM=64`, `WN=64`
- Command template:
  - `./scripts/matmul_bf16/run.sh --nvcc-flag <flags...> -- 8192 --run tile:bm=128,bn=128,bk=32,wm=64,wn=64`

| Experiment | Mask | Global -> Shared Load | Shared -> Register Load | MMA (Tensor Core Op) | Register -> Global Store | Latency (us) |
| --- | --- | --- | --- | --- | --- | --- |
| cuBLAS | - |  |  |  |  | 3923 |
| Ablation 00 | 0 | O | O | O | O | 9476 |
| Ablation 01 | 1 |  | O | O | O | 7374 |
| Ablation 02 | 2 | O |  | O | O | 7399 |
| Ablation 03 | 3 |  |  | O | O | 3938 |
| Ablation 04 | 4 | O | O |  | O | 8968 |
| Ablation 05 | 5 |  | O |  | O | 7864 |
| Ablation 06 | 6 | O |  |  | O | 6681 |
| Ablation 07 | 7 |  |  |  | O | 3091 |
| Ablation 08 | 8 | O | O | O |  | 9497 |
| Ablation 09 | 9 |  | O | O |  | 7179 |
| Ablation 10 | 10 | O |  | O |  | 7229 |
| Ablation 11 | 11 |  |  | O |  | 3857 |
| Ablation 12 | 12 | O | O |  |  | 8967 |
| Ablation 13 | 13 |  | O |  |  | 7164 |
| Ablation 14 | 14 | O |  |  |  | 6304 |
| Ablation 15 | 15 |  |  |  |  | 2986 |
