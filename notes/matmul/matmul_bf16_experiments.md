# Matmul BF16 Experiments (`tileKernel.cu`)

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


| Experiment  | Mask | Global -> Shared Load | Shared -> Register Load | MMA (Tensor Core Op) | Register -> Global Store | Latency (us) |
| ----------- | ---- | --------------------- | ----------------------- | -------------------- | ------------------------ | ------------ |
| cuBLAS      | -    |                       |                         |                      |                          | 3923         |
| Ablation 00 | 0    | O                     | O                       | O                    | O                        | 9476         |
| Ablation 01 | 1    |                       | O                       | O                    | O                        | 7374         |
| Ablation 02 | 2    | O                     |                         | O                    | O                        | 7399         |
| Ablation 03 | 3    |                       |                         | O                    | O                        | 3938         |
| Ablation 04 | 4    | O                     | O                       |                      | O                        | 8968         |
| Ablation 05 | 5    |                       | O                       |                      | O                        | 7864         |
| Ablation 06 | 6    | O                     |                         |                      | O                        | 6681         |
| Ablation 07 | 7    |                       |                         |                      | O                        | 3091         |
| Ablation 08 | 8    | O                     | O                       | O                    |                          | 9497         |
| Ablation 09 | 9    |                       | O                       | O                    |                          | 7179         |
| Ablation 10 | 10   | O                     |                         | O                    |                          | 7229         |
| Ablation 11 | 11   |                       |                         | O                    |                          | 3857         |
| Ablation 12 | 12   | O                     | O                       |                      |                          | 8967         |
| Ablation 13 | 13   |                       | O                       |                      |                          | 7164         |
| Ablation 14 | 14   | O                     |                         |                      |                          | 6304         |
| Ablation 15 | 15   |                       |                         |                      |                          | 2986         |


- Observation
  - `cuBLAS`: `3923 us`
  - `only MMA` (Ablation 03): `3938 us` (almost same as cuBLAS)
  - `only ldmatrix` (Ablation 13): `7164 us` (much slower)
  - `ldmatrix + MMA` (Ablation 09): `7179 us` (almost same as only ldmatrix)
- Ideal Check
  - Estimated total `ldmatrix` traffic for this config: `~128 GiB`
  - A100 shared-memory peak model: `128 B/cycle/SM * 108 * 1.41 GHz ~= 19.49 TB/s`
  - Ideal lower bound: `128 GiB / 19.49 TB/s ~= 7.05 ms`
  - Measured `~7.16-7.18 ms` is close to this bound
- Conclusion
  - In the `ldmatrix + MMA` slice, `ldmatrix` is the dominant limiter.
  - This is not a low-utilization issue; the main issue is excessive `shared -> register` (`ldmatrix`) traffic.
  - Optimization priority: reduce `ldmatrix` traffic (more reuse, fewer reloads).

---

## Benchmark Configuration (Rerun: 2026-03-10)

- `m=8192`
- `warmup=5`, `iters=10`, `kernels=1`
- Git commit (`HEAD`): `a3ff045a515f81b7106bd34dac56046389a2eb41` (`a3ff045`)
- Tile config: `BM=128`, `BN=128`, `BK=32`, `WM=64`, `WN=64`, `RM=32`, `RN=32`
- Raw log: `logs/matmul_bf16_16_ablations_20260310_a3ff045.log`
- cuBLAS-only log: `logs/matmul_bf16_cublas_only_20260310_a3ff045.log`
- Command template:
  - `./scripts/matmul_bf16/run.sh --nvcc-flag <flags...> -- 8192 --run tile:bm=128,bn=128,bk=32,wm=64,wn=64,rm=32,rn=32`

| Experiment  | Mask | Global -> Shared Load | Shared -> Register Load | MMA (Tensor Core Op) | Register -> Global Store | Latency (us) |
| ----------- | ---- | --------------------- | ----------------------- | -------------------- | ------------------------ | ------------ |
| cuBLAS      | -    |                       |                         |                      |                          | 4022         |
| Ablation 00 | 0    | O                     | O                       | O                    | O                        | 7688         |
| Ablation 01 | 1    |                       | O                       | O                    | O                        | 4010         |
| Ablation 02 | 2    | O                     |                         | O                    | O                        | 8340         |
| Ablation 03 | 3    |                       |                         | O                    | O                        | 4087         |
| Ablation 04 | 4    | O                     | O                       |                      | O                        | 7029         |
| Ablation 05 | 5    |                       | O                       |                      | O                        | 3981         |
| Ablation 06 | 6    | O                     |                         |                      | O                        | 6577         |
| Ablation 07 | 7    |                       |                         |                      | O                        | 3079         |
| Ablation 08 | 8    | O                     | O                       | O                    |                          | 7449         |
| Ablation 09 | 9    |                       | O                       | O                    |                          | 3882         |
| Ablation 10 | 10   | O                     |                         | O                    |                          | 7378         |
| Ablation 11 | 11   |                       |                         | O                    |                          | 3919         |
| Ablation 12 | 12   | O                     | O                       |                      |                          | 6778         |
| Ablation 13 | 13   |                       | O                       |                      |                          | 3673         |
| Ablation 14 | 14   | O                     |                         |                      |                          | 6375         |
| Ablation 15 | 15   |                       |                         |                      |                          | 2985         |

- Brief comment
  - With `RM=RN=32`, register reuse cuts `ldmatrix` traffic roughly in half.
  - Ideal `ldmatrix` lower bound is therefore about `3.5 ms`, consistent with `Ablation 13 = 3673 us` (only `shared -> register` load).
  - `Ablation 01 = 4010 us` (`shared -> register` + MMA + store) is close to cuBLAS (`4022 us`), so `global -> shared` load is now the dominant remaining overhead in this kernel.
