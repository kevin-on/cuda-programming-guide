# Matmul BF16 Ablation Experiments (`tileKernel.cu`)

- Target kernel: `[src/matmul_bf16/tileKernel.cu](../../src/matmul_bf16/tileKernel.cu)`
- Rule: `O` means the stage is included in that experiment. Blank means ablated (skipped).
- Fill in `Latency (us)` after each run.

## Benchmark Configuration

- `m=8192`
- `warmup=5`, `iters=10`, `kernels=1`
- Tile config: `BM=128`, `BN=128`, `BK=32`, `WM=64`, `WN=64`


| Experiment  | Global -> Shared Load | Shared -> Register Load | MMA (Tensor Core Op) | Register -> Global Store | Latency (us) |
| ----------- | --------------------- | ----------------------- | -------------------- | ------------------------ | ------------ |
| cuBLAS      |                       |                         |                      |                          | 4027         |
| MMA+R2G     |                       |                         | O                    | O                        | 4075         |
| S2R+R2G     |                       | O                       |                      | O                        | 7599         |
| S2R+MMA+R2G |                       | O                       | O                    | O                        | 9552         |
| G2S+MMA+R2G | O                     |                         | O                    | O                        | 7046         |
| G2S+S2R+R2G | O                     | O                       |                      | O                        | 10110        |
| G2S+S2R+MMA+R2G | O                 | O                       | O                    | O                        | 11628        |
