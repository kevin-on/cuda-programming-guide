#include <cuda/cmath>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdint.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

using namespace nvcuda;

#define INDX(row, col, ld) ((row) * (ld) + (col))

template <int BM, int BN, int BK>
__global__ void
tileMatmulKernel(bf16 *A, bf16 *B, bf16 *C, int m, uint64_t *d_prefetch_debug = nullptr,
                 uint64_t *d_sram_to_reg_debug = nullptr, uint64_t *d_wait_debug = nullptr) {
    /**
     * IMPORTANT: Assume BM = BN = BK for simplicity. Otherwise, the kernel will not work.
     *
     * - Each thread block computes a BM x BN tile of the output matrix C.
     * - A single warp (32 threads) computes a 16x16 tile of the output matrix C.
     * - # of warps needed = (BM / 16) * (BN / 16)
     * - # of threads per block = 32 * (BM / 16) * (BN / 16) = BM * BN / 8
     * - gridDim: (m / BN, m / BM)
     * - blockDim: BM * BN / 8
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * Note: cuBLAS=271.32 TFLOPS, theoretical peak=312 TFLOPS
     * BM=64, BN=64, BK=64: 77.92 TFLOPS
     */

    // Note: leading dimensions should be multiple of 16 bytes for wmma::load_matrix_sync and
    // wmma::store_matrix_sync.
    constexpr int ldsa = BK + 8;
    constexpr int ldsb = BN + 8;
    __shared__ bf16 sA[2 * BM * ldsa];
    __shared__ bf16 sB[2 * BK * ldsb];

    int bx0 = blockIdx.x * BN;
    int by0 = blockIdx.y * BM;
    int warpIdx = threadIdx.x / 32; // 1d warp index in 2d grid (BM / 16) * (BN / 16)
    int wy0 = 16 * (warpIdx / (BN / 16));
    int wx0 = 16 * (warpIdx % (BN / 16));

    bf16 a_frag[8];
    bf16 b_frag[8];
    float c_frag[8] = {0.0f};
    uint32_t *ra = reinterpret_cast<uint32_t *>(a_frag);
    uint32_t *rb = reinterpret_cast<uint32_t *>(b_frag);

    uint32_t laneId = threadIdx.x % 32;

    int iterCount = 0;
#ifdef DEBUG
    uint64_t total_prefetch_cycles = 0;
    uint64_t total_sram_to_reg_cycles = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t sample_count = 0;
#endif
    for (int k0 = 0; k0 < m; k0 += BK, iterCount++) {
        // /* -- DRAM -> SRAM -- */
        // Initialize SRAM with the first tile
        if (iterCount == 0) {
            for (int idx = threadIdx.x; idx < BM * (BK / 2); idx += blockDim.x) {
                int ay = idx / (BK / 2);
                int ax = (idx % (BK / 2)) * 2;
                __pipeline_memcpy_async(&sA[INDX(ay, ax, ldsa) + (iterCount & 1) * BM * ldsa],
                                        &A[INDX(by0 + ay, k0 + ax, m)], 2 * sizeof(bf16));
            }
            for (int idx = threadIdx.x; idx < BK * (BN / 2); idx += blockDim.x) {
                int by = idx / (BN / 2);
                int bx = (idx % (BN / 2)) * 2;
                __pipeline_memcpy_async(&sB[INDX(by, bx, ldsb) + (iterCount & 1) * BK * ldsb],
                                        &B[INDX(k0 + by, bx0 + bx, m)], 2 * sizeof(bf16));
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        // Prefetch next tile
        int nextK0 = k0 + BK;
#ifdef DEBUG
        uint64_t prefetch_t0 = 0;
        if (laneId == 0)
            prefetch_t0 = clock64();
#endif
        if (nextK0 < m) {
            for (int idx = threadIdx.x; idx < BM * (BK / 8); idx += blockDim.x) {
                int ay = idx / (BK / 8);
                int ax = (idx % (BK / 8)) * 8;
                __pipeline_memcpy_async(&sA[INDX(ay, ax, ldsa) + ((iterCount + 1) & 1) * BM * ldsa],
                                        &A[INDX(by0 + ay, nextK0 + ax, m)], 8 * sizeof(bf16));
            }
            for (int idx = threadIdx.x; idx < BK * (BN / 8); idx += blockDim.x) {
                int by = idx / (BN / 8);
                int bx = (idx % (BN / 8)) * 8;
                __pipeline_memcpy_async(&sB[INDX(by, bx, ldsb) + ((iterCount + 1) & 1) * BK * ldsb],
                                        &B[INDX(nextK0 + by, bx0 + bx, m)], 8 * sizeof(bf16));
            }
        }
        __pipeline_commit();
#ifdef DEBUG
        if (laneId == 0) {
            uint64_t prefetch_t1 = clock64();
            total_prefetch_cycles += prefetch_t1 - prefetch_t0;
        }
#endif

        /* -- SRAM -> Register  -- */
        /* -- Register @ Register => Register  -- */
#ifdef DEBUG
        uint64_t sram_to_reg_t0 = 0;
        if (laneId == 0)
            sram_to_reg_t0 = clock64();
#endif
        for (int k = 0; k < BK; k += 16) {
            bf16 *curSA = &sA[(iterCount & 1) * BM * ldsa];
            bf16 *curSB = &sB[(iterCount & 1) * BK * ldsb];
            uint32_t pRow = laneId % 16;
            uint32_t pCol = (laneId < 16) ? 0 : 8;
            uint32_t pa = __cvta_generic_to_shared(&curSA[INDX(wy0 + pRow, k + pCol, ldsa)]);
            uint32_t pb = __cvta_generic_to_shared(&curSB[INDX(k + pRow, wx0 + pCol, ldsb)]);

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                         : "r"(pa));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(rb[0]), "=r"(rb[1]), "=r"(rb[2]), "=r"(rb[3])
                         : "r"(pb));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3])
                         : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c_frag[4]), "+f"(c_frag[5]), "+f"(c_frag[6]), "+f"(c_frag[7])
                         : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[2]), "r"(rb[3]));
        }
#ifdef DEBUG
        if (laneId == 0) {
            uint64_t sram_to_reg_t1 = clock64();
            total_sram_to_reg_cycles += sram_to_reg_t1 - sram_to_reg_t0;
        }
#endif

#ifdef DEBUG
        uint64_t wait_t0 = 0;
        if (laneId == 0)
            wait_t0 = clock64();
#endif
        __pipeline_wait_prior(0);
#ifdef DEBUG
        if (laneId == 0) {
            uint64_t wait_t1 = clock64();
            total_wait_cycles += wait_t1 - wait_t0;
            sample_count++;
        }
#endif
        __syncthreads();
    }

#ifdef DEBUG
    int warpsPerBlock = blockDim.x / 32;
    int blockLinear = blockIdx.x + blockIdx.y * gridDim.x;
    int globalWarp = blockLinear * warpsPerBlock + warpIdx;
    if (laneId == 0 && sample_count > 0) {
        d_prefetch_debug[globalWarp] += total_prefetch_cycles / sample_count;
        d_sram_to_reg_debug[globalWarp] += total_sram_to_reg_cycles / sample_count;
        d_wait_debug[globalWarp] += total_wait_cycles / sample_count;
    }
#endif

    /* -- Register -> (Cache) -> DRAM  -- */
    uint32_t groupId = laneId >> 2;
    uint32_t threadId_in_group = laneId % 4;

    for (int i = 0; i < 8; i++) {
        uint32_t row = ((i % 4) < 2) ? groupId : groupId + 8;
        uint32_t col = (threadId_in_group * 2) + (i & 0x1) + (i < 4 ? 0 : 8);
        C[INDX(by0 + wy0 + row, bx0 + wx0 + col, m)] = __float2bfloat16(c_frag[i]);
    }
}

template <int BM, int BN, int BK>
void launchTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(BM * BN / 8);
    dim3 grid(cuda::ceil_div(m, BN), cuda::ceil_div(m, BM));

#ifdef DEBUG
    uint64_t *d_prefetch_debug;
    uint64_t *d_sram_to_reg_debug;
    uint64_t *d_wait_debug;
    int numBlocks = grid.x * grid.y;
    int warpsPerBlock = block.x / 32;
    int numWarps = numBlocks * warpsPerBlock;

    cudaMallocManaged(&d_prefetch_debug, numWarps * sizeof(uint64_t));
    cudaMallocManaged(&d_sram_to_reg_debug, numWarps * sizeof(uint64_t));
    cudaMallocManaged(&d_wait_debug, numWarps * sizeof(uint64_t));
    cudaMemset(d_prefetch_debug, 0, numWarps * sizeof(uint64_t));
    cudaMemset(d_sram_to_reg_debug, 0, numWarps * sizeof(uint64_t));
    cudaMemset(d_wait_debug, 0, numWarps * sizeof(uint64_t));
#endif

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(bf16))); };
    auto fetch = [&]() { return std::vector<bf16>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
#ifdef DEBUG
    int launch_count = 0;
    auto result = runKernelBenchmark<std::vector<bf16>>(
        [&]() {
            ++launch_count;
            tileMatmulKernel<BM, BN, BK><<<grid, block>>>(ctx.A, ctx.B, ctx.C, m, d_prefetch_debug,
                                                          d_sram_to_reg_debug, d_wait_debug);
        },
        reset, fetch, ctx.warmup, ctx.iters, stats);
#else
    auto result = runKernelBenchmark<std::vector<bf16>>(
        [&]() { tileMatmulKernel<BM, BN, BK><<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); }, reset,
        fetch, ctx.warmup, ctx.iters, stats);
#endif

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqualBf16(result.data(), ctx.ref, ctx.numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }

#ifdef DEBUG
    uint64_t total_prefetch_cycles = 0;
    uint64_t total_sram_to_reg_cycles = 0;
    uint64_t total_wait_cycles = 0;

    for (int i = 0; i < numWarps; i++) {
        uint64_t prefetch_avg = d_prefetch_debug[i] / launch_count;
        uint64_t sram_to_reg_avg = d_sram_to_reg_debug[i] / launch_count;
        uint64_t wait_avg = d_wait_debug[i] / launch_count;

        total_prefetch_cycles += prefetch_avg;
        total_sram_to_reg_cycles += sram_to_reg_avg;
        total_wait_cycles += wait_avg;
    }

    printf("Warp(lane0)-scope avg cycles: prefetch=%lu, sram_to_reg+mmas=%lu, wait_prior=%lu\n",
           total_prefetch_cycles / numWarps, total_sram_to_reg_cycles / numWarps,
           total_wait_cycles / numWarps);

    cudaFree(d_prefetch_debug);
    cudaFree(d_sram_to_reg_debug);
    cudaFree(d_wait_debug);
#endif
}

// --- Config dispatch table ---

using TileFn = void (*)(const MatmulBenchCtx &, const KernelSpec &);

struct TileConfig {
    int bm, bn, bk;
    TileFn fn;
};

#define TILE_CFG(BM, BN, BK)                                                                       \
    { BM, BN, BK, launchTileMatmul<BM, BN, BK> }

static const TileConfig tileConfigs[] = {
    TILE_CFG(64, 64, 64),
    // TILE_CFG(64, 64, 32),
};

void runTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int bm = spec.at("bm"), bn = spec.at("bn"), bk = spec.at("bk");
    for (auto &cfg : tileConfigs) {
        if (cfg.bm == bm && cfg.bn == bn && cfg.bk == bk) {
            cfg.fn(ctx, spec);
            return;
        }
    }
    fprintf(stderr, "No compiled config for bm=%d bn=%d bk=%d\n", bm, bn, bk);
}
