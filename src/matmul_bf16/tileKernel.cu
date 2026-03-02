#include <cuda/cmath>
#include <cuda_pipeline.h>
#include <mma.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

using namespace nvcuda;

#define INDX(row, col, ld) ((row) * (ld) + (col))

template <int BM, int BN, int BK>
__global__ void tileMatmulKernel(bf16 *A, bf16 *B, bf16 *C, int m) {
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
     * BM=64, BN=64, BK=64: 41.34 TFLOPS
     */

    // Note: leading dimensions should be multiple of 16 bytes for wmma::load_matrix_sync and
    // wmma::store_matrix_sync.
    constexpr int ldsa = BK + 8;
    constexpr int ldsb = BN + 8;
    constexpr int ldsc = BN + 8; // +8 is slightly better than +4. TODO: Investigate why.
    __shared__ bf16 sA[BM * ldsa];
    __shared__ bf16 sB[BK * ldsb];
    __shared__ float sC[BM * ldsc];

    int bx0 = blockIdx.x * BN;
    int by0 = blockIdx.y * BM;
    int warpIdx = threadIdx.x / 32; // 1d warp index in 2d grid (BM / 16) * (BN / 16)
    int wy0 = 16 * (warpIdx / (BN / 16));
    int wx0 = 16 * (warpIdx % (BN / 16));

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < m; k0 += BK) {

        /* -- DRAM -> SRAM -- */
        // __pipeline_memcpy_async requires the size parameter to be 4, 8, or 16 bytes.
        for (int idx = threadIdx.x; idx < BM * (BK / 2); idx += blockDim.x) {
            int ay = idx / (BK / 2);
            int ax = (idx % (BK / 2)) * 2;
            __pipeline_memcpy_async(&sA[INDX(ay, ax, ldsa)], &A[INDX(by0 + ay, k0 + ax, m)],
                                    2 * sizeof(bf16));
        }
        for (int idx = threadIdx.x; idx < BK * (BN / 2); idx += blockDim.x) {
            int by = idx / (BN / 2);
            int bx = (idx % (BN / 2)) * 2;
            __pipeline_memcpy_async(&sB[INDX(by, bx, ldsb)], &B[INDX(k0 + by, bx0 + bx, m)],
                                    2 * sizeof(bf16));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        /* -- SRAM -> Register  -- */
        /* -- Register @ Register => Register  -- */
        for (int k = 0; k < BK; k += 16) {
            wmma::load_matrix_sync(a_frag, &sA[INDX(wy0, k, ldsa)], ldsa);
            wmma::load_matrix_sync(b_frag, &sB[INDX(k, wx0, ldsb)], ldsb);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    /* -- Register -> (Cache) -> DRAM  -- */
    wmma::store_matrix_sync(&sC[INDX(wy0, wx0, ldsc)], c_frag, ldsc, wmma::mem_row_major);
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            C[INDX(by0 + wy0 + y, bx0 + wx0 + x, m)] =
                __float2bfloat16(sC[INDX(wy0 + y, wx0 + x, ldsc)]);
        }
    }
}

template <int BM, int BN, int BK>
void launchTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(BM * BN / 8);
    dim3 grid(cuda::ceil_div(m, BN), cuda::ceil_div(m, BM));

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(bf16))); };
    auto fetch = [&]() { return std::vector<bf16>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
    auto result = runKernelBenchmark<std::vector<bf16>>(
        [&]() { tileMatmulKernel<BM, BN, BK><<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); }, reset,
        fetch, ctx.warmup, ctx.iters, stats);

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqualBf16(result.data(), ctx.ref, ctx.numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }
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
