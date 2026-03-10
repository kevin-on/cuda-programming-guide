#include <cuda/cmath>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdint.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

using namespace nvcuda;

#define INDX(row, col, ld) ((row) * (ld) + (col))
#define SWZ8(row, col, ld) ((row) * (ld) + (((col) >> 3) ^ ((row)&0x7)) * 8)

constexpr int LDSA_PAD = 0;
constexpr int LDSB_PAD = 0;

template <int BM, int BN, int BK, int WM, int WN, int RM, int RN>
__global__ void tileMatmulKernel(bf16 *A, bf16 *B, bf16 *C, int m) {
    /**
     * - Each thread block computes a BM x BN tile of the output matrix C.
     * - Each warp computes a WM x WN tile of the output matrix C.
     * - Each thread should keep WM x WN / 32 elements of the output matrix C in registers.
     * - WM should be a multiple of 16 & WN should be a multiple of 16 as we use following
     * instructions:
     *   - ldmatrix.sync.aligned.m8n8.x4.shared.b16
     *   - mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
     * - # of warps needed = (BM / WM) * (BN / WN)
     * - # of threads per block = 32 * (BM / WM) * (BN / WN) = 32 * BM * BN / (WM * WN)
     * - gridDim: (m / BN, m / BM)
     * - blockDim: BM * BN / (WM * WN)
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * Note: cuBLAS=271.32 TFLOPS, theoretical peak=312 TFLOPS
     *       BM=128, BN=128, BK=32, WM=64, WN=64: 117.41 TFLOPS
     */

    constexpr int ldsa = BK + LDSA_PAD;
    constexpr int ldsb = BN + LDSB_PAD;
    constexpr size_t sABytes = 2ULL * BM * ldsa * sizeof(bf16);
    constexpr size_t sABytesAligned = ((sABytes + 15) / 16) * 16;
    extern __shared__ __align__(16) unsigned char smem[];
    bf16 *sA = reinterpret_cast<bf16 *>(smem);
    bf16 *sB = reinterpret_cast<bf16 *>(smem + sABytesAligned);

    int bx0 = blockIdx.x * BN;
    int by0 = blockIdx.y * BM;
    int warpIdx = threadIdx.x / 32; // 1d warp index in 2d grid (BM / WM) x (BN / WN)
    int laneId = threadIdx.x % 32;
    int wy0 = WM * (warpIdx / (BN / WN));
    int wx0 = WN * (warpIdx % (BN / WN));

    bf16 a_frag[(RM / 16)][8];
    bf16 b_frag[(RN / 16)][8];
    float c_frag[(WM / 16)][(WN / 16)][8] = {0.0f};

    int iterCount = 0;
    for (int k0 = 0; k0 < m; k0 += BK, iterCount++) {
        bf16 *readSA = &sA[(iterCount & 1) * BM * ldsa];
        bf16 *readSB = &sB[(iterCount & 1) * BK * ldsb];
        bf16 *writeSA = &sA[((iterCount + 1) & 1) * BM * ldsa];
        bf16 *writeSB = &sB[((iterCount + 1) & 1) * BK * ldsb];

// /* -- DRAM -> SRAM -- */
#ifdef DEBUG_SKIP_G2S_LOAD
        // Do nothing
#else
        // First iteration: load the first tile
        if (iterCount == 0) {
            for (int idx = threadIdx.x; idx < BM * (BK / 8); idx += blockDim.x) {
                int ay = idx / (BK / 8);
                int ax = (idx % (BK / 8)) * 8;
                __pipeline_memcpy_async(&readSA[SWZ8(ay, ax, ldsa)], &A[INDX(by0 + ay, k0 + ax, m)],
                                        8 * sizeof(bf16));
            }
            for (int idx = threadIdx.x; idx < BK * (BN / 8); idx += blockDim.x) {
                int by = idx / (BN / 8);
                int bx = (idx % (BN / 8)) * 8;
                __pipeline_memcpy_async(&readSB[SWZ8(by, bx, ldsb)], &B[INDX(k0 + by, bx0 + bx, m)],
                                        8 * sizeof(bf16));
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        // Prefetch next tile
        int nextK0 = k0 + BK;
        if (nextK0 < m) {
            for (int idx = threadIdx.x; idx < BM * (BK / 8); idx += blockDim.x) {
                int ay = idx / (BK / 8);
                int ax = (idx % (BK / 8)) * 8;
                __pipeline_memcpy_async(&writeSA[SWZ8(ay, ax, ldsa)],
                                        &A[INDX(by0 + ay, nextK0 + ax, m)], 8 * sizeof(bf16));
            }
            for (int idx = threadIdx.x; idx < BK * (BN / 8); idx += blockDim.x) {
                int by = idx / (BN / 8);
                int bx = (idx % (BN / 8)) * 8;
                __pipeline_memcpy_async(&writeSB[SWZ8(by, bx, ldsb)],
                                        &B[INDX(nextK0 + by, bx0 + bx, m)], 8 * sizeof(bf16));
            }
        }
        __pipeline_commit();
#endif

        /* -- SRAM -> Register  -- */
        /* -- Register @ Register => Register  -- */
        uint32_t pRow = laneId % 16;
        uint32_t pCol = (laneId < 16) ? 0 : 8;
#ifdef DEBUG_SKIP_MMA
        uint32_t ldsm_keep = 0;
#endif
        for (int k = 0; k < BK; k += 16) {
            for (int ty0 = 0; ty0 < WM; ty0 += RM) {
                for (int tx0 = 0; tx0 < WN; tx0 += RN) {
                    for (int raIdx = 0; raIdx < (RM / 16); raIdx++) {
                        uint32_t *ra_ptr = reinterpret_cast<uint32_t *>(&a_frag[raIdx][0]);
                        int ry0 = raIdx * 16;
                        uint32_t pa = __cvta_generic_to_shared(
                            &readSA[SWZ8(wy0 + ty0 + ry0 + pRow, k + pCol, ldsa)]);
#ifdef DEBUG_SKIP_LDMATRIX
                        ra_ptr[0] = pa;
                        ra_ptr[1] = pa ^ 0x11111111u;
                        ra_ptr[2] = pa ^ 0x22222222u;
                        ra_ptr[3] = pa ^ 0x33333333u;
                        asm volatile(""
                                     : "+r"(ra_ptr[0]), "+r"(ra_ptr[1]), "+r"(ra_ptr[2]),
                                       "+r"(ra_ptr[3]));
#else
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                            : "=r"(ra_ptr[0]), "=r"(ra_ptr[1]), "=r"(ra_ptr[2]), "=r"(ra_ptr[3])
                            : "r"(pa));
#endif
                    }

                    for (int rbIdx = 0; rbIdx < (RN / 16); rbIdx++) {
                        uint32_t *rb_ptr = reinterpret_cast<uint32_t *>(&b_frag[rbIdx][0]);
                        int rx0 = rbIdx * 16;
                        uint32_t pb = __cvta_generic_to_shared(
                            &readSB[SWZ8(k + pRow, wx0 + tx0 + rx0 + pCol, ldsb)]);
#ifdef DEBUG_SKIP_LDMATRIX
                        rb_ptr[0] = pb;
                        rb_ptr[1] = pb ^ 0x11111111u;
                        rb_ptr[2] = pb ^ 0x22222222u;
                        rb_ptr[3] = pb ^ 0x33333333u;
                        asm volatile(""
                                     : "+r"(rb_ptr[0]), "+r"(rb_ptr[1]), "+r"(rb_ptr[2]),
                                       "+r"(rb_ptr[3]));
#else
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                            : "=r"(rb_ptr[0]), "=r"(rb_ptr[1]), "=r"(rb_ptr[2]), "=r"(rb_ptr[3])
                            : "r"(pb));
#endif
                    }

                    for (int raIdx = 0; raIdx < (RM / 16); raIdx++) {
                        for (int rbIdx = 0; rbIdx < (RN / 16); rbIdx++) {
                            int ry0 = raIdx * 16;
                            int rx0 = rbIdx * 16;
                            uint32_t *ra_ptr = reinterpret_cast<uint32_t *>(&a_frag[raIdx][0]);
                            uint32_t *rb_ptr = reinterpret_cast<uint32_t *>(&b_frag[rbIdx][0]);
                            float *c_frag_ptr = &c_frag[(ty0 + ry0) / 16][(tx0 + rx0) / 16][0];
#ifdef DEBUG_SKIP_MMA
                            ldsm_keep ^= (ra_ptr[0] ^ ra_ptr[1] ^ rb_ptr[0] ^ rb_ptr[1]);
                            ldsm_keep ^= (ra_ptr[2] ^ ra_ptr[3] ^ rb_ptr[2] ^ rb_ptr[3]);
#else
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(c_frag_ptr[0]), "+f"(c_frag_ptr[1]),
                                           "+f"(c_frag_ptr[2]), "+f"(c_frag_ptr[3])
                                         : "r"(ra_ptr[0]), "r"(ra_ptr[1]), "r"(ra_ptr[2]),
                                           "r"(ra_ptr[3]), "r"(rb_ptr[0]), "r"(rb_ptr[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(c_frag_ptr[4]), "+f"(c_frag_ptr[5]),
                                           "+f"(c_frag_ptr[6]), "+f"(c_frag_ptr[7])
                                         : "r"(ra_ptr[0]), "r"(ra_ptr[1]), "r"(ra_ptr[2]),
                                           "r"(ra_ptr[3]), "r"(rb_ptr[2]), "r"(rb_ptr[3]));
#endif
                        }
                    }
                }
            }
        }
#ifdef DEBUG_SKIP_MMA
        if (laneId == 0) {
            volatile uint32_t *sink = reinterpret_cast<volatile uint32_t *>(sA);
            sink[warpIdx] = ldsm_keep;
        }
#endif
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    /* -- Register -> (Cache) -> DRAM  -- */
    int groupId = laneId >> 2;
    int threadId_in_group = laneId % 4;

    for (int ty0 = 0; ty0 < WM; ty0 += 16) {
        for (int tx0 = 0; tx0 < WN; tx0 += 16) {
            float *c_frag_ptr = &c_frag[(ty0 / 16)][(tx0 / 16)][0];
#ifdef DEBUG_SKIP_R2G_STORE
            volatile float *sink = reinterpret_cast<volatile float *>(sA);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += c_frag_ptr[j];
            if (laneId == 0)
                sink[warpIdx] = acc;
            asm volatile("" ::: "memory");
#else
            for (int i = 0; i < 8; i++) {
                uint32_t row = ((i % 4) < 2) ? groupId : groupId + 8;
                uint32_t col = (threadId_in_group * 2) + (i & 0x1) + (i < 4 ? 0 : 8);
                C[INDX(by0 + wy0 + ty0 + row, bx0 + wx0 + tx0 + col, m)] =
                    __float2bfloat16(c_frag_ptr[i]);
            }
#endif
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int RM, int RN>
void launchTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(32 * BM * BN / (WM * WN));
    dim3 grid(cuda::ceil_div(m, BN), cuda::ceil_div(m, BM));
    constexpr size_t sABytes = 2ULL * BM * (BK + LDSA_PAD) * sizeof(bf16);
    constexpr size_t sABytesAligned = ((sABytes + 15) / 16) * 16;
    constexpr size_t sBBytes = 2ULL * BK * (BN + LDSB_PAD) * sizeof(bf16);
    constexpr size_t smemBytes = sABytesAligned + sBBytes;
    CUDA_CHECK(cudaFuncSetAttribute(tileMatmulKernel<BM, BN, BK, WM, WN, RM, RN>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smemBytes)));

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(bf16))); };
    auto fetch = [&]() { return std::vector<bf16>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
    auto result = runKernelBenchmark<std::vector<bf16>>(
        [&]() {
            tileMatmulKernel<BM, BN, BK, WM, WN, RM, RN>
                <<<grid, block, smemBytes>>>(ctx.A, ctx.B, ctx.C, m);
        },
        reset, fetch, ctx.warmup, ctx.iters, stats);

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
    int bm, bn, bk, wm, wn, rm, rn;
    TileFn fn;
};

#define TILE_CFG(BM, BN, BK, WM, WN, RM, RN)                                                       \
    { BM, BN, BK, WM, WN, RM, RN, launchTileMatmul<BM, BN, BK, WM, WN, RM, RN> }

static const TileConfig tileConfigs[] = {
    TILE_CFG(128, 128, 32, 64, 64, 32, 32),
    TILE_CFG(128, 128, 64, 64, 64, 32, 32),
};

void runTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int bm = spec.at("bm"), bn = spec.at("bn"), bk = spec.at("bk");
    int wm = spec.at("wm"), wn = spec.at("wn"), rm = spec.at("rm"), rn = spec.at("rn");
    for (auto &cfg : tileConfigs) {
        if (cfg.bm == bm && cfg.bn == bn && cfg.bk == bk && cfg.wm == wm && cfg.wn == wn &&
            cfg.rm == rm && cfg.rn == rn) {
            cfg.fn(ctx, spec);
            return;
        }
    }
    fprintf(stderr, "No compiled config for bm=%d bn=%d bk=%d wm=%d wn=%d rm=%d rn=%d\n", bm, bn,
            bk, wm, wn, rm, rn);
}
