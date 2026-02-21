// #include <cooperative_groups.h>
#include <chrono>
#include <cuda/cmath>
#include <stdio.h>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t result = expr;                                                                 \
        if (result != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result,     \
                    cudaGetErrorString(result));                                                   \
        }                                                                                          \
    } while (0)

// // cluster requires sm_90 or higher. Current devices do not support it.
// __global__ void clusterHistogram(int *bins, const int nbins, const int bins_per_block,
//                                  const int *__restrict__ input, size_t array_size) {
//     extern __shared__ int smem[];
//     namespace cg = cooperative_groups;

//     int tid_global = cg::this_grid().thread_rank();

//     cg::cluster_group cluster = cg::this_cluster();
//     unsigned int cluster_block_rank = cluster.block_rank();
//     int cluster_size = cluster.dim_blocks().x;

//     for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
//         smem[i] = 0;
//     }

//     cluster.sync();

//     for (int i = tid_global; i < array_size; i += blockDim.x * gridDim.x) {
//         int ldata = input[i];

//         int binid = ldata;
//         if (ldata < 0)
//             binid = 0;
//         else if (ldata >= nbins)
//             binid = nbins - 1;

//         int dst_block_rank = (int)(binid / bins_per_block);
//         int dst_offset = binid % bins_per_block;
//         int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
//         atomicAdd(dst_smem + dst_offset, 1);
//     }

//     cluster.sync();

//     int *lbins = bins + cluster.block_rank() * bins_per_block;
//     for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
//         atomicAdd(&lbins[i], smem[i]);
//     }
// }

// void launchClusterHistogram(int threads_per_block, int cluster_size, int *bins, const int nbins,
//                             const int *__restrict__ input, size_t array_size) {
//     cudaLaunchConfig_t config = {0};
//     config.gridDim = cuda::ceil_div(array_size, threads_per_block);
//     config.blockDim = threads_per_block;

//     int nbins_per_block = cuda::ceil_div(nbins, cluster_size);

//     config.dynamicSmemBytes = nbins_per_block * sizeof(int);

//     CUDA_CHECK(cudaFuncSetAttribute((void *)clusterHistogram,
//                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
//                                     config.dynamicSmemBytes));

//     cudaLaunchAttribute attribute[1];
//     attribute[0].id = cudaLaunchAttributeClusterDimension;
//     attribute[0].val.clusterDim.x = cluster_size;
//     attribute[0].val.clusterDim.y = 1;
//     attribute[0].val.clusterDim.z = 1;

//     config.numAttrs = 1;
//     config.attrs = attribute;

//     cudaLaunchKernelEx(&config, clusterHistogram, bins, nbins, nbins_per_block, input, array_size);
//     CUDA_CHECK(cudaGetLastError());
// }

__global__ void globalHistogram(int *bins, const int nbins, const int *__restrict__ input,
                                size_t array_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;
        atomicAdd(&bins[binid], 1);
    }
}

__global__ void sharedHistogram(int *bins, const int nbins, const int *__restrict__ input,
                                size_t array_size) {
    extern __shared__ int smem[];
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;
        atomicAdd(&smem[binid], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd(&bins[i], smem[i]);
    }
}

void sequentialHistogram(int *bins, const int nbins, const int *__restrict__ input,
                         size_t array_size) {
    memset(bins, 0, nbins * sizeof(int));
    for (size_t i = 0; i < array_size; i++) {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;
        bins[binid]++;
    }
}

void initArray(int *array, size_t size, int min, int max) {
    for (size_t i = 0; i < size; i++) {
        array[i] = rand() % (max - min + 1) + min;
    }
}

int main(int argc, char **argv) {
    // Get array size and nbins as argv
    if (argc != 3) {
        printf("Usage: %s <array_size> <nbins>\n", argv[0]);
        return 1;
    }

    size_t array_size = atoi(argv[1]);
    int nbins = atoi(argv[2]);

    // Initialize input array
    int *input = (int *)malloc(array_size * sizeof(int));
    int *input_dev = nullptr;
    initArray(input, array_size, 0, nbins - 1);
    cudaMalloc(&input_dev, array_size * sizeof(int));
    cudaMemcpy(input_dev, input, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // For tracking latency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sequential histogram
    int *bins_seq = (int *)malloc(nbins * sizeof(int));
    auto t0 = std::chrono::steady_clock::now();
    sequentialHistogram(bins_seq, nbins, input, array_size);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("[Sequential histogram] time: %f ms\n", ms);

    // Global histogram
    int *bins_global = (int *)malloc(nbins * sizeof(int));
    int *bins_global_dev = nullptr;
    cudaMalloc(&bins_global_dev, nbins * sizeof(int));
    int threads_global = 256;
    int blocks_global = cuda::ceil_div(array_size, threads_global);

    int iters = 10;
    for (int i = 0; i < iters; i++) {
        cudaMemset(bins_global_dev, 0, nbins * sizeof(int));
        cudaEventRecord(start);
        globalHistogram<<<blocks_global, threads_global>>>(bins_global_dev, nbins, input_dev,
                                                           array_size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("[Global histogram] Iter %d: time: %f ms\n", i, ms);
    }
    cudaMemcpy(bins_global, bins_global_dev, nbins * sizeof(int), cudaMemcpyDeviceToHost);

    // Shared histogram
    int *bins_shared = (int *)malloc(nbins * sizeof(int));
    int *bins_shared_dev = nullptr;
    cudaMalloc(&bins_shared_dev, nbins * sizeof(int));
    int threads_shared = 256;
    int blocks_shared = cuda::ceil_div(array_size, threads_shared);
    int sharedMemBytes = nbins * sizeof(int);

    CUDA_CHECK(cudaFuncSetAttribute((void *)sharedHistogram,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes));

    iters = 10;
    for (int i = 0; i < iters; i++) {
        cudaMemset(bins_shared_dev, 0, nbins * sizeof(int));
        cudaEventRecord(start);
        sharedHistogram<<<blocks_shared, threads_shared, sharedMemBytes>>>(bins_shared_dev, nbins,
                                                                           input_dev, array_size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("[Shared histogram] Iter %d: time: %f ms\n", i, ms);
    }
    cudaMemcpy(bins_shared, bins_shared_dev, nbins * sizeof(int), cudaMemcpyDeviceToHost);

    // // cluster histogram using dsm
    // int *bins_dsm = (int *)malloc(nbins * sizeof(int));
    // int *bins_dsm_dev = nullptr;
    // cudaMalloc(&bins_dsm_dev, nbins * sizeof(int));
    // cudaMemset(&bins_dsm_dev, 0, nbins * sizeof(int));
    // int cluster_size = 2;
    // int threads_dsm = 256;
    // launchClusterHistogram(threads_dsm, cluster_size, bins_dsm_dev, nbins, input_dev, array_size);
    // cudaMemcpy(bins_dsm, bins_dsm_dev, nbins * sizeof(int), cudaMemcpyDeviceToHost);

    // Compare results
    bool is_correct = true;
    for (int i = 0; i < nbins; i++) {
        // if (bins_seq[i] != bins_dsm[i]) {
        //     printf("Error: bins[%d] = %d, bins_dsm[%d] = %d\n", i, bins_seq[i], i, bins_dsm[i]);
        // }
        if (bins_seq[i] != bins_global[i]) {
            is_correct = false;
            printf("Error: bins[%d] = %d, bins_global[%d] = %d\n", i, bins_seq[i], i,
                   bins_global[i]);
        }
        if (bins_seq[i] != bins_shared[i]) {
            is_correct = false;
            printf("Error: bins[%d] = %d, bins_shared[%d] = %d\n", i, bins_seq[i], i,
                   bins_shared[i]);
        }
    }
    if (is_correct) {
        printf("Results are correct\n");
    }

    free(input);
    free(bins_seq);
    free(bins_global);
    // free(bins_dsm);
    cudaFree(input_dev);
    cudaFree(bins_global_dev);
    // cudaFree(bins_dsm_dev);
    return 0;
}