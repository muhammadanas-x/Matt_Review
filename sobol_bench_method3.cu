

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

constexpr uint32_t BENCH_ITERS = 256;
constexpr uint32_t WORKGROUP_SIZE = 256;
constexpr uint32_t DEPTH = 8;

constexpr uint32_t TRIPLETS = 2;
constexpr uint32_t COMPONENTS = 3;
constexpr uint32_t TEST_BATCH_COUNT = 256;
constexpr uint32_t SLOTS_PER_DEPTH = TRIPLETS * COMPONENTS; // == 6, matches "DEPTH * 6" in HLSL

constexpr uint32_t PACKED_MATRIX_PAIRS_PER_DEPTH = SLOTS_PER_DEPTH / 2;
constexpr uint32_t NUM_PACKED_MATRIX_PAIRS = DEPTH * PACKED_MATRIX_PAIRS_PER_DEPTH;

#define CUDA_CHECK(x)                                                     \
    do {                                                                  \
        cudaError_t err = (x);                                             \
        if (err != cudaSuccess) {                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                              \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

struct PushConstants
{
    uint64_t pSampleBuffer;        
    uint32_t sequenceSamplesLog2;
    uint32_t depthStaggerMask;
    float probAtDepth0;
    float probAtDepthMax;
};

struct uint2_32
{
    uint32_t x, y;
};

__device__ __forceinline__ uint32_t bit_reverse_32(uint32_t x)
{
    return __brev(x);
}

__device__ __forceinline__ uint32_t parity32(uint32_t x)
{
    return __popc(x) & 1u;
}

__device__ __constant__ uint32_t g_packedMatrixPairs[NUM_PACKED_MATRIX_PAIRS][16];
static uint32_t h_packedMatrixPairs[NUM_PACKED_MATRIX_PAIRS][16];

__device__ __forceinline__ uint16_t col_major_matrix_apply(
    const uint16_t* cols,
    const uint16_t sampleIx)
{
    uint16_t val = 0u;

#pragma unroll
    for (uint16_t i = 0; i < 16; i++)
    {
        const uint16_t keep = uint16_t(0u - uint16_t((sampleIx >> i) & 1u));
        val ^= uint16_t(cols[i] & keep);
    }

    return val;
}

__device__ __forceinline__ uint32_t warp_col_major_matrix_apply_2d(
    const uint32_t* packedCols,
    const uint16_t sampleIx)
{
    constexpr uint32_t FULL_MASK = 0xffffffffu;
    const uint32_t laneID = threadIdx.x & 31u;

    const uint32_t myPackedCols =
        laneID < 16u ? packedCols[laneID] : 0u;

    uint32_t packedVal = 0u;

#pragma unroll
    for (uint16_t i = 0; i < 16; ++i)
    {
        const uint32_t packedCol =
            __shfl_sync(FULL_MASK, myPackedCols, int(i));
        const uint32_t keep =
            0u - uint32_t((sampleIx >> i) & 1u);
        packedVal ^= packedCol & keep;
    }

    return packedVal;
}

struct QuantizedSequence3
{
    uint32_t data[2];

    static constexpr uint32_t BitsPerComponent = 21u;
    static constexpr uint32_t DiscardBits = 11u;

    __device__ __host__ static __forceinline__ uint32_t bitfieldExtractU32(
        const uint32_t x, const uint32_t offset, const uint32_t bits)
    {
        return (x >> offset) & ((1u << bits) - 1u);
    }

    __device__ __host__ static __forceinline__ uint32_t bitfieldInsertU32(
        const uint32_t base, const uint32_t insert, const uint32_t offset, const uint32_t bits)
    {
        const uint32_t fieldMask = ((1u << bits) - 1u) << offset;
        return (base & ~fieldMask) | ((insert << offset) & fieldMask);
    }

    __device__ __host__ __forceinline__ uint32_t get32(const int idx) const
    {
        if (idx == 0)
            return bitfieldExtractU32(data[0], 0u, BitsPerComponent);
        else if (idx == 1)
        {
            uint32_t y = bitfieldExtractU32(data[0], BitsPerComponent, DiscardBits);
            y |= bitfieldExtractU32(data[1], 0u, BitsPerComponent - DiscardBits) << DiscardBits;
            return y;
        }
        else
            return bitfieldExtractU32(data[1], BitsPerComponent - DiscardBits, BitsPerComponent);
    }

    __device__ __host__ __forceinline__ uint16_t get(const int idx) const
    {
        return uint16_t(get32(idx));
    }

    __device__ __host__ __forceinline__ void set(const int idx, const uint32_t value)
    {
        const uint32_t truncVal = value >> DiscardBits;

        if (idx == 0)
            data[0] = bitfieldInsertU32(data[0], truncVal, 0u, BitsPerComponent);
        else if (idx == 1)
        {
            data[0] = bitfieldInsertU32(data[0], truncVal, BitsPerComponent, DiscardBits);
            data[1] = bitfieldInsertU32(data[1], truncVal >> DiscardBits, 0u, BitsPerComponent - DiscardBits);
        }
        else
            data[1] = bitfieldInsertU32(data[1], truncVal, BitsPerComponent - DiscardBits, BitsPerComponent);
    }
};

struct Xoroshiro64Star
{
    uint32_t s0;
    uint32_t s1;

    __device__ __forceinline__ static Xoroshiro64Star construct(uint2_32 seed)
    {
        Xoroshiro64Star rng;
        rng.s0 = seed.x;
        rng.s1 = seed.y;
        return rng;
    }

    __device__ __forceinline__ static uint32_t rotl(uint32_t x, int k)
    {
        return (x << k) | (x >> (32 - k));
    }

    __device__ __forceinline__ uint32_t operator()()
    {
        uint32_t result = s0 * 0x9E3779BBu;
        s1 ^= s0;
        s0 = rotl(s0, 26) ^ s1 ^ (s1 << 9);
        s1 = rotl(s1, 13);
        return result;
    }
};

__global__ void sobol_bench_kernel(
    uint32_t* output,
    const uint2_32* seedTexture,
    const QuantizedSequence3* pSampleBuffer, 
    PushConstants pc)
{
    const uint32_t invID = blockIdx.x * blockDim.x + threadIdx.x;

    const uint16_t rank = uint16_t(bit_reverse_32(invID) >> 16);

    const uint16_t laneOffset =
        uint16_t(invID & pc.depthStaggerMask) & uint16_t(DEPTH - 1u);

    const float thresholdScale = 65535.0f;
    float threshold = pc.probAtDepth0 * thresholdScale;

    const float thresholdDelta =
        (pc.probAtDepthMax - pc.probAtDepth0) * thresholdScale / float(DEPTH - 1);

    uint16_t seed = uint16_t(0xdeadu);

    for (uint16_t s = 0; s < BENCH_ITERS; s++)
    {
        float perSampleThreshold = threshold;

        for (uint16_t d = 0; d < DEPTH; d++)
        {
           
            const uint16_t dIdx = uint16_t(d & uint16_t(DEPTH - 1u));

            uint16_t triples[TRIPLETS][COMPONENTS];

#pragma unroll
            for (uint16_t pair = 0; pair < PACKED_MATRIX_PAIRS_PER_DEPTH; ++pair)
            {
                const uint32_t packedPair = warp_col_major_matrix_apply_2d(
                    g_packedMatrixPairs[uint32_t(dIdx) * PACKED_MATRIX_PAIRS_PER_DEPTH + pair],
                    uint16_t(s ^ rank));

                const uint16_t linear0 = uint16_t(pair * 2u);
                const uint16_t linear1 = uint16_t(linear0 + 1u);
                triples[linear0 / COMPONENTS][linear0 % COMPONENTS] =
                    uint16_t(packedPair);
                triples[linear1 / COMPONENTS][linear1 % COMPONENTS] =
                    uint16_t(packedPair >> 16);
            }

            seed ^= triples[0][0];
            seed += triples[0][1];
            seed ^= triples[0][2];

            seed += triples[1][0];
            seed ^= triples[1][1];
            seed += triples[1][2];

            if (triples[0][2] < uint16_t(perSampleThreshold))
                break;

            perSampleThreshold += thresholdDelta;
        }
    }

    output[invID] = uint32_t(seed);
}

static uint32_t ceil_log2_u32(uint32_t x)
{
    uint32_t r = 0;
    x--;
    while (x > 0)
    {
        x >>= 1;
        r++;
    }
    return r;
}

int main()
{
    printf("CUDA Sobol exact-port benchmark\n");
    printf("METHOD=3, BENCH_ITERS=%d, WORKGROUP_SIZE=%d, DEPTH=%d\n",
           BENCH_ITERS, WORKGROUP_SIZE, DEPTH);

    static_assert((BENCH_ITERS & (BENCH_ITERS - 1)) == 0,
                  "BENCH_ITERS must be power of two");

    const uint32_t totalThreadsPerDispatch = TEST_BATCH_COUNT * WORKGROUP_SIZE;
    const uint32_t blocks = TEST_BATCH_COUNT;
    const uint32_t threads = WORKGROUP_SIZE;

    const uint32_t sequenceEntryCount = DEPTH * TRIPLETS;
    const uint32_t sequenceSamplesLog2 = ceil_log2_u32(BENCH_ITERS);

    const size_t outputBytes = sizeof(uint32_t) * totalThreadsPerDispatch;

    uint32_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    const uint32_t SeedDim = 512;
    const size_t seedEntries = SeedDim * SeedDim;

    uint2_32* h_seedTexture = (uint2_32*)malloc(sizeof(uint2_32) * seedEntries);
    for (size_t i = 0; i < seedEntries; i++)
    {
        h_seedTexture[i].x = uint32_t(0xbadc0ffeu + i * 1664525u);
        h_seedTexture[i].y = uint32_t(0x12345678u + i * 1013904223u);
    }

    uint2_32* d_seedTexture = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seedTexture, sizeof(uint2_32) * seedEntries));
    CUDA_CHECK(cudaMemcpy(
        d_seedTexture, h_seedTexture, sizeof(uint2_32) * seedEntries,
        cudaMemcpyHostToDevice));

    const size_t fakeSeqCount = size_t(sequenceEntryCount) * size_t(BENCH_ITERS);
    QuantizedSequence3* h_fakeSeq = (QuantizedSequence3*)malloc(sizeof(QuantizedSequence3) * fakeSeqCount);
    for (size_t i = 0; i < fakeSeqCount; i++)
    {
        h_fakeSeq[i].data[0] = uint32_t(i * 1103515245u + 12345u);
        h_fakeSeq[i].data[1] = uint32_t(i * 2654435761u + 0x9E3779B9u);
    }

    QuantizedSequence3* d_fakeSeq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fakeSeq, sizeof(QuantizedSequence3) * fakeSeqCount));
    CUDA_CHECK(cudaMemcpy(
        d_fakeSeq, h_fakeSeq, sizeof(QuantizedSequence3) * fakeSeqCount,
        cudaMemcpyHostToDevice));

    for (uint32_t pair = 0; pair < NUM_PACKED_MATRIX_PAIRS; ++pair)
        for (int i = 0; i < 16; ++i)
            h_packedMatrixPairs[pair][i] =
                uint32_t(0xdeadu) | (uint32_t(0xdeadu) << 16);

    CUDA_CHECK(cudaMemcpyToSymbol(
        g_packedMatrixPairs,
        h_packedMatrixPairs,
        sizeof(h_packedMatrixPairs)));

    PushConstants pc;
    pc.pSampleBuffer = 0; 
    pc.sequenceSamplesLog2 = sequenceSamplesLog2;
    pc.depthStaggerMask = 0x0u;
    pc.probAtDepth0 = 0.0f;
    pc.probAtDepthMax = 0.0f;

    constexpr uint32_t warmupDispatches = 100;
    constexpr uint32_t benchDispatches = 500;

    for (uint32_t i = 0; i < warmupDispatches; i++)
    {
        sobol_bench_kernel<<<blocks, threads>>>(d_output, d_seedTexture, d_fakeSeq, pc);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (uint32_t i = 0; i < benchDispatches; i++)
    {
        sobol_bench_kernel<<<blocks, threads>>>(d_output, d_seedTexture, d_fakeSeq, pc);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    const uint64_t samplesPerDispatch =
        uint64_t(totalThreadsPerDispatch) * uint64_t(BENCH_ITERS);

    const uint64_t totalSamples =
        uint64_t(benchDispatches) * samplesPerDispatch;

    const double elapsedNs = double(elapsedMs) * 1.0e6;
    const double psPerSample = elapsedNs * 1.0e3 / double(totalSamples);
    const double gSamplesPerSec = double(totalSamples) / elapsedNs;

    printf("[Benchmark] METHOD 3 | ps/sample: %.3f | GSamples/s: %.3f | ms total: %.3f\n",
           psPerSample, gSamplesPerSec, elapsedMs);

    uint32_t h_check[8];
    CUDA_CHECK(cudaMemcpy(h_check, d_output, sizeof(h_check), cudaMemcpyDeviceToHost));

    printf("First outputs:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("output[%d] = 0x%08x\n", i, h_check[i]);
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_seedTexture));
    CUDA_CHECK(cudaFree(d_fakeSeq));

    free(h_seedTexture);
    free(h_fakeSeq);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
