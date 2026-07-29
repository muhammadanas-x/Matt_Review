#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Shared benchmark configuration.
// All methods include this file, so the launch/workload settings cannot drift.
constexpr uint32_t BENCH_ITERS = 256;
constexpr uint32_t WORKGROUP_SIZE = 256;
constexpr uint32_t DEPTH = 8;

constexpr uint32_t TRIPLETS = 2;
constexpr uint32_t COMPONENTS = 3;
constexpr uint32_t TEST_BATCH_COUNT = 256;
constexpr uint32_t SLOTS_PER_DEPTH = TRIPLETS * COMPONENTS;

constexpr uint32_t SEED_DIM = 512;
constexpr uint32_t WARMUP_DISPATCHES = 100;
constexpr uint32_t BENCH_DISPATCHES = 500;

#define CUDA_CHECK(x)                                                     \
    do                                                                    \
    {                                                                     \
        const cudaError_t err__ = (x);                                    \
        if (err__ != cudaSuccess)                                         \
        {                                                                 \
            printf(                                                       \
                "CUDA error %s:%d: %s\n",                                 \
                __FILE__,                                                 \
                __LINE__,                                                 \
                cudaGetErrorString(err__));                               \
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
    uint32_t x;
    uint32_t y;
};

__device__ __forceinline__
uint32_t bit_reverse_32(uint32_t x)
{
    return __brev(x);
}

__device__ __forceinline__
uint32_t parity32(uint32_t x)
{
    return __popc(x) & 1u;
}

struct QuantizedSequence3
{
    uint32_t data[2];

    static constexpr uint32_t BitsPerComponent = 21u;
    static constexpr uint32_t DiscardBits = 11u;

    __device__ __host__ static __forceinline__
    uint32_t bitfieldExtractU32(
        uint32_t x,
        uint32_t offset,
        uint32_t bits)
    {
        return
            (x >> offset) &
            ((1u << bits) - 1u);
    }

    __device__ __host__ static __forceinline__
    uint32_t bitfieldInsertU32(
        uint32_t base,
        uint32_t insert,
        uint32_t offset,
        uint32_t bits)
    {
        const uint32_t fieldMask =
            ((1u << bits) - 1u) << offset;

        return
            (base & ~fieldMask) |
            ((insert << offset) & fieldMask);
    }

    __device__ __host__ __forceinline__
    uint32_t get32(int idx) const
    {
        if (idx == 0)
        {
            return bitfieldExtractU32(
                data[0],
                0u,
                BitsPerComponent);
        }

        if (idx == 1)
        {
            uint32_t y =
                bitfieldExtractU32(
                    data[0],
                    BitsPerComponent,
                    DiscardBits);

            y |=
                bitfieldExtractU32(
                    data[1],
                    0u,
                    BitsPerComponent - DiscardBits)
                << DiscardBits;

            return y;
        }

        return bitfieldExtractU32(
            data[1],
            BitsPerComponent - DiscardBits,
            BitsPerComponent);
    }

    __device__ __host__ __forceinline__
    uint16_t get(int idx) const
    {
        return uint16_t(get32(idx));
    }

    __device__ __host__ __forceinline__
    void set(
        int idx,
        uint32_t value)
    {
        const uint32_t truncVal =
            value >> DiscardBits;

        if (idx == 0)
        {
            data[0] =
                bitfieldInsertU32(
                    data[0],
                    truncVal,
                    0u,
                    BitsPerComponent);

            return;
        }

        if (idx == 1)
        {
            data[0] =
                bitfieldInsertU32(
                    data[0],
                    truncVal,
                    BitsPerComponent,
                    DiscardBits);

            data[1] =
                bitfieldInsertU32(
                    data[1],
                    truncVal >> DiscardBits,
                    0u,
                    BitsPerComponent - DiscardBits);

            return;
        }

        data[1] =
            bitfieldInsertU32(
                data[1],
                truncVal,
                BitsPerComponent - DiscardBits,
                BitsPerComponent);
    }
};

struct Xoroshiro64Star
{
    uint32_t s0;
    uint32_t s1;

    __device__ __forceinline__
    static Xoroshiro64Star construct(uint2_32 seed)
    {
        Xoroshiro64Star rng;
        rng.s0 = seed.x;
        rng.s1 = seed.y;
        return rng;
    }

    __device__ __forceinline__
    static uint32_t rotl(
        uint32_t x,
        int k)
    {
        return
            (x << k) |
            (x >> (32 - k));
    }

    __device__ __forceinline__
    uint32_t operator()()
    {
        const uint32_t result =
            s0 * 0x9E3779BBu;

        s1 ^= s0;
        s0 = rotl(s0, 26) ^ s1 ^ (s1 << 9);
        s1 = rotl(s1, 13);

        return result;
    }
};

inline uint32_t ceil_log2_u32(uint32_t x)
{
    uint32_t result = 0u;

    --x;

    while (x > 0u)
    {
        x >>= 1u;
        ++result;
    }

    return result;
}

/*
    Shared host-side benchmark runner.

    prepareMethodData:
        Initializes and uploads method-specific matrix data.

    launchKernel:
        Launches the method's kernel using the supplied common buffers,
        push constants, grid size, and block size.
*/
template <typename PrepareMethodData, typename LaunchKernel>
int run_sobol_benchmark(
    const char* methodLabel,
    PrepareMethodData prepareMethodData,
    LaunchKernel launchKernel)
{
    printf("CUDA Sobol exact-port benchmark\n");
    printf(
        "METHOD=%s, BENCH_ITERS=%u, WORKGROUP_SIZE=%u, DEPTH=%u\n",
        methodLabel,
        BENCH_ITERS,
        WORKGROUP_SIZE,
        DEPTH);

    static_assert(
        (BENCH_ITERS & (BENCH_ITERS - 1u)) == 0u,
        "BENCH_ITERS must be a power of two");

    const uint32_t totalThreadsPerDispatch =
        TEST_BATCH_COUNT * WORKGROUP_SIZE;

    const uint32_t blocks =
        TEST_BATCH_COUNT;

    const uint32_t threads =
        WORKGROUP_SIZE;

    const uint32_t sequenceEntryCount =
        DEPTH * TRIPLETS;

    const uint32_t sequenceSamplesLog2 =
        ceil_log2_u32(BENCH_ITERS);

    const size_t outputBytes =
        sizeof(uint32_t) *
        size_t(totalThreadsPerDispatch);

    uint32_t* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            outputBytes));

    const size_t seedEntries =
        size_t(SEED_DIM) *
        size_t(SEED_DIM);

    uint2_32* h_seedTexture =
        static_cast<uint2_32*>(
            malloc(
                sizeof(uint2_32) *
                seedEntries));

    if (h_seedTexture == nullptr)
    {
        printf("Failed to allocate the host seed texture.\n");
        CUDA_CHECK(cudaFree(d_output));
        return 1;
    }

    for (size_t i = 0u;
         i < seedEntries;
         ++i)
    {
        h_seedTexture[i].x =
            uint32_t(
                0xbadc0ffeu +
                i * 1664525u);

        h_seedTexture[i].y =
            uint32_t(
                0x12345678u +
                i * 1013904223u);
    }

    uint2_32* d_seedTexture = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_seedTexture,
            sizeof(uint2_32) *
            seedEntries));

    CUDA_CHECK(
        cudaMemcpy(
            d_seedTexture,
            h_seedTexture,
            sizeof(uint2_32) *
                seedEntries,
            cudaMemcpyHostToDevice));

    const size_t fakeSeqCount =
        size_t(sequenceEntryCount) *
        size_t(BENCH_ITERS);

    QuantizedSequence3* h_fakeSeq =
        static_cast<QuantizedSequence3*>(
            malloc(
                sizeof(QuantizedSequence3) *
                fakeSeqCount));

    if (h_fakeSeq == nullptr)
    {
        printf("Failed to allocate the host sequence buffer.\n");

        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_seedTexture));
        free(h_seedTexture);

        return 1;
    }

    for (size_t i = 0u;
         i < fakeSeqCount;
         ++i)
    {
        h_fakeSeq[i].data[0] =
            uint32_t(
                i * 1103515245u +
                12345u);

        h_fakeSeq[i].data[1] =
            uint32_t(
                i * 2654435761u +
                0x9E3779B9u);
    }

    QuantizedSequence3* d_fakeSeq = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_fakeSeq,
            sizeof(QuantizedSequence3) *
                fakeSeqCount));

    CUDA_CHECK(
        cudaMemcpy(
            d_fakeSeq,
            h_fakeSeq,
            sizeof(QuantizedSequence3) *
                fakeSeqCount,
            cudaMemcpyHostToDevice));

    prepareMethodData();

    PushConstants pc{};

    pc.pSampleBuffer = 0u;
    pc.sequenceSamplesLog2 =
        sequenceSamplesLog2;

    pc.depthStaggerMask = 0x0u;
    pc.probAtDepth0 = 0.0f;
    pc.probAtDepthMax = 0.0f;

    for (uint32_t i = 0u;
         i < WARMUP_DISPATCHES;
         ++i)
    {
        launchKernel(
            d_output,
            d_seedTexture,
            d_fakeSeq,
            pc,
            blocks,
            threads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (uint32_t i = 0u;
         i < BENCH_DISPATCHES;
         ++i)
    {
        launchKernel(
            d_output,
            d_seedTexture,
            d_fakeSeq,
            pc,
            blocks,
            threads);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop));

    const uint64_t samplesPerDispatch =
        uint64_t(totalThreadsPerDispatch) *
        uint64_t(BENCH_ITERS);

    const uint64_t totalSamples =
        uint64_t(BENCH_DISPATCHES) *
        samplesPerDispatch;

    const double elapsedNs =
        double(elapsedMs) *
        1.0e6;

    const double psPerSample =
        elapsedNs *
        1.0e3 /
        double(totalSamples);

    const double gSamplesPerSec =
        double(totalSamples) /
        elapsedNs;

    printf(
        "[Benchmark] METHOD %s | "
        "ps/sample: %.3f | "
        "GSamples/s: %.3f | "
        "ms total: %.3f\n",
        methodLabel,
        psPerSample,
        gSamplesPerSec,
        elapsedMs);

    uint32_t h_check[8];

    CUDA_CHECK(
        cudaMemcpy(
            h_check,
            d_output,
            sizeof(h_check),
            cudaMemcpyDeviceToHost));

    printf("First outputs:\n");

    for (int i = 0;
         i < 8;
         ++i)
    {
        printf(
            "output[%d] = 0x%08x\n",
            i,
            h_check[i]);
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
