#include "sobol_bench_common.cuh"

constexpr uint32_t PACKED_MATRIX_PAIRS_PER_DEPTH =
    SLOTS_PER_DEPTH / 2u;

constexpr uint32_t NUM_PACKED_MATRIX_PAIRS =
    DEPTH * PACKED_MATRIX_PAIRS_PER_DEPTH;

__device__ __constant__
uint32_t g_packedMatrixPairs
    [NUM_PACKED_MATRIX_PAIRS]
    [16];

static uint32_t h_packedMatrixPairs
    [NUM_PACKED_MATRIX_PAIRS]
    [16];

__device__ __forceinline__
uint16_t col_major_matrix_apply(
    const uint16_t* cols,
    uint16_t sampleIx)
{
    uint16_t val = 0u;

#pragma unroll
    for (uint16_t i = 0u;
         i < 16u;
         ++i)
    {
        const uint16_t keep =
            uint16_t(
                0u -
                uint16_t(
                    (sampleIx >> i) & 1u));

        val ^=
            uint16_t(
                cols[i] & keep);
    }

    return val;
}

__device__ __forceinline__
uint32_t warp_col_major_matrix_apply_2d(
    const uint32_t* packedCols,
    uint16_t sampleIx)
{
    constexpr uint32_t FULL_MASK =
        0xffffffffu;

    const uint32_t laneID =
        threadIdx.x & 31u;

    const uint32_t myPackedCols =
        laneID < 16u
            ? packedCols[laneID]
            : 0u;

    uint32_t packedVal = 0u;

#pragma unroll
    for (uint16_t i = 0u;
         i < 16u;
         ++i)
    {
        const uint32_t packedCol =
            __shfl_sync(
                FULL_MASK,
                myPackedCols,
                int(i));

        const uint32_t keep =
            0u -
            uint32_t(
                (sampleIx >> i) & 1u);

        packedVal ^=
            packedCol & keep;
    }

    return packedVal;
}


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

static void prepare_method_data()
{
    for (uint32_t pair = 0u;
         pair < NUM_PACKED_MATRIX_PAIRS;
         ++pair)
    {
        for (uint32_t i = 0u;
             i < 16u;
             ++i)
        {
            h_packedMatrixPairs[pair][i] =
                uint32_t(0xdeadu) |
                (uint32_t(0xdeadu) << 16u);
        }
    }

    CUDA_CHECK(
        cudaMemcpyToSymbol(
            g_packedMatrixPairs,
            h_packedMatrixPairs,
            sizeof(h_packedMatrixPairs)));
}

static void launch_method_kernel(
    uint32_t* output,
    const uint2_32* seedTexture,
    const QuantizedSequence3* sampleBuffer,
    PushConstants pc,
    uint32_t blocks,
    uint32_t threads)
{
    sobol_bench_kernel<<<blocks, threads>>>(
        output,
        seedTexture,
        sampleBuffer,
        pc);
}

int main()
{
    return run_sobol_benchmark(
        "3",
        prepare_method_data,
        launch_method_kernel);
}
