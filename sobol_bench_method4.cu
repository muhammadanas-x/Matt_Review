#include "sobol_bench_common.cuh"

constexpr uint32_t PACKED_MATRIX_PAIRS_PER_DEPTH =
    SLOTS_PER_DEPTH / 2u;

constexpr uint32_t NUM_PACKED_MATRIX_PAIRS =
    DEPTH * PACKED_MATRIX_PAIRS_PER_DEPTH;

constexpr uint32_t MATRIX_COLUMN_COUNT = 16u;

constexpr uint32_t MATRIX_WORD_COUNT =
    NUM_PACKED_MATRIX_PAIRS *
    MATRIX_COLUMN_COUNT;

static_assert(
    (SLOTS_PER_DEPTH % 2u) == 0u,
    "SLOTS_PER_DEPTH must be even.");

static_assert(
    PACKED_MATRIX_PAIRS_PER_DEPTH == 3u,
    "This optimized path expects six outputs packed into three uint32 values.");

static_assert(
    DEPTH > 1u,
    "DEPTH must be greater than one.");

static_assert(
    (DEPTH & (DEPTH - 1u)) == 0u,
    "DEPTH must be a power of two.");

__device__ __align__(16)
uint32_t g_packedMatrixPairs
    [NUM_PACKED_MATRIX_PAIRS]
    [MATRIX_COLUMN_COUNT];

alignas(16)
static uint32_t h_packedMatrixPairs
    [NUM_PACKED_MATRIX_PAIRS]
    [MATRIX_COLUMN_COUNT];


__global__ void sobol_bench_kernel(
    uint32_t* __restrict__ output,
    const uint2_32* __restrict__ seedTexture,
    const QuantizedSequence3* __restrict__ pSampleBuffer,
    PushConstants pc)
{
    (void)seedTexture;
    (void)pSampleBuffer;


    __shared__ __align__(16)
    uint32_t sharedPackedMatrixPairs
        [NUM_PACKED_MATRIX_PAIRS]
        [MATRIX_COLUMN_COUNT];

    uint32_t* const sharedWords =
        &sharedPackedMatrixPairs[0][0];

    const uint32_t* const globalWords =
        &g_packedMatrixPairs[0][0];

   
    for (uint32_t wordIndex = threadIdx.x;
         wordIndex < MATRIX_WORD_COUNT;
         wordIndex += blockDim.x)
    {
        sharedWords[wordIndex] =
            globalWords[wordIndex];
    }

    __syncthreads();

    const uint32_t invID =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const uint16_t rank =
        uint16_t(
            bit_reverse_32(invID) >>
            16u);

    
    const uint16_t laneOffset =
        uint16_t(
            invID &
            pc.depthStaggerMask) &
        uint16_t(
            DEPTH - 1u);

    constexpr float THRESHOLD_SCALE =
        65535.0f;

    const float threshold =
        pc.probAtDepth0 *
        THRESHOLD_SCALE;

    const float thresholdDelta =
        (pc.probAtDepthMax -
         pc.probAtDepth0) *
        THRESHOLD_SCALE /
        float(DEPTH - 1u);

    const uint32_t warpMask =
        __activemask();

    uint16_t seed =
        uint16_t(0xdeadu);

    for (uint32_t s = 0u;
         s < BENCH_ITERS;
         ++s)
    {
        const uint32_t sampleBits =
            uint32_t(
                uint16_t(s) ^
                rank);

        float perSampleThreshold =
            threshold;

        bool laneActive = true;

        for (uint32_t d = 0u;
             d < DEPTH;
             ++d)
        {
          
            if (!__any_sync(
                    warpMask,
                    laneActive))
            {
                break;
            }

          
            const uint32_t dIdx =
                (d +
                 uint32_t(laneOffset)) &
                (DEPTH - 1u);

            const uint32_t pairBase =
                dIdx *
                PACKED_MATRIX_PAIRS_PER_DEPTH;

            uint32_t packedPair0 = 0u;
            uint32_t packedPair1 = 0u;
            uint32_t packedPair2 = 0u;

        
#pragma unroll
            for (uint32_t column = 0u;
                 column < MATRIX_COLUMN_COUNT;
                 ++column)
            {
                const uint32_t keep =
                    0u -
                    ((sampleBits >> column) &
                     1u);

                packedPair0 ^=
                    sharedPackedMatrixPairs
                        [pairBase + 0u]
                        [column] &
                    keep;

                packedPair1 ^=
                    sharedPackedMatrixPairs
                        [pairBase + 1u]
                        [column] &
                    keep;

                packedPair2 ^=
                    sharedPackedMatrixPairs
                        [pairBase + 2u]
                        [column] &
                    keep;
            }

            /*
                Pair mapping:

                    pair 0 low  -> triples[0][0]
                    pair 0 high -> triples[0][1]

                    pair 1 low  -> triples[0][2]
                    pair 1 high -> triples[1][0]

                    pair 2 low  -> triples[1][1]
                    pair 2 high -> triples[1][2]
            */
            const uint16_t value00 =
                uint16_t(
                    packedPair0);

            const uint16_t value01 =
                uint16_t(
                    packedPair0 >>
                    16u);

            const uint16_t value02 =
                uint16_t(
                    packedPair1);

            const uint16_t value10 =
                uint16_t(
                    packedPair1 >>
                    16u);

            const uint16_t value11 =
                uint16_t(
                    packedPair2);

            const uint16_t value12 =
                uint16_t(
                    packedPair2 >>
                    16u);

            if (laneActive)
            {
                seed ^= value00;
                seed += value01;
                seed ^= value02;

                seed += value10;
                seed ^= value11;
                seed += value12;

                const uint16_t integerThreshold =
                    uint16_t(
                        perSampleThreshold);

                if (value02 < integerThreshold)
                {
                    laneActive = false;
                }
            }

            perSampleThreshold +=
                thresholdDelta;
        }
    }

    output[invID] =
        uint32_t(seed);
}


static void prepare_method_data()
{
    constexpr uint32_t PACKED_TEST_VALUE =
        uint32_t(0xdeadu) |
        (uint32_t(0xdeadu) << 16u);

    for (uint32_t pair = 0u;
         pair < NUM_PACKED_MATRIX_PAIRS;
         ++pair)
    {
        for (uint32_t column = 0u;
             column < MATRIX_COLUMN_COUNT;
             ++column)
        {
            h_packedMatrixPairs
                [pair]
                [column] =
                PACKED_TEST_VALUE;
        }
    }

    CUDA_CHECK(
        cudaMemcpyToSymbol(
            g_packedMatrixPairs,
            h_packedMatrixPairs,
            sizeof(h_packedMatrixPairs),
            0u,
            cudaMemcpyHostToDevice));
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
        "4",
        prepare_method_data,
        launch_method_kernel);
}