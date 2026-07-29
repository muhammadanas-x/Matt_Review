#include "sobol_bench_common.cuh"

__global__ void sobol_bench_kernel(
    uint32_t* output,
    const uint2_32* seedTexture,
    const QuantizedSequence3* pSampleBuffer, // BDA-equivalent flat pointer
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

    const uint32_t SeedDim = 512u;
    const uint32_t SeedDimMask = SeedDim - 1u;

    const uint32_t invX = invID & SeedDimMask;
    const uint32_t invY = (invID >> 9) & SeedDimMask;

    for (uint16_t s = 0; s < BENCH_ITERS; s++)
    {
        const uint32_t coordX = (invX + uint32_t(s)) & SeedDimMask;
        const uint32_t coordY = (invY + uint32_t(s)) & SeedDimMask;
        Xoroshiro64Star rng = Xoroshiro64Star::construct(seedTexture[coordY * SeedDim + coordX]);

        float perSampleThreshold = threshold;

        for (uint16_t d = 0; d < DEPTH; d++)
        {
            const uint16_t dIdx = uint16_t((d + laneOffset) & uint16_t(DEPTH - 1u));

            uint16_t triples[TRIPLETS][COMPONENTS];

#pragma unroll
            for (uint16_t t = 0; t < TRIPLETS; t++)
            {
                const uint32_t baseDim = uint32_t(dIdx) * uint32_t(TRIPLETS) + uint32_t(t);
                const uint32_t address = uint32_t(s) | (baseDim << pc.sequenceSamplesLog2);

                QuantizedSequence3 tmpSeq = pSampleBuffer[address];

              
                QuantizedSequence3 scrambleKey;
                scrambleKey.data[0] = rng();
                scrambleKey.data[1] = rng();
                tmpSeq.data[0] ^= scrambleKey.data[0];
                tmpSeq.data[1] ^= scrambleKey.data[1];

                triples[t][0] = tmpSeq.get(0);
                triples[t][1] = tmpSeq.get(1);
                triples[t][2] = tmpSeq.get(2);
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
    // Method 0 uses only the shared seed texture and sequence buffer.
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
        "0",
        prepare_method_data,
        launch_method_kernel);
}
