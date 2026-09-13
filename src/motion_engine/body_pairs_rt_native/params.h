#pragma once
#include <optix.h>
#include <cuda_runtime.h>
struct PairParams {
    OptixTraversableHandle gas;
    const double3 *lower, *upper;
    const float3 *centers;
    int *slots, *counts, *overflow;
    unsigned capacity;
};
