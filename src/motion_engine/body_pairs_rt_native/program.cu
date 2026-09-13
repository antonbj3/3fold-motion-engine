#include "params.h"
#include <optix_device.h>
extern "C" { __constant__ PairParams params; }
extern "C" __global__ void __raygen__pairs() {
    unsigned a=optixGetLaunchIndex().x;
    unsigned count=params.lower[a].z<0.0 ? 1 : 0;
    if(count) params.slots[size_t(a)*params.capacity]=-1;
    float3 c=params.centers[a]; c.z-=1.f;
    optixTrace(params.gas,c,make_float3(0,0,1),0.f,2.f,0.f,255,
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,0,count);
    params.counts[a]=int(count);
    if(count>params.capacity) atomicExch(params.overflow,1);
}
extern "C" __global__ void __miss__pairs() {}
extern "C" __global__ void __intersection__pairs() {
    unsigned a=optixGetLaunchIndex().x,b=optixGetPrimitiveIndex();
    if(b<=a) return;
    double3 al=params.lower[a],ah=params.upper[a],bl=params.lower[b],bh=params.upper[b];
    if(al.x<=bh.x && bl.x<=ah.x && al.y<=bh.y && bl.y<=ah.y && al.z<=bh.z && bl.z<=ah.z)
        optixReportIntersection(1.f,0);
}
extern "C" __global__ void __anyhit__pairs() {
    unsigned n=optixGetPayload_0(),a=optixGetLaunchIndex().x;
    if(n<params.capacity) params.slots[size_t(a)*params.capacity+n]=int(optixGetPrimitiveIndex());
    optixSetPayload_0(n+1); optixIgnoreIntersection();
}
