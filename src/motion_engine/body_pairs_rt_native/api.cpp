#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "params.h"
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <cstdint>
#include <string>
static void check(OptixResult r) { if(r!=OPTIX_SUCCESS) throw std::runtime_error(optixGetErrorName(r)); }
static void check(cudaError_t r) { if(r!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(r)); }
static CUdeviceptr allocate(size_t bytes,const void* source=nullptr) {
    void* p=nullptr; check(cudaMalloc(&p,bytes));
    if(source) check(cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(p);
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct State {
    OptixDeviceContext context{}; OptixModule module{}; OptixProgramGroup groups[3]{};
    OptixPipeline pipeline{}; OptixShaderBindingTable sbt{};
    CUdeviceptr sbt_data{},scratch{},storage{},params_data{},boxes{};
    OptixBuildInput build{}; OptixAccelBuildOptions accel{}; OptixAccelBufferSizes sizes{};
    unsigned flags=OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,n{},capacity{};
    PairParams parameters{}; std::thread::id owner=std::this_thread::get_id();
    void initialize(const std::string& ptx) {
        check(cudaFree(nullptr)); check(optixInit());
        OptixDeviceContextOptions options{}; check(optixDeviceContextCreate(nullptr,&options,&context));
        OptixPipelineCompileOptions compile{};
        compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile.numPayloadValues = 1;
        compile.numAttributeValues = 0;
        compile.pipelineLaunchParamsVariableName = "params";
        compile.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        OptixModuleCompileOptions module_options{};

        check(optixModuleCreate(context, &module_options, &compile,
                               ptx.data(), ptx.size(), nullptr, nullptr, &module));
        OptixProgramGroupDesc descriptions[3]{};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen = {module, "__raygen__pairs"};
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss = {module, "__miss__pairs"};
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleAH = module;
        descriptions[2].hitgroup.entryFunctionNameAH = "__anyhit__pairs";

        descriptions[2].hitgroup.moduleIS = module;
        descriptions[2].hitgroup.entryFunctionNameIS = "__intersection__pairs";
        OptixProgramGroupOptions group_options{};
        check(optixProgramGroupCreate(context, descriptions, 3, &group_options, nullptr, nullptr, groups));
        OptixPipelineLinkOptions link{};
        link.maxTraceDepth = 1;

        check(optixPipelineCreate(context, &compile, &link, groups, 3, nullptr, nullptr, &pipeline));
        OptixStackSizes stack{};
        for (auto group : groups) check(optixUtilAccumulateStackSizes(group, &stack, pipeline));
        unsigned traversal{}, state{}, continuation{};
        check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
        check(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1));
        Record records[3]{};
        for (int i=0; i<3; ++i) check(optixSbtRecordPackHeader(groups[i], &records[i]));
        sbt_data = allocate(sizeof(records), records);

        sbt.raygenRecord = sbt_data;
        sbt.missRecordBase = sbt_data + sizeof(Record);
        sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = sbt_data + 2*sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = 1;
        params_data=allocate(sizeof(PairParams));
        build.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build.customPrimitiveArray.aabbBuffers=&boxes;
        build.customPrimitiveArray.numPrimitives=n;
        build.customPrimitiveArray.flags=&flags;
        build.customPrimitiveArray.numSbtRecords=1;
        accel.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        accel.operation=OPTIX_BUILD_OPERATION_BUILD;
        check(optixAccelComputeMemoryUsage(context,&accel,&build,1,&sizes));
        scratch=allocate(sizes.tempSizeInBytes); storage=allocate(sizes.outputSizeInBytes);
    }
    int release() noexcept {
        int rc=0;
        for(auto p:{sbt_data,scratch,storage,params_data})
            if(p && cudaFree(reinterpret_cast<void*>(p))!=cudaSuccess) rc=2;
        if(pipeline && optixPipelineDestroy(pipeline)!=OPTIX_SUCCESS) rc=2;
        for(auto g:groups) if(g && optixProgramGroupDestroy(g)!=OPTIX_SUCCESS) rc=2;
        if(module && optixModuleDestroy(module)!=OPTIX_SUCCESS) rc=2;
        if(context && optixDeviceContextDestroy(context)!=OPTIX_SUCCESS) rc=2;
        return rc;
    }
};
extern "C" int pairs_create(const char* path,unsigned n,unsigned capacity,void** out) {
    if(!path || !out || *out || !n || n>100000 || !capacity || capacity>256) return 1;
    State* s=nullptr;
    try {
        std::ifstream f(path); std::string ptx{std::istreambuf_iterator<char>(f),{}};
        if(ptx.empty()) return 1;
        s=new State; s->n=n; s->capacity=capacity; s->initialize(ptx); *out=s; return 0;
    } catch(...) { if(s) {s->release();delete s;} return 2; }
}
extern "C" int pairs_run(void* handle,void* stream,void* boxes,const void* lower,const void* upper,
                          const void* centers,void* slots,void* counts,void* overflow) {
    if(!handle || !boxes || !lower || !upper || !centers || !slots || !counts || !overflow) return 1;
    State& s=*static_cast<State*>(handle);
    if(s.owner!=std::this_thread::get_id()) return 3;
    cudaStream_t st=reinterpret_cast<cudaStream_t>(stream);
    try {
        s.boxes=reinterpret_cast<CUdeviceptr>(boxes);
        OptixTraversableHandle gas{};
        check(optixAccelBuild(s.context,st,&s.accel,&s.build,1,s.scratch,s.sizes.tempSizeInBytes,
                             s.storage,s.sizes.outputSizeInBytes,&gas,nullptr,0));
        s.parameters={gas,static_cast<const double3*>(lower),static_cast<const double3*>(upper),
            static_cast<const float3*>(centers),static_cast<int*>(slots),static_cast<int*>(counts),
            static_cast<int*>(overflow),s.capacity};
        check(cudaMemcpyAsync(reinterpret_cast<void*>(s.params_data),&s.parameters,sizeof(PairParams),cudaMemcpyHostToDevice,st));
        check(optixLaunch(s.pipeline,st,s.params_data,sizeof(PairParams),&s.sbt,s.n,1,1));
        check(cudaStreamSynchronize(st)); return 0;
    } catch(...) { cudaStreamSynchronize(st); return 2; }
}
extern "C" int pairs_destroy(void** handle) {
    if(!handle) return 1; if(!*handle) return 0;
    State* s=static_cast<State*>(*handle);
    if(s->owner!=std::this_thread::get_id()) return 3;
    int rc=s->release(); delete s; *handle=nullptr; return rc;
}
