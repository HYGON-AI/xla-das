// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// TODO(ROCm): Enable and include ROCm Triton passes when ROCm Triton is
// included in build.

#include <cstdlib>
#include <string>

#include "absl/strings/str_cat.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/Passes.h"
#include "third_party/hcu/include/TritonHCU/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;

bool IsHcuRocmTarget(const stream_executor::RocmComputeCapability& rocm_cc) {
  return GetHcuRegisterFileClass(rocm_cc) != HcuRegisterFileClass::kNotHcu;
}

HcuRegisterFileClass GetHcuRegisterFileClass(
    const stream_executor::RocmComputeCapability& rocm_cc) {
  const std::string arch = rocm_cc.gfx_version();
  // Only these two parts are built with a 768-entry VGPR file; every other HCU
  // part has 512. Matches -support-768-vgprs / -support-512-vgprs in the Triton
  // HCU backend.
  if (arch == "gfx936" || arch == "gfx938") {
    return HcuRegisterFileClass::k768Vgpr;
  }
  if (arch == "gfx926" || arch == "gfx928" || arch == "gfx92a" ||
      arch == "gfx946") {
    return HcuRegisterFileClass::k512Vgpr;
  }
  return HcuRegisterFileClass::kNotHcu;
}

namespace {

// Integer tuning knob read from the environment, falling back to the pipeline
// default. These let the HCU warp-specialization knobs be explored without
// adding fields to the autotuning proto (which would invalidate every cached
// autotune result).
int GetEnvInt(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value;
  }
  return std::atoi(value);
}

bool GetEnvBool(const char* name, bool default_value) {
  return GetEnvInt(name, default_value ? 1 : 0) != 0;
}

}  // namespace

// Applies XLA_HCU_* environment overrides to the caller-supplied tuning.
static TritonHcuTuning ApplyHcuTuningEnvOverrides(TritonHcuTuning tuning) {
  tuning.wdra_enabled = GetEnvBool("XLA_HCU_WDRA", tuning.wdra_enabled);
  tuning.wasp_num_load_warps =
      GetEnvInt("XLA_HCU_WASP_LOAD_WARPS", tuning.wasp_num_load_warps);
  tuning.wasp_num_mma_warps =
      GetEnvInt("XLA_HCU_WASP_MMA_WARPS", tuning.wasp_num_mma_warps);
  tuning.global_prefetch =
      GetEnvInt("XLA_HCU_GLOBAL_PREFETCH", tuning.global_prefetch);
  tuning.async_copy_single_buffer = GetEnvInt(
      "XLA_HCU_ASYNC_COPY_SINGLE_BUFFER", tuning.async_copy_single_buffer);
  return tuning;
}

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeTTIR(mlir::OpPassManager* pm,
                     const stream_executor::RocmComputeCapability& rocm_cc) {
  pm->addPass(mlir::createInlinerPass());
  if (!rocm_cc.has_tdm_support()) {
    pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

static bool is_pingpong_schedule_enabled(
    const stream_executor::RocmComputeCapability& rocm_cc,
    bool use_async_copy) {
  return rocm_cc.gfx9_mi300() || (rocm_cc.gfx9_mi350() && use_async_copy);
}

static bool is_in_thread_transpose_enabled(
    const stream_executor::RocmComputeCapability& rocm_cc) {
  return rocm_cc.gfx9_mi300();
}

// Based on make_ttgir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeTTGIR(mlir::OpPassManager* pm,
                      const stream_executor::RocmComputeCapability& rocm_cc,
                      int num_warps, int num_ctas, int num_stages,
                      const TritonHcuTuning& hcu_tuning) {
  const bool is_hcu = IsHcuRocmTarget(rocm_cc);
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {absl::StrCat("hip:", rocm_cc.gfx_version()), num_warps,
       rocm_cc.threads_per_warp(), num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  pm->addPass(mt::gpu::createTritonGPUF32DotTC({false}));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  if (is_hcu) {
    pm->addPass(mlir::createTritonHCUAccelerateMatmul(
        {/*archGenerationName=*/rocm_cc.gfx_version(),
         /*matrixInstructionSize=*/0, /*kPack=*/1,
         /*mmacLayoutForce=*/-1}));
  } else {
    pm->addPass(
        mlir::createTritonAMDGPUAccelerateMatmul({rocm_cc.gfx_version()}));
  }
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  // TODO ROCm Check if we want to compare MI100 and greater
  // Experiment: the Triton HCU backend runs this only when
  // `optimize_epilogue` is set (knob default false).
  if (!is_hcu) {
    pm->addPass(mlir::createTritonAMDGPUOptimizeEpilogue());
  }
  pm->addPass(mt::amdgpu::createTritonAMDGPUOptimizeDotOperands(
      {rocm_cc.gfx_version()}));
  if (is_hcu) {
    pm->addPass(mlir::createTritonHCUMlsEncodingInsertion());
    pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUHoistLayoutConversions());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUSinkLayoutConversions());

  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  bool use_async_copy = false;  // Not enabled by default.
  bool use_block_pingpong =
      is_pingpong_schedule_enabled(rocm_cc, use_async_copy);

  pm->addPass(mlir::createTritonAMDGPUOptimizeDescriptorEncoding());
  if (is_hcu) {
    // `asyncCopySingleBuffer` defaults to on inside the pass; -1 keeps that
    // behaviour unless the caller pins it. Returning `num_stages == 2`
    // unconditionally would silently disable the async LDS copy path for every
    // other stage count, which is what XLA used to do here.
    const int async_copy_single_buffer =
        hcu_tuning.async_copy_single_buffer < 0
            ? (num_stages >= 2 ? 1 : 0)
            : hcu_tuning.async_copy_single_buffer;
    pm->addPass(mlir::createTritonHCUMlsStreamPipeline(
        {/*numStages=*/num_stages,
         /*globalPrefetch=*/hcu_tuning.global_prefetch,
         /*asyncCopySingleBuffer=*/async_copy_single_buffer}));
  }
  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({num_stages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({use_async_copy, use_block_pingpong}));
  if (use_async_copy) {
    pm->addPass(
        mlir::createTritonAMDGPUCoalesceAsyncCopy({rocm_cc.gfx_version()}));
  }
  if (!is_hcu) {
    pm->addPass(mlir::createTritonAMDGPUConvertToTensorOps());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  if (is_hcu) {
    pm->addPass(mlir::createTritonHCUMlsLowering());
  }
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  if (is_in_thread_transpose_enabled(rocm_cc)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUMoveUpPrologueLoads());
  if (use_block_pingpong && num_stages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({num_stages}));
  }

  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUCanonicalizePointers());
  pm->addPass(mlir::createCanonicalizerPass());
  if (is_hcu) {
    // The HCU backend wraps the same pass with two extra options:
    // `emitBufferOpsOffsetAssert` and `bufferCacheSwizzle`. Going through the
    // AMD entry point here silently dropped both. Defaults mirror the HCU
    // backend: `TRITON_BUFFER_CACHE_SWIZZLE` is off by default there.
    pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
        {rocm_cc.gfx_version(), /*allowBufferAtomics=*/true,
         /*analyzeSmallTensorOfst=*/false,
         /*emitBufferOpsOffsetAssert=*/
         GetEnvBool("XLA_HCU_BUFFER_OPS_OFFSET_ASSERT", false),
         /*bufferCacheSwizzle=*/
         GetEnvBool("XLA_HCU_BUFFER_CACHE_SWIZZLE", false)}));
  } else {
    pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
        {rocm_cc.gfx_version(), /*allowBufferAtomics*/ true,
         /*analyzeSmallTensorOfst*/ false}));
  }
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUOptimizeBufferOpPtr());

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUPrepareIfCombining());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  // Unimplemented if instrumentation_mode == "fpsan"
  // pm->addPass(mlir::createTritonAMDGPUFpSanitizer());
  // pm->addPass(mt::gpu::createTritonInstrumentFpSanitizer());
}

// Based on make_llir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeLLIR(mlir::OpPassManager* pm,
                     const stream_executor::RocmComputeCapability& rocm_cc,
                     int num_stages, const TritonHcuTuning& hcu_tuning) {
  // Consumed by the HCU warp-spec lowering once that pass is linked in.
  (void)hcu_tuning;
  pm->addPass(
      mlir::createTritonAMDGPUUpdateAsyncWaitCount({rocm_cc.gfx_version()}));
  pm->addPass(
      mlir::triton::AMD::createConvertWarpPipelinePass(rocm_cc.gfx_version()));
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());
  pm->addPass(mt::createAllocateAMDGPUSharedMemoryPass(rocm_cc.gfx_version()));
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  if (IsHcuRocmTarget(rocm_cc)) {
    pm->addPass(
        mt::createConvertTritonHCUToLLVMPass(rocm_cc.gfx_version(), true));
  } else {
    pm->addPass(
        mt::createConvertTritonAMDGPUToLLVMPass(rocm_cc.gfx_version(), true));
  }
  if (IsHcuRocmTarget(rocm_cc)) {
    // HCU partitions are lowered by the Triton HCU backend. Must be driven by
    // the same knobs as the TTGIR warp-specialization pass above; a mismatch
    // miscompiles.
  } else {
    pm->addPass(
        mlir::triton::AMD::createTritonAMDGPUConvertWarpSpecializeToLLVMPass(
            rocm_cc.gfx_version()));
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createConvertBuiltinFuncToLLVMPass(rocm_cc.gfx_version(),
                                                     /*ftz=*/true));

  // Add XLA custom pass to implement extern_elementwise functions
  // This must run after MLIR->LLVM conversion but before final optimizations
  pm->addPass(mlir::triton::xla::CreateTritonXLAImplementExternElementWisePass(
      mlir::triton::xla::TargetBackend::ROCM));
}

void CreateTritonRocmPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::RocmComputeCapability& rocm_cc, int num_warps,
    int num_ctas, int num_stages, const TritonHcuTuning& hcu_tuning) {
  const TritonHcuTuning tuning = ApplyHcuTuningEnvOverrides(hcu_tuning);
  MakeTTIR(pm, rocm_cc);
  MakeTTGIR(pm, rocm_cc, num_warps, num_ctas, num_stages, tuning);
  MakeLLIR(pm, rocm_cc, num_stages, tuning);
}

}  // namespace gpu
}  // namespace xla
