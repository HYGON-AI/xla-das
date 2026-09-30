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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_

#include "mlir/Pass/PassManager.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// HCU-specific tuning for the Triton pipeline. Ignored on other targets.
//
// These mirror the equivalent knobs of
// @triton//:third_party/hcu/backend/compiler.py. `enable_warp_specialization`,
// `wdra_enabled` and the two warp counts must stay consistent between the TTGIR
// warp-specialization pass and the LLIR warp-specialize lowering: the lowering
// reconstructs the partition layout that the pass produced, so a mismatch is a
// miscompile rather than a tuning regression.
struct TritonHcuTuning {
  // Insert the HCU automatic warp-specialization pass (WASP).
  bool enable_warp_specialization = false;
  // Give each partition its own VGPR budget (WDRA).
  bool wdra_enabled = false;
  // Number of warps in the load / MMA partitions.
  int wasp_num_load_warps = 4;
  int wasp_num_mma_warps = 8;
  // Additional global-prefetch stages for the MLS stream pipeline.
  int global_prefetch = 0;
  // Use the async single-buffer LDS copy path. -1 derives it from
  // `num_stages >= 2`, matching the pass default of "on".
  int async_copy_single_buffer = -1;
};

// True for ROCm targets that are Hygon HCU parts.
bool IsHcuRocmTarget(const stream_executor::RocmComputeCapability& rocm_cc);

// Register-file size class of an HCU part. This, rather than the raw arch
// string, is what selects default tile shapes and with them the default Triton
// config table.
enum class HcuRegisterFileClass {
  kNotHcu,
  k512Vgpr,
  k768Vgpr,
};

HcuRegisterFileClass GetHcuRegisterFileClass(
    const stream_executor::RocmComputeCapability& rocm_cc);

// Adds TritonXLA passes to the pipeline.
void CreateTritonXlaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, bool rewrite_int4,
    bool allow_tma, int num_stages, bool warp_specialization_allowed,
    bool enable_pdl);

// Creates a Triton compilation pipeline.
void CreateTritonPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, int num_warps,
    int num_ctas, int num_stages,
    const TritonHcuTuning& hcu_tuning = TritonHcuTuning());

// Returns the default PTX version for a given CUDA compute capability.
int GetDefaultPtxVersion(const stream_executor::CudaComputeCapability& cuda_cc);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_
