// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla {
namespace gpu {

using ::testing::Contains;
using ::testing::Not;

namespace {

std::vector<std::string> GetRocmPassNames(absl::string_view gfx,
                                          const TritonHcuTuning& tuning) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);
  stream_executor::RocmComputeCapability rocm_cc{std::string(gfx)};
  CreateTritonPipeline(&pm,
                       stream_executor::GpuComputeCapability(std::move(rocm_cc)),
                       /*num_warps=*/4, /*num_ctas=*/1, /*num_stages=*/2,
                       tuning);
  std::vector<std::string> names;
  for (const mlir::Pass& pass : pm.getPasses()) {
    names.push_back(pass.getName().str());
  }
  return names;
}

int IndexOf(absl::Span<const std::string> names, absl::string_view needle) {
  for (int i = 0; i < names.size(); ++i) {
    if (names[i] == needle) {
      return i;
    }
  }
  return -1;
}

int CountOf(absl::Span<const std::string> names, absl::string_view needle) {
  int n = 0;
  for (const std::string& name : names) {
    n += (name == needle) ? 1 : 0;
  }
  return n;
}

}  // namespace

// Lock the HCU arch-classification table. These parts differ in register file
// size, which is what selects the default Triton config bucket.
TEST(CompilationPipelineTest, HcuRegisterFileClass) {
  EXPECT_EQ(GetHcuRegisterFileClass(
                stream_executor::RocmComputeCapability("gfx936")),
            HcuRegisterFileClass::k768Vgpr);
  EXPECT_EQ(GetHcuRegisterFileClass(
                stream_executor::RocmComputeCapability("gfx938")),
            HcuRegisterFileClass::k768Vgpr);
  for (const char* arch : {"gfx926", "gfx928", "gfx92a", "gfx946"}) {
    EXPECT_EQ(GetHcuRegisterFileClass(
                  stream_executor::RocmComputeCapability(arch)),
              HcuRegisterFileClass::k512Vgpr)
        << arch;
    EXPECT_TRUE(IsHcuRocmTarget(stream_executor::RocmComputeCapability(arch)))
        << arch;
  }
  // Plain ROCm targets must not be treated as HCU.
  for (const char* arch : {"gfx942", "gfx950", "gfx90a"}) {
    EXPECT_EQ(GetHcuRegisterFileClass(
                  stream_executor::RocmComputeCapability(arch)),
              HcuRegisterFileClass::kNotHcu)
        << arch;
    EXPECT_FALSE(IsHcuRocmTarget(stream_executor::RocmComputeCapability(arch)))
        << arch;
  }
}

// Warp specialization must be requested explicitly; without the flag the HCU
// pipeline must look exactly like it did before, so that the default path is
// unaffected.
TEST(CompilationPipelineTest, HcuWarpSpecializationOffByDefault) {
  std::vector<std::string> names =
      GetRocmPassNames("gfx92a", TritonHcuTuning());
  EXPECT_THAT(names, Not(Contains("tritongpu-automatic-warp-specialization-hcu")));
  EXPECT_THAT(names, Not(Contains("triton-hcu-convert-warp-specialize-to-llvm")));
}

// The HCU warp-specialization pass only rewrites loops tagged with
// `tt.warp_specialize`, so the gate and the pass must be driven by the same
// flag.
TEST(CompilationPipelineTest, HcuWarpSpecializationWiredWhenEnabled) {
  TritonHcuTuning tuning;
  tuning.enable_warp_specialization = true;
  std::vector<std::string> names = GetRocmPassNames("gfx92a", tuning);

  const int ws = IndexOf(names, "tritongpu-automatic-warp-specialization-hcu");
  ASSERT_GE(ws, 0) << "warp-specialization pass missing";

  // accelerate-matmul runs twice: once before warp specialization and once
  // after, because the warp-specialization pass resets the layouts of every
  // partition whose warp count changed and leaves the dots un-accelerated.
  EXPECT_EQ(CountOf(names, "tritonhcu-accelerate-matmul"), 2);
  int second = -1;
  for (int i = ws + 1; i < names.size(); ++i) {
    if (names[i] == "tritonhcu-accelerate-matmul") {
      second = i;
      break;
    }
  }
  EXPECT_GE(second, 0)
      << "the post-warp-specialization accelerate-matmul is missing; MMA "
         "partitions would keep no MMAC encoding";

  // The LLIR lowering must be present and come after the TTGIR pass.
  const int llir = IndexOf(names, "triton-hcu-convert-warp-specialize-to-llvm");
  EXPECT_GE(llir, 0) << "HCU warp-specialize lowering missing";
  EXPECT_GT(llir, ws);
}

// A non-HCU ROCm target must keep using the AMD warp-specialize lowering.
TEST(CompilationPipelineTest, NonHcuRocmKeepsAmdPath) {
  TritonHcuTuning tuning;
  tuning.enable_warp_specialization = true;
  std::vector<std::string> names = GetRocmPassNames("gfx942", tuning);
  EXPECT_THAT(names, Not(Contains("tritongpu-automatic-warp-specialization-hcu")));
  EXPECT_THAT(names, Not(Contains("triton-hcu-convert-warp-specialize-to-llvm")));
  EXPECT_THAT(names,
              Contains("triton-amdgpu-convert-warp-specialize-to-llvm"));
}

TEST(CompilationPipelineTest, ContainsUnswitchLoopsCompositePass) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  CreateTritonXlaPipeline(&pm, stream_executor::CudaComputeCapability(),
                          /*rewrite_int4=*/false, /*allow_tma=*/true,
                          /*num_stages=*/1,
                          /*warp_specialization_allowed=*/true,
                          /*enable_pdl=*/false);

  std::vector<std::string> pass_names;
  for (const mlir::Pass& pass : pm.getPasses()) {
    pass_names.push_back(pass.getName().str());
  }
  ASSERT_THAT(pass_names, Contains("TritonXLAUnswitchLoopsComposite"));
}

}  // namespace gpu
}  // namespace xla
