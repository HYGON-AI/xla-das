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

#include <cstdio>
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

bool AnyNameContains(absl::Span<const std::string> names, absl::string_view needle) {
  for (const std::string& n : names) {
    if (n.find(needle) != std::string::npos) return true;
  }
  return false;
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
// unaffected.  The HCU warp-specialization passes are not registered in the
// MLIR pass registry, so MLIR reports their class names; match on the common
// "WarpSpecialize" part.
TEST(CompilationPipelineTest, HcuWarpSpecializationOffByDefault) {
  std::vector<std::string> names =
      GetRocmPassNames("gfx92a", TritonHcuTuning());
  EXPECT_FALSE(AnyNameContains(names, "WarpSpecialize"));
}

// The HCU warp-specialization passes are not wired in this tree yet: their
// definitions are not part of any Triton Bazel target (XLA links Triton
// through Bazel), so constructing them would break the link.  Enabling the
// knob must therefore leave the pass list unchanged; restore the
// "wired when enabled" expectations together with the Triton-side Bazel
// target.
TEST(CompilationPipelineTest, HcuWarpSpecializationDeferred) {
  TritonHcuTuning tuning;
  tuning.enable_warp_specialization = true;
  std::vector<std::string> names = GetRocmPassNames("gfx92a", tuning);
  EXPECT_FALSE(AnyNameContains(names, "WarpSpecialize"));
}

// A non-HCU ROCm target must keep using the AMD warp-specialize lowering.
TEST(CompilationPipelineTest, NonHcuRocmKeepsAmdPath) {
  TritonHcuTuning tuning;
  tuning.enable_warp_specialization = true;
  std::vector<std::string> names = GetRocmPassNames("gfx942", tuning);
  EXPECT_TRUE(AnyNameContains(names, "TritonAMDGPUConvertWarpSpecializeToLLVM"));
  EXPECT_FALSE(AnyNameContains(names, "AutomaticWarpSpecializationHCU"));
  EXPECT_FALSE(AnyNameContains(names, "HCUConvertWarpSpecializeToLLVM"));
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
