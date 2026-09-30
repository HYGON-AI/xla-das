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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_TRITON_CONFIGS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_TRITON_CONFIGS_H_

#include <vector>

#include "xla/service/gpu/matmul_utils.h"

namespace xla::gpu {

enum class TritonConfigsPlatform {
  kAmpere,
  kBlackwell,
  kBlackwellConsumer,  // SM 12.0 (RTX 5090, etc.)
  kDefaultCuda,
  kDefaultRocm,
  // HCU parts, split by register-file size: the 768-entry parts need
  // different tile shapes and warp counts than the 512-entry ones.
  kHcuBowen,
  kDefaultHcu,
  kHopper,
};

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_TRITON_CONFIGS_H_
