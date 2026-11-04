/* Copyright 2026 The OpenXLA Authors.
   Copyright (c) 2026 Hygon Information Technology Co., Ltd.
   SPDX-License-Identifier: Apache-2.0
   Modified by Hygon Information Technology Co., Ltd., 2026.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep  (defines TF_ROCM_VERSION)
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

namespace xla::gpu {

#if TF_ROCM_VERSION >= 70000
using RcclWindow = ncclWindow_t;
#else
using RcclWindow = void*;
#endif  // TF_ROCM_VERSION >= 70000

// An RCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration. Analogous to
// NcclSymmetricMemory for the ROCm/RCCL platform.
class RcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~RcclSymmetricMemory() final;

  static absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr);

  stream_executor::DeviceAddressBase addr() const final;

  // multimem_addr() and peer_addr() are not supported by RCCL; the base-class
  // defaults (returning Unimplemented) are used.

  RcclWindow win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  RcclSymmetricMemory(ncclComm_t comm, RcclWindow win,
                      stream_executor::DeviceAddressBase addr);

  ncclComm_t comm_;
  RcclWindow win_;
  stream_executor::DeviceAddressBase addr_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
