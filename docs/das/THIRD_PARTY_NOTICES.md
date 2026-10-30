# Third Party Notices

This project is not an official XLA release but is instead a derivative based on [ROCm/xla](https://github.com/ROCm/xla); ROCm XLA itself is derived from [openxla/xla](https://github.com/openxla/xla). The base project is listed first; 
all remaining sections correspond one-to-one with the entries under `third_party/` 
and are ordered to match that directory listing.

## XLA

- Source: https://github.com/ROCm/xla
- Fixed source commit: `d5734fef24ced94d13d7f28a5d41868e78415ffa`
- License: Apache License, Version 2.0
- Original copyright: Copyright The OpenXLA Authors
- Hygon modifications: HCU platform adaptations for the DTK toolchain, HCU codegen backend enablement, ROCm stream-executor and hipBLASLt runtime adjustments, Triton compilation-pipeline routing and IR-builder fixes for HCU targets, LLVM/Triton build-patch and workspace updates, and legacy HCU gfx9 target support.

---

| Project | Repository URL | Version / Commit | License | Local path | Hygon modifications |
| --- | --- | --- | --- | --- | --- |
| FP16 | https://github.com/Maratyszcza/FP16 | `4dfe081cf6bcd15db339cf2680b9281b8451eeb3` | MIT License | `third_party/FP16` | — |
| Abseil | https://github.com/abseil/abseil-cpp | `255c84dadd029fd8ad25c5efb5933e47beaa00c7` (LTS 20260107.1) | Apache License, Version 2.0 | `third_party/absl` | — |
| Google Benchmark | https://github.com/google/benchmark | `f7547e29ccaed7b64ef4f7495ecfff1c9f6f3d03` | Apache License, Version 2.0 | `third_party/benchmark` | — |
| BoringSSL | https://boringssl.googlesource.com/boringssl/ | `c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc` | OpenSSL License and ISC License | `third_party/boringssl.BUILD`, `third_party/boringssl.patch` | — |
| Brotli | https://github.com/google/brotli | 1.1.0 | MIT License | `third_party/brotli` | — |
| CCCL | https://github.com/NVIDIA/cccl | 2.8.5 / 3.2.0 | Apache License, Version 2.0 | `third_party/cccl` | — |
| Clang Prebuilt Toolchain | https://commondatastorage.googleapis.com/chromium-browser-clang (prebuilt Clang releases from https://github.com/llvm/llvm-project) | `b4160cb94c54f0b31d0ce14694950dac7b6cd83f` | Apache License, Version 2.0 with LLVM Exceptions | `third_party/clang_toolchain` | — |
| Compute Library | https://github.com/ARM-software/ComputeLibrary | 24.12 | MIT License | `third_party/compute_library` | — |
| cpuinfo | https://github.com/pytorch/cpuinfo | `8a9210069b5a37dd89ed118a783945502a30a4ae` | BSD-2-Clause License | `third_party/cpuinfo` | — |
| cuDNN Frontend | https://github.com/NVIDIA/cudnn-frontend | 1.22.1 | MIT License | `third_party/cudnn_frontend` | — |
| Curl | https://github.com/curl/curl | 8.11.0 | curl License | `third_party/curl.BUILD` | — |
| CUTLASS | https://github.com/NVIDIA/cutlass | 3.8.0 | BSD-3-Clause License | `third_party/cutlass` | — |
| Cython | https://github.com/cython/cython | 3.1.2 | Apache License, Version 2.0 | `third_party/cython.BUILD` | — |
| DLPack | https://github.com/dmlc/dlpack | 1.1 | Apache License, Version 2.0 | `third_party/dlpack` | — |
| DUCC | https://gitlab.mpcdf.mpg.de/mtr/ducc | `aa46a4c21e440b3d416c16eca3c96df19c74f316` | GNU General Public License v2.0 (with additional MIT-licensed components) | `third_party/ducc` | — |
| Eigen | https://gitlab.com/libeigen/eigen | `dcbaf2d608f306450f1e74949eb87e9a22a7ef4b` | Mozilla Public License 2.0 (with additional components under Apache 2.0, BSD-3-Clause and LGPL-2.1-only / LGPL-2.1-or-later) | `third_party/eigen3` | — |
| FarmHash | https://github.com/google/farmhash | `0d859a811870d10f53a594927d0d0b97573ad06d` | MIT License | `third_party/farmhash` | — |
| FMT | https://github.com/fmtlib/fmt | 8.1.1 | MIT License | `third_party/fmt` | — |
| FXdiv | https://github.com/Maratyszcza/FXdiv | `63058eff77e11aa15bf531df5dd34395ec3017c8` | MIT License | `third_party/fxdiv` | — |
| GEMM-Lowp | https://github.com/google/gemmlowp | `16e8662c34917be0065110bfcd9cc27d30f52fdf` | Apache License, Version 2.0 | `third_party/gemmlowp` | — |
| Gloo | https://github.com/facebookincubator/gloo | `54cbae0d3a67fa890b4c3d9ee162b7860315e341` | BSD-3-Clause License | `third_party/gloo` | — |
| GoogleTest | https://github.com/google/googletest | `28e9d1f26771c6517c3b4be10254887673c94018` | Apache License, Version 2.0 | `third_party/googletest` | — |
| CUDA | https://developer.nvidia.com/cuda-toolkit | 12.9.1 (default hermetic) | NVIDIA CUDA Toolkit EULA (Proprietary / Closed Source) | `third_party/gpus/cuda` | — |
| cuDNN | https://developer.nvidia.com/cudnn | 9.8.0 (default hermetic) | NVIDIA cuDNN Software License Agreement (Proprietary / Closed Source) | `third_party/gpus/cuda/hermetic` | — |
| ROCm | https://github.com/ROCm/ROCm | 7.10.0  | MIT License | `third_party/gpus/rocm` | Yes |
| HIP | https://github.com/ROCm/hip | bundled with ROCm  | MIT License | `third_party/gpus/rocm` | Yes |
| oneAPI Level Zero | https://github.com/oneapi-src/level-zero | 1.21.10 | MIT License | `third_party/gpus/sycl/level_zero.bzl` | — |
| Intel oneAPI Base Toolkit | https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html | 2025.1.3.7 | Intel End User License Agreement for the Intel(R) oneAPI Base Toolkit (Proprietary / Closed Source redistributables) | `third_party/gpus/sycl/sycl_dl_essential.bzl` | — |
| gRPC | https://github.com/grpc/grpc | 1.78.0 | Apache License, Version 2.0 | `third_party/grpc` | — |
| gutil | https://github.com/google/gutil | `b498c8d364ac96c32194f71f8f719707a398e82b` (LTS 20250502.0) | Apache License, Version 2.0 | `third_party/gutil` | — |
| Highwayhash | https://github.com/google/highwayhash | `c13d28517a4db259d738ea4886b1f00352a3cc33` | Apache License, Version 2.0 | `third_party/highwayhash` | — |
| hwloc | https://github.com/open-mpi/hwloc | 2.7.1 | BSD-3-Clause License | `third_party/hwloc` | — |
| Implib.so | https://github.com/yugr/Implib.so | `2cce6cab8ff2c15f9da858ea0b68646a8d62aef2` | MIT License | `third_party/implib_so` | — |
| LLVM Project | https://github.com/llvm/llvm-project | `9a4faee1068c09efbf837cfb7b0f5693b24635f4` | Apache License, Version 2.0 with LLVM Exceptions | `third_party/llvm` | Yes |
| LLVM OpenMP Runtime | https://github.com/llvm/llvm-project (openmp 10.0.1 release tarball) | 10.0.1 | Apache License, Version 2.0 with LLVM Exceptions | `third_party/llvm_openmp` | — |
| Intel MKL | https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html | n/a (license-declaration stub) | BSD-3-Clause License (as declared by the vendored Bazel package) | `third_party/mkl` | — |
| oneDNN (MKL-DNN) | https://github.com/oneapi-src/oneDNN | 3.7.3 | Apache License, Version 2.0 | `third_party/mkl_dnn` | — |
| MPItrampoline | https://github.com/eschnett/mpitrampoline | `25efb0f7a4cd00ed82bafb8b1a6285fc50d297ed` | MIT License | `third_party/mpitrampoline` | — |
| nanobind | https://github.com/wjakob/nanobind | `30f12ae6650ecec86042053d522d9af585f269b0` | BSD-3-Clause License | `third_party/nanobind` | — |
| NASM | https://www.nasm.us/ | 2.14.02 | BSD-2-Clause License | `third_party/nasm` | — |
| NCCL | https://github.com/NVIDIA/nccl | 2.29.7-1 | Apache License, Version 2.0 | `third_party/nccl` | — |
| net_zstd (Zstandard) | https://github.com/facebook/zstd | 1.5.7 | BSD-3-Clause License + GPLv2 License | `third_party/net_zstd` | — |
| NVSHMEM | https://github.com/NVIDIA/nvshmem | 3.1.7-1 | Apache License, Version 2.0 | `third_party/nvshmem` | — |
| NVTX | https://github.com/NVIDIA/NVTX | 3.5.0 | Apache License, Version 2.0 (with LLVM Exception) | `third_party/nvtx` | — |
| Google OR-Tools | https://github.com/google/or-tools | 9.11 | Apache License, Version 2.0 | `third_party/ortools` | — |
| Protocol Buffers | https://github.com/protocolbuffers/protobuf | 34.1 | BSD-3-Clause License | `third_party/protobuf` | — |
| PThreadpool | https://github.com/google/pthreadpool | `9003ee6c137cea3b94161bd5c614fb43be523ee1` | BSD-2-Clause License | `third_party/pthreadpool` | — |
| absl-py | https://github.com/abseil/abseil-py | 2.1.0 | Apache License, Version 2.0 | `third_party/py/absl_py` | — |
| ml_dtypes | https://github.com/jax-ml/ml_dtypes | `00d98cd92ade342fef589c0470379abb27baebe9` | Apache License, Version 2.0 | `third_party/py/ml_dtypes` | — |
| NumPy | https://github.com/numpy/numpy | 2.4.4 (pinned in `requirements_lock_3_11.txt`) | BSD-3-Clause License | `third_party/py/numpy` | — |
| pybind11 | https://github.com/pybind/pybind11 | 2.13.6 | BSD-3-Clause License | `third_party/pybind11.BUILD` | — |
| Bazel pybind11_abseil | https://github.com/pybind/pybind11_abseil | `13d4f99d5309df3d5afa80fe2ae332d7a2a64c6b` | Apache License, Version 2.0 | `third_party/pybind11_abseil` | — |
| Bazel pybind11_bazel | https://github.com/pybind/pybind11_bazel | 2.13.6 | Apache License, Version 2.0 | `third_party/pybind11_bazel` | — |
| RAFT | https://github.com/rapidsai/raft | 26.02.00 | Apache License, Version 2.0 | `third_party/raft` | — |
| rapids-logger | https://github.com/rapidsai/rapids-logger | 0.2.3 | Apache License, Version 2.0 | `third_party/rapids_logger` | — |
| Riegeli | https://github.com/google/riegeli | `9f2744dc23e81d84c02f6f51244e9e9bb9802d57` | Apache License, Version 2.0 | `third_party/riegeli` | — |
| RMM | https://github.com/rapidsai/rmm | 26.02.00 | Apache License, Version 2.0 | `third_party/rmm` | — |
| RobinMap | https://github.com/Tessil/robin-map | 1.3.0 | MIT License | `third_party/robin_map` | — |
| ROCm Device Libraries | https://github.com/ROCm/llvm-project (`amd/device-libs`) | `04484e4c8fa747928c59c49fd063e6d7eea9fd87` | University of Illinois/NCSA Open Source License | `third_party/rocm_device_libs` | — |
| Bazel rules_python | https://github.com/bazelbuild/rules_python | 1.8.5 | Apache License, Version 2.0 | `third_party/rules_python` | — |
| Shardy | https://github.com/openxla/shardy | `4b908d99c4be7e7809d57d2b89cf1404d6e52f68` | Apache License, Version 2.0 | `third_party/shardy` | — |
| six | https://pypi.org/project/six/ | 1.16.0 | MIT License | `third_party/six.BUILD` | — |
| Slinky | https://github.com/dsharlet/slinky | `1032be67d7033d736eb6afbb9b51865baa5c77ae` | Apache License, Version 2.0 | `third_party/slinky` | — |
| Snappy | https://github.com/google/snappy | 1.2.1 | BSD-3-Clause License | `third_party/snappy.BUILD` | — |
| spdlog | https://github.com/gabime/spdlog | 1.15.2 | MIT License | `third_party/spdlog` | — |
| SPIRV-LLVM-Translator | https://github.com/KhronosGroup/SPIRV-LLVM-Translator | `dad1f0eaab8047a4f73c50ed5f3d1694b78aae97` | University of Illinois/NCSA Open Source License | `third_party/spirv_llvm_translator` | — |
| StableHLO | https://github.com/openxla/stablehlo | `806a6844dfd92cca1ce5391c86dca0ef9e952550` | Apache License, Version 2.0 | `third_party/stablehlo` | — |
| TensorRT | https://github.com/NVIDIA/TensorRT | `9ec6eb6db39188c9f3d25f49c8ee3a9721636b56` | Apache License, Version 2.0 | `third_party/tensorrt` | — |
| Transformer Engine | https://github.com/NVIDIA/TransformerEngine | 2.5 | Apache License, Version 2.0 | `third_party/transformer_engine` | — |
| OpenAI Triton | https://github.com/triton-lang/triton | `feb6c04abe15e74faa23f0935ae74c0a38c5e37f` | MIT License | `third_party/triton` | Yes |
| TensorFlow Standard Library | https://github.com/openxla/xla/tree/main/xla/tsl | vendored with XLA `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` | Apache License, Version 2.0 | `third_party/tsl`, `xla/tsl` | — |
| libuv | https://github.com/libuv/libuv | 1.38.0 | MIT License | `third_party/uv` | — |
| XNNPACK | https://github.com/google/XNNPACK | `bccfe73347861f07a0257c0c546c0babcf3257b8` | BSD-3-Clause License | `third_party/xnnpack` | — |
| xxd | Part of the Vim project (https://github.com/vim/vim) | Vim 9.1.0917 | Vim License (GPL-compatible) | `third_party/xxd` | — |
| zlib | https://github.com/madler/zlib | 1.3.1 | zlib License (Permissive) | `third_party/zlib.BUILD` | — |

---

Any additional third-party components not listed above but present under `third_party/` retain their upstream licenses. 
See the corresponding `LICENSE`, `LICENSE.txt`, or `README` file inside each subdirectory of `third_party/` for the authoritative terms.
