#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Usage:
#   build_tools/hcu/run_xla_ci_build.sh --config=hcu --config=ci_single_gpu [extra bazel flags...]
#   build_tools/hcu/run_xla_ci_build.sh --config=hcu --config=ci_multi_gpu  [extra bazel flags...]
#
# Environment overrides:
#   DTK_DIR     DTK installation. Defaults to /opt/dtk.
#   AILLVM_DIR  HCU LLVM installation. Defaults to ${DTK_DIR}/aillvm.
#   PROFILE_DIR Bazel profile output. Defaults to <workspace>/ci_artifacts.

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")
WORKSPACE_DIR=$(realpath "${SCRIPT_DIR}/../..")

DTK_DIR="${DTK_DIR:-/opt/dtk}"
AILLVM_DIR="${AILLVM_DIR:-${DTK_DIR}/aillvm}"
HCU_CLANG="${AILLVM_DIR}/bin/clang"

if [[ ! -x "${HCU_CLANG}" ]]; then
    echo "HCU LLVM clang not found: ${HCU_CLANG}" >&2
    echo "Set AILLVM_DIR=/path/to/aillvm." >&2
    exit 1
fi

EXCLUDED_TESTS=(
# The HipnnConvolution_1257 algorithm in convolution_test exhibits FP32-to-FP16 conversion mismatches against the IEEE standard
# This behavior is intentionally designed in DTK, so the test is skipped
    "Convolve2D_1x3x3x5_3x3x5x3_Valid/1.Types"
# gpu_hlo_schedule_test abort on test exit, wait DTK fix
    "GpuHloScheduleParameterizedTest/*"
# Tiling params not support on rocm for too large shard memory using
    "TritonFusionNumericsVerifierTestSuite/TritonFusionNumericsVerifierTest.VerifyMultipleNestedFusionNumerics/*"
# HCU not support libprofiler sdk now
    "HloOpProfilerTest.*"
    "MatmulPerfTableGenTest.ProfilesSmallMatmul"
# HCU Triton backend cannot legalize the f8e5m2->f32 upcast
    "DotTestTestSuite/DotTest.IsTritonSupportedExecutesCorrectlyForDot/f8e5m2_dot"
# Align with rocm upstream excluded tests
    "HostMemoryAllocateTest.Numa"
    "NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_tf32_tf32_f32_x3"
    "TritonEmitterTest.ScaledDotIsSupportedByReferencePlatform"
    "VmmTest.CommandBufferSkipProfiledTwoGemmChain"
    "GpuKernelTilingTest.ReductionInputTooLarge"
    "MxScaledDotExecutionTest.MxFp4Fp8MixedBatchedCorrectness"
    "ConvolutionTest.Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid"
    "ConvolutionTest.Convolve_1x1x4x4_1x1x2x2_Valid"
    "ConvolutionTest.Convolve_1x1x4x4_1x1x2x2_Same"
    "ConvolutionTest.Convolve_1x1x4x4_1x1x3x3_Same"
    "StreamExecutorGpuCompilerTest.AotCompileDeserializeRoundTrip"
    "StreamExecutorGpuClientTest.NumaNode"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll*"
)

TAG_FILTERS=$(bash "${SCRIPT_DIR}/hcu_tag_filters.sh")

TARGET_SET_ARGS=()
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --config=ci_multi_gpu)
            TAG_FILTERS="${TAG_FILTERS},multi_gpu"
            PASSTHROUGH_ARGS+=("$arg")
            ;;
        --config=ci_single_gpu)
            TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
            PASSTHROUGH_ARGS+=("$arg")
            ;;
        --config=hcu_sgpu|--config=hcu_mgpu)
            TARGET_SET_ARGS+=("$arg")
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

TEST_FILTER_ARG=()
if ((${#EXCLUDED_TESTS[@]})); then
    TEST_FILTER_ARG=(--test_filter="-$(IFS=:; echo "${EXCLUDED_TESTS[*]}")")
fi

PROFILE_DIR="${PROFILE_DIR:-${WORKSPACE_DIR}/ci_artifacts}"
mkdir -p "${PROFILE_DIR}"

cd "${WORKSPACE_DIR}"

HIPBLASLT_ENV_FILE="${DTK_DIR}/env.hipblaslt"
if [[ -z "${HIPBLASLT_TENSILE_LIBPATH:-}" && -f "${HIPBLASLT_ENV_FILE}" ]]; then
    source "${HIPBLASLT_ENV_FILE}"
fi
HIPBLASLT_TEST_ENV_ARGS=()
if [[ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
    HIPBLASLT_TEST_ENV_ARGS+=("--test_env=HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}")
fi

bazel --bazelrc="${SCRIPT_DIR}/hcu_xla.bazelrc" test \
    --config=hcu_test \
    --build_tag_filters="${TAG_FILTERS}" \
    --test_tag_filters="${TAG_FILTERS}" \
    --profile="${PROFILE_DIR}/profile.json.gz" \
    --spawn_strategy=local \
    --strategy=TestRunner=local \
    --nokeep_going \
    --cache_test_results=yes \
    --curses=no \
    --color=yes \
    "${TEST_FILTER_ARG[@]}" \
    "${HIPBLASLT_TEST_ENV_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}" \
    --repo_env=ROCM_PATH="${DTK_DIR}" \
    --action_env=ROCM_PATH="${DTK_DIR}" \
    --action_env=TF_ROCM_CLANG=1 \
    --action_env=CLANG_COMPILER_PATH="${HCU_CLANG}" \
    "${TARGET_SET_ARGS[@]}"
