#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# The script is maintained in XLA (build_tools/hcu/) but *runs in JAX workspace*

set -ex -o history -o allexport

source ci/envs/default.env

DTK_DIR="${DTK_DIR:-/opt/dtk}"
AILLVM_DIR="${AILLVM_DIR:-${DTK_DIR}/aillvm}"
TF_ROCM_HCUGPU_TARGETS="${TF_ROCM_HCUGPU_TARGETS:-gfx906,gfx926,gfx928,gfx936,gfx938}"
HCU_CODEGEN_CONFIG="${HCU_CODEGEN_CONFIG:-hcu}"
HCU_CLANG="${AILLVM_DIR}/bin/clang"
HCU_CLANGXX="${AILLVM_DIR}/bin/clang++"
if [[ ! -x "${HCU_CLANG}" ]]; then
  echo "HCU LLVM clang not found: ${HCU_CLANG}" >&2
  echo "Set AILLVM_DIR=/path/to/aillvm." >&2
  exit 1
fi

if [[ -z "$JAXCI_XLA_GIT_DIR" && "$JAXCI_CLONE_MAIN_XLA" != 1 ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

source "ci/utilities/setup_build_environment.sh"

OVERRIDE_XLA_REPO=""
if [[ -n "$JAXCI_XLA_GIT_DIR" ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

# align with rocm upstream
TAG_FILTERS="jax_test_gpu,-config-cuda-only,-manual"
TESTS_TO_IGNORE=(
    -//tests/pallas:pallas_test_gpu
    -//tests/pallas:ops_test_gpu
    -//tests/pallas:ops_test_mgpu_gpu
    -//tests/pallas:pallas_shape_poly_test_gpu
    -//tests/pallas:pallas_vmap_test_gpu
    -//tests/pallas:triton_pallas_test_gpu
    -//tests:export_harnesses_multi_platform_test_gpu
    -//tests:jet_test_gpu
    -//tests:lax_autodiff_test_gpu
    -//tests:lax_numpy_setops_test_gpu
    -//tests:lax_numpy_test_gpu
    -//tests:lax_test_gpu
    -//tests:linalg_test_gpu
    -//tests:logging_test_gpu
    -//tests:random_lax_test_gpu
    -//tests:scipy_signal_test_gpu
    -//tests:stax_test_gpu
    -//tests:ode_test_gpu
    -//tests:lobpcg_test_gpu
    -//tests:scipy_stats_test_gpu
    -//tests:nn_test_gpu
    -//tests:lax_scipy_sparse_test_gpu
    -//tests:lax_scipy_spectral_dac_test_gpu
    -//tests:lax_scipy_special_functions_test_gpu
    -//tests:cholesky_update_test_gpu
    -//tests:api_test_gpu
    -//tests:ann_test_gpu
    -//tests:experimental_rnn_test_gpu
    -//tests:lax_vmap_test_gpu
    -//tests:qdwh_test_gpu
    -//tests:scaled_dot_test_gpu
    -//tests:scipy_spatial_test_gpu
    -//tests:shape_poly_test_gpu
    -//tests:sparsify_test_gpu
    -//tests:lax_numpy_reducers_test_gpu
    -//tests:scipy_optimize_test_gpu
)

GPU_MODE="single_gpu"
BAZEL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --single_gpu)
            GPU_MODE="single_gpu"
            ;;
        --multi_gpu)
            GPU_MODE="multi_gpu"
            ;;
        --//jax:build_jaxlib=false)
            # Tests to ignore for pre-built plugin wheels.
            TESTS_TO_IGNORE+=(
                -//tests:buffer_callback_test_gpu
            )
            BAZEL_ARGS+=("$arg")
            ;;
        *)
            BAZEL_ARGS+=("$arg")
            ;;
    esac
done

if [[ -z "$ROCMINFO" || ! -x "$ROCMINFO" ]]; then
  for candidate in "${ROCM_PATH:-/opt/dtk}/bin/rocminfo" "/opt/dtk/bin/rocminfo"; do
    if [[ -x "$candidate" ]]; then
      ROCMINFO="$candidate"
      break
    fi
  done
fi

if [[ -z "$ROCMINFO" || ! -x "$ROCMINFO" ]]; then
  echo "ERROR: cannot locate rocminfo (looked in runfiles and \${ROCM_PATH:-/opt/dtk}/bin)." >&2
  echo "Refusing to run: without a device count, tests would run unlocked and share one HCU." >&2
  exit 1
fi

TF_GPU_COUNT=$("$ROCMINFO" | grep -c "Name: *gfx")
TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU:-4}
if [[ "$TF_GPU_COUNT" -eq 0 ]]; then
  echo "No HCU devices reported by ${ROCMINFO}." >&2
  exit 1
fi

echo "HCU devices: ${TF_GPU_COUNT}, tests per device: ${TF_TESTS_PER_GPU}"

MODE_ARGS=()
if [[ "$GPU_MODE" == "multi_gpu" ]]; then
    TAG_FILTERS="${TAG_FILTERS},multiaccelerator"
    MODE_ARGS+=(
        --run_under=
        --test_sharding_strategy=disabled
        --local_test_jobs=1
        --test_env=HIP_VISIBLE_DEVICES="$(seq -s, 0 $((TF_GPU_COUNT - 1)))"
    )
else
    TAG_FILTERS="${TAG_FILTERS},gpu,-multiaccelerator"
    MODE_ARGS+=(
        --run_under=@xla//build_tools/hcu:parallel_gpu_execute
        --test_env=TF_GPU_COUNT="${TF_GPU_COUNT}"
        --test_env=TF_TESTS_PER_GPU="${TF_TESTS_PER_GPU}"
        --local_test_jobs=1
    )
fi

HIPBLASLT_ENV_FILE="${DTK_DIR}/env.hipblaslt"
if [[ -z "${HIPBLASLT_TENSILE_LIBPATH:-}" && -f "${HIPBLASLT_ENV_FILE}" ]]; then
    source "${HIPBLASLT_ENV_FILE}"
fi
HIPBLASLT_TEST_ENV_ARGS=()
if [[ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
    HIPBLASLT_TEST_ENV_ARGS+=("--test_env=HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}")
fi

# Don't abort before the test XMLs are collected.
set +e

bazel --bazelrc=build/rocm/rocm.bazelrc test \
    --config=rocm \
    --config="${HCU_CODEGEN_CONFIG}" \
    $OVERRIDE_XLA_REPO \
    --repo_env=ROCM_PATH="${DTK_DIR}" \
    --action_env=ROCM_PATH="${DTK_DIR}" \
    --test_env=ROCM_PATH="${DTK_DIR}" \
    --action_env=TF_ROCM_CLANG=1 \
    --action_env=TF_HIPCC_CLANG=1 \
    --action_env=CLANG_COMPILER_PATH="${HCU_CLANG}" \
    --repo_env=CC="${HCU_CLANG}" \
    --repo_env=CXX="${HCU_CLANGXX}" \
    --repo_env=BAZEL_COMPILER="${HCU_CLANG}" \
    --repo_env=TF_ROCM_HCUGPU_TARGETS="${TF_ROCM_HCUGPU_TARGETS}" \
    --action_env=TF_ROCM_HCUGPU_TARGETS="${TF_ROCM_HCUGPU_TARGETS}" \
    --repo_env=HERMETIC_PYTHON_VERSION="${JAXCI_HERMETIC_PYTHON_VERSION}" \
    --//jax:build_jaxlib="${JAXCI_BUILD_JAXLIB}" \
    --//jax:build_jax="${JAXCI_BUILD_JAX}" \
    --action_env=JAX_ENABLE_X64="${JAXCI_ENABLE_X64}" \
    --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
    --test_env=JAX_SKIP_SLOW_TESTS=true \
    --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
    --test_output=errors \
    --test_verbose_timeout_warnings \
    --build_tag_filters="${TAG_FILTERS}" \
    --test_tag_filters="${TAG_FILTERS}" \
    --spawn_strategy=local \
    --strategy=TestRunner=local \
    --color=yes \
    "${HIPBLASLT_TEST_ENV_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${BAZEL_ARGS[@]}" \
    -- \
    //tests:gpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:gpu_tests \
    //tests/pallas:backend_independent_tests \
    //jaxlib/tools:check_gpu_wheel_sources_test \
    "${TESTS_TO_IGNORE[@]}"

bazel_retval=$?

set -e

bash ci/utilities/collect_bazel_test_xmls.sh test-artifacts

exit "${bazel_retval}"
