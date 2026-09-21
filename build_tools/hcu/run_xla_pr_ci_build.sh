#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")
WORKSPACE_DIR=$(realpath "${SCRIPT_DIR}/../..")
CI_BUILD_SCRIPT="${SCRIPT_DIR}/run_xla_ci_build.sh"

# ---------------------------------------------------------------------------
# Tier 1 -- GPU compiler / HLO pass tests.
# ---------------------------------------------------------------------------
TIER_PASS=(
    //xla/backends/gpu/transforms:async_wrapper_test_amdgpu_any
    //xla/backends/gpu/transforms:conv_fusion_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:cublas_gemm_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:cudnn_norm_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:dot_dimension_sorter_test_amdgpu_any
    //xla/backends/gpu/transforms:gemm_broadcast_folding_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:gemm_rewriter_allocation_test_amdgpu_any
    //xla/backends/gpu/transforms:gemm_rewriter_fp8_test_amdgpu_any
    //xla/backends/gpu/transforms:gemm_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:onehot_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:reduction_layout_normalizer_test_amdgpu_any
    //xla/backends/gpu/transforms:scatter_determinism_expander_test_amdgpu_any
    //xla/backends/gpu/transforms:sort_rewriter_test_amdgpu_any
    //xla/backends/gpu/transforms:topk_splitter_test_amdgpu_any
    //xla/backends/gpu/transforms:topk_specializer_test_amdgpu_any
    //xla/service/gpu:conv_layout_normalization_test_amdgpu_any
    //xla/service/gpu:custom_call_test_amdgpu_any
    //xla/service/gpu:gpu_compiler_legacy_hlo_runner_test_amdgpu_any
    //xla/service/gpu:gpu_compiler_test_amdgpu_any
    //xla/service/gpu:gpu_offloading_test_amdgpu_any
    //xla/service:batchnorm_expander_test_amdgpu_any
    //xla/service:compiler_test_amdgpu_any
    //xla/service:dynamic_padder_ir_test_amdgpu_any
    //xla/service:dynamic_update_slice_test_amdgpu_any
    //xla/service:elemental_ir_emitter_test_amdgpu_any
    //xla/service:hlo_execution_profile_test_amdgpu_any
)

# ---------------------------------------------------------------------------
# Tier 2 -- ROCm device layer: driver, streams, kernels, allocations.
# ---------------------------------------------------------------------------
TIER_ROCM=(
    //xla/stream_executor/rocm:cub_scan_kernel_rocm_test_amdgpu_any
    //xla/stream_executor/rocm:rocm_event_test_amdgpu_any
    //xla/stream_executor/rocm:rocm_executor_test_amdgpu_any
    //xla/stream_executor/rocm:rocm_kernel_test_amdgpu_any
    //xla/stream_executor/rocm:rocm_stream_test_amdgpu_any
    //xla/stream_executor/rocm:rocm_timer_test_amdgpu_any
    //xla/stream_executor/gpu:buffer_debug_log_test_amdgpu_any
    //xla/stream_executor/gpu:gpu_command_buffer_test_amdgpu_any
    //xla/stream_executor/gpu:gpu_executor_test_amdgpu_any
    //xla/stream_executor/gpu:gpu_kernel_test_amdgpu_any
    //xla/stream_executor/gpu:memcpy_test_amdgpu_any
    //xla/stream_executor/gpu:redzone_allocator_test_amdgpu_any
    //xla/stream_executor/gpu:repeat_buffer_kernel_test_amdgpu_any
    //xla/stream_executor/gpu:stream_search_test_amdgpu_any
)

# ---------------------------------------------------------------------------
# Tier 3 -- thunk / runtime execution on device.
# ---------------------------------------------------------------------------
TIER_THUNK=(
    //xla/backends/gpu/runtime:async_execution_test_amdgpu_any
    //xla/backends/gpu/runtime:async_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:buffer_comparator_test_amdgpu_any
    //xla/backends/gpu/runtime:command_buffer_cmd_test_amdgpu_any
    //xla/backends/gpu/runtime:command_buffer_conversion_pass_test_amdgpu_any
    //xla/backends/gpu/runtime:command_buffer_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:custom_call_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:dynamic_slice_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:gpublas_lt_matmul_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:host_execute_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:kernel_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:make_batch_pointers_test_amdgpu_any
    //xla/backends/gpu/runtime:memset_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:print_buffer_contents_test_amdgpu_any
    //xla/backends/gpu/runtime:replica_id_thunk_test_amdgpu_any
    //xla/backends/gpu/runtime:runtime_intrinsics_test_amdgpu_any
    //xla/backends/gpu/runtime:thunk_executor_test_amdgpu_any
    //xla/backends/gpu/runtime:topk_test_amdgpu_any
    //xla/backends/gpu/runtime:while_thunk_test_amdgpu_any
)

# ---------------------------------------------------------------------------
# Tier 4 -- end-to-end numerical and codegen coverage on device.
# ---------------------------------------------------------------------------
TIER_E2E=(
    //xla/backends/gpu/tests:async_command_buffer_test_amdgpu_any
    //xla/backends/gpu/tests:async_kernel_launch_test_amdgpu_any
    //xla/backends/gpu/tests:command_buffer_test_amdgpu_any
    //xla/backends/gpu/tests:dump_autotune_results_to_test_outputs_test_amdgpu_any
    //xla/backends/gpu/tests:dynamic_slice_fusion_test_amdgpu_any
    //xla/backends/gpu/tests:element_wise_row_vectorization_test_amdgpu_any
    //xla/backends/gpu/tests:float_conversions_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_alignment_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_atomic_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_compilation_parallelism_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_copy_alone_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_copy_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_cub_sort_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_dyn_shape_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_ftz_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_index_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_infeed_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_int4_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_kernel_tiling_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_ldg_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_noalias_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_spmd_e2e_compile_test_amdgpu_any
    //xla/backends/gpu/tests:gpu_unrolling_test_amdgpu_any
    //xla/backends/gpu/tests:in_place_op_test_amdgpu_any
    //xla/backends/gpu/tests:kernel_launch_test_amdgpu_any
    //xla/backends/gpu/tests:load_autotune_results_from_test_workspace_test_amdgpu_any
    //xla/backends/gpu/tests:load_autotune_results_using_execpath_test_amdgpu_any
    //xla/backends/gpu/tests:matmul_test_amdgpu_any
    //xla/backends/gpu/tests:mock_custom_call_test_amdgpu_any
    //xla/backends/gpu/tests:multioutput_fusion_test_amdgpu_any
    //xla/backends/gpu/tests:nop_custom_call_test_amdgpu_any
    //xla/backends/gpu/tests:parallel_reduction_test_amdgpu_any
    //xla/backends/gpu/tests:pred_arithmetic_test_amdgpu_any
    //xla/backends/gpu/tests:ragged_dot_test_amdgpu_any
    //xla/backends/gpu/tests:reduction_vectorization_test_amdgpu_any
    //xla/backends/gpu/tests:regression_dot_test_amdgpu_any
    //xla/backends/gpu/tests:select_and_scatter_test_amdgpu_any
    //xla/backends/gpu/tests:simplify_fp_conversions_test_amdgpu_any
    //xla/backends/gpu/tests:sorting_test_amdgpu_any
)

# ---------------------------------------------------------------------------
# Tier 5 -- xla/tests numerics and compilation.
# ---------------------------------------------------------------------------
TIER_NUM=(
    //xla/tests:complex_unary_op_test_amdgpu_any
    //xla/tests:convolution_gpu_alternative_layout_test_amdgpu_any
    //xla/tests:cpu_gpu_fusion_test_amdgpu_any
    //xla/tests:half_test_amdgpu_any
    //xla/tests:int4_test_amdgpu_any
    //xla/tests:llvm_compiler_test_amdgpu_any
    //xla/tests:multithreaded_compilation_test_amdgpu_any
    //xla/tests:reduce_hlo_test_amdgpu_any
    //xla/tests:reduce_window_rewriter_execution_test_amdgpu_any
    //xla/tests:sample_file_test_amdgpu_any
    //xla/tests:sample_text_test_amdgpu_any
    //xla/tests:scatter_deterministic_expander_test_amdgpu_any
    //xla/codegen/intrinsic/accuracy:intrinsic_accuracy_test_amdgpu_any
)

# ---------------------------------------------------------------------------
# Tier 6 -- PJRT GPU client.
# ---------------------------------------------------------------------------
TIER_PJRT=(
    //xla/pjrt/gpu:se_gpu_pjrt_compiler_aot_test_amdgpu_any
    //xla/pjrt/gpu:se_gpu_pjrt_compiler_test_amdgpu_any
)

CORE_TARGETS=(
    "${TIER_PASS[@]}"
    "${TIER_ROCM[@]}"
    "${TIER_THUNK[@]}"
    "${TIER_E2E[@]}"
    "${TIER_NUM[@]}"
    "${TIER_PJRT[@]}"
)

PROFILE_DIR="${PROFILE_DIR:-${WORKSPACE_DIR}/ci_artifacts}"
mkdir -p "${PROFILE_DIR}"

bash "${CI_BUILD_SCRIPT}" \
    --config=hcu \
    --config=ci_single_gpu \
    --local_test_jobs=4 \
    --test_output=errors \
    --test_summary=detailed \
    --test_timeout=600,1800,3600,7200 \
    "$@" \
    "${CORE_TARGETS[@]}"
