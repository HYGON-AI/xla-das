# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    default_visibility = ["//visibility:public"],
)

genrule(
    name = "parallel_gpu_execute_script",
    srcs = ["parallel_gpu_execute.sh"],
    outs = ["parallel_gpu_execute_gen.sh"],
    cmd = "cp $< $@ && chmod +x $@",
)

sh_binary(
    name = "parallel_gpu_execute",
    srcs = [":parallel_gpu_execute_script"],
    data = [
        "@local_config_rocm//rocm:rocminfo",
    ],
    visibility = ["//visibility:public"],
)

exports_files([
    "hcu_tag_filters.sh",
    "hcu_xla.bazelrc",
])
