<!-- DAS-INTRO:BEGIN -->

> [!IMPORTANT]
> **xla-das** 是面向 DAS 架构的 xla 下游适配发行版，
> 基于 [ROCm/xla](https://github.com/ROCm/xla) 的 `76282465a00b54ddcef59a2ee95412cd5ecf5551` 基线构建并集成 DAS 支持。
> 
> 本项目不是 xla 官方发行版，而是基于 [ROCm/xla](https://github.com/ROCm/xla) 二次开发；ROCm xla 源自 [openxla/xla](https://github.com/openxla/xla) 二次开发。

- **目标架构：** xla
- **上游分支：** rocm-jaxlib-v0.10.0
- **上游版本：** 76282465a00b54ddcef59a2ee95412cd5ecf5551
- **上游基线：** `https://github.com/ROCm/xla/tree/76282465a00b54ddcef59a2ee95412cd5ecf5551`
- **上游许可：** Apache-2.0
- **源码编译：** [BUILDING.md](docs/das/BUILDING.md)
- **已知问题：** [BUILDING.md 中的已知问题](docs/das/BUILDING.md#已知问题)
- **Python 发布包名：** `xla-das`
- **Python 导入名：** `xla`

> `xla-das` 与上游 `xla` 会安装同名 `xla` 模块，
> 请勿在同一虚拟环境或容器中混装。

<!-- DAS-INTRO:END -->

---

# XLA

XLA (Accelerated Linear Algebra) is an open-source machine learning (ML)
compiler for GPUs, CPUs, and ML accelerators.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/openxla_dark.svg">
  <img alt="OpenXLA Ecosystem" src="docs/images/openxla.svg">
</picture>

The XLA compiler takes models from popular ML frameworks such as PyTorch,
TensorFlow, and JAX, and optimizes them for high-performance execution across
different hardware platforms including GPUs, CPUs, and ML accelerators.

[openxla.org](https://openxla.org/) is the project's website.

## Get started

If you want to use XLA to compile your ML project, refer to the corresponding
documentation for your ML framework:

* [PyTorch](https://pytorch.org/xla)
* [TensorFlow](https://www.tensorflow.org/xla)
* [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

If you're not contributing code to the XLA compiler, you don't need to clone and
build this repo. Everything here is intended for XLA contributors who want to
develop the compiler and XLA integrators who want to debug or add support for ML
frontends and hardware backends.

## Contribute

If you'd like to contribute to XLA, review
[How to Contribute](docs/contributing.md) and then see the
[developer guide](docs/developer_guide.md).

## Contacts

*   For questions, contact the maintainers - maintainers at openxla.org

## Resources

*   [Community Resources](https://github.com/openxla/community)

## Code of Conduct

While under TensorFlow governance, all community spaces for SIG OpenXLA are
subject to the
[TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE). Some files are adapted from xla and other third-party sources; 
see [Third-Party Notices](docs/das/THIRD_PARTY_NOTICES.md) for source and modification notices.

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd. are licensed under the Apache License, Version 2.0.

Modified by Hygon Information Technology Co., Ltd.