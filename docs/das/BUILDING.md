# xla-das 构建指导

> [!IMPORTANT]
> **xla-das** 是面向 DAS 架构的 xla 下游适配发行版，
> 基于 [ROCm/xla](https://github.com/ROCm/xla) 的 `76282465a00b54ddcef59a2ee95412cd5ecf5551` 基线构建并集成 DAS 支持。
> 
> 本项目不是 xla 官方发行版，而是基于 [ROCm/xla](https://github.com/ROCm/xla) 二次开发；ROCm xla 源自 [openxla/xla](https://github.com/openxla/xla) 二次开发。

## 运行与工具链依赖

| 依赖名称 | 依赖版本 | 说明 |
| - | - | - |
| Ubuntu | >= 20.04 | x86_64，仅支持 Linux |
| DTK | >= 26.04 | 提供 HIP / hipBLASLt / MIOpen 等运行时，默认安装于 `/opt/dtk` |
| aillvm | >= 18 | 提供优化的 llvm，默认安装于 `/opt/dtk/aillvm` |
| Python | 3.11 / 3.12 | 默认 3.11，依赖锁定见 `requirements_lock_3_11.txt` / `requirements_lock_3_12.txt` |
| Bazel | 7.7.0 | 见 `.bazelversion`，建议使用 bazelisk 自动匹配版本 |

## 支持的 HCU 目标

`gfx906`、`gfx926`、`gfx928`、`gfx936`、`gfx938`。

未显式指定目标时，构建会调用 `rocm_agent_enumerator` 自动探测本机 HCU 架构。

## 构建指导

本项目不单独构建，而是在 jax 中一并构建，[jax 构建指导](https://github.com/HYGON-AI/jax-das/blob/v0.10.0-das/docs/das/BUILDING.md)。

## 已知问题

暂无
