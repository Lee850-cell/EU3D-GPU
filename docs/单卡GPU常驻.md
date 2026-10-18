# 单卡 GPU 常驻时间推进

`EU3D_FULL_RESIDENT=1` 将普通时间步的边界、流体、状态恢复和化学反应连接成设备端流程。默认关闭。这个版本只支持单 MPI rank、固定 dt、RSBI、MUSCL_2、EE、自适应梯形反应、关闭 DLB，并依赖之前的状态链、packed output、resident EE 和 resident MUSCL。多卡开启会被配置检查拒绝。

## 数据生命周期

1. 初始化及首次种子状态仍由 CPU 提供。首步使用原输入上传路径。
2. 反应结束后保留设备状态、源项与 Nchem，不下载整个场，也不在 CPU 回填。
3. 下一步从设备状态建立 MUSCL 输入。x/z 镜像边界和 y 后边界在 GPU 上处理；y 前边界在 `resident_front_thermo` 内由镜像温度、组分和原热力学公式重新求 D/H/Gamma。
4. EE 使用设备上的守恒量及通量 RHS，状态恢复和化学反应继续在同一条设备状态链上执行。
5. GPU 归约检查 Nchem 和 D/T 的有限性，每步只回传 3 个整数（12 字节）。CPU 记录最大子步数和日志。
6. `Euler::Output()` 调用 `SyncResidentOutput()`，才下载并回填完整输出状态及 Nchem。文件包含 ghost cells，因此输出同步时在 CPU 重建当前输出边界；这些主机边界不参与后续 GPU 推进。初始化、输出准备、文件写入、MPI 调用和调度不宣称全部由 GPU 完成。

常量目前仍有小量重复上传，首次启动仍需种子上传；“GPU 常驻”指普通时间步不再搬运完整流场，并非完全没有 CPU 或 PCIe 活动。

## 代码入口

- `src/ReactionCuda.cu`：`configure_resident_ee_state` 检查支持范围；`resident_health_kernel` 做 GPU 检查；`download_resident_output` 提供显式输出同步。
- `src/FlowfieldCuda.cu`：`resident_front_thermo` 计算前边界；`build_resident_muscl` 在 full 模式不再上传 CPU 前边界。
- `src/Euler.cpp`：普通步跳过 CPU 物理边界、全场检查及反应回填；`SyncResidentOutput` 同步输出。
- `include/ResidentMusclState.hpp`：只保存设备缓冲区的借用指针、形状、常量视图和有效性，不拥有这些缓冲区。每次 Computing 重置，反应结束后发布，EE 更新前失效，输出后清除 dirty。

## 本机运行

```bash
cd ~/code/mission8/EU3D
make -j4
bash scripts/run_full_resident.sh 100
```

脚本复制输入到独立 runs 目录，使用 281×141×2、100 步，不改 input 模板。运行环境为本机 RTX 3050 Laptop、CUDA 12.5。不要把本机结果当作 A800 或多卡性能。

```bash
bash tests/validate_full_resident.sh
bash scripts/benchmark_full_resident.sh
```

验证三条路径：旧驻留路径、full 模式每步同步的 shadow 路径、full 模式延迟同步路径。shadow 使用 `EU3D_VALIDATE_FULL_RESIDENT=1`，与同一步 CPU 边界按原严格容差 `1e-12 + 1e-10*abs(reference)` 对照全部 MUSCL 输入及 halo，同时报告逐位差异。CPU libm 与 CUDA 热力学不保证逐位一致；旧的纯复制驻留路径仍要求逐位一致。

整程序 3/100 步与旧路径使用原严格容差比较；shadow 与延迟同步要求所有导出字段最大绝对差为 0，并逐字节比较中途/最终可视化输出。还比较整个 Nchem 最大值历史，核查输出同步次数和每步健康检查次数。中途输出通过 count=2 覆盖，不能只测最终一次导出。

## 限制

- 尚未实现此路径的多卡 GPU halo 交换；之前的多卡旧路径保留。
- 本次验证是和自己的旧 GPU 路径比较，不解决此前与 CPU 或第三方版本已有的差异。
- full 模式输出时重建当前 ghost cells；旧路径输出保留此前时间步的 ghost cells，因此两者输出文件的 ghost 区域不作为逐位基准。内部状态仍按原比较标准验证。
- 支持场景之外不得通过删除 guard 强行开启。

## 验证记录（2026-10-18）

本机 281×141×2：3/100 步均通过，和旧路径所有导出内部字段最大绝对差均为 0；shadow 与按需同步也完全一致。中途及最终可视化文件逐字节一致，Nchem 最大值完整历史一致。GPU 边界的局部焓等计算有少量末位舍入差异，但均通过原严格容差，不能宣称边界逐位一致。

原流体上传复用回归通过 9 个 sweep（方向、输入改变、尺寸改变、独立调用），通量逐位一致。

原始验证目录：`runs/full_resident_check_RIYFnYHf/`。摘要和对比日志：`docs/full_resident_results/`。

计时需同时报告 Loop time 与 Total Time：最终状态的显式下载位于主循环后、文件输出前，计入总时间，不计入主循环。不得把这部分遗漏当作纯计算加速。

## 性能（三轮交替测试）

两组都开启此前的 RESIDENT_EE 和 RESIDENT_MUSCL，仅切换 FULL_RESIDENT。全部验证与状态 dump 关闭，正常输出保留，281×141×2、100 步。

| 三轮中位数 | 上一阶段 | FULL_RESIDENT |
| --- | ---: | ---: |
| 主循环 | 16.8822 s | 13.0926 s |
| 总时间 | 20.4313 s | 16.3662 s |

主循环耗时减少 22.45%，总时间减少约 19.90%。有运行波动，尤其输出时间；这是本机小网格结果，不代表 A800、多卡或其他网格。最终完整下载已计入 Total Time。

性能目录：`runs/full_resident_perf_EUBH9jaK/`，原始三轮数据见 `docs/full_resident_results/summary.json`。

补充 `bash tests/validate_full_resident.sh volume`：17×13×4、3/100 步也通过原严格容差，与 shadow 的内部状态、输出文件及子步数历史检查均通过。目录 `runs/full_resident_check_OYgagxvU/`，摘要位于 `docs/full_resident_results/volume/`。
