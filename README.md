# MILP 工程说明与实时操作指南

## 1. 工程结构
- `src/milp_sim/config.py`：全局参数配置（地图、车辆、代价、协商、动态事件等）
- `src/milp_sim/entities.py`：核心实体定义 `Vehicle`、`Task`
- `src/milp_sim/map_utils.py`：地图对象与几何查询
- `src/milp_sim/obstacle_generator.py`：随机场景生成（不规则障碍、车辆、任务）
- `src/milp_sim/planner_astar.py`：高分辨率 A* 路径规划（含障碍膨胀）
- `src/milp_sim/cost_estimator.py`：快速代价估计器
- `src/milp_sim/verification.py`：高精度校对（`c_tilde` 与 `e_under`）
- `src/milp_sim/neighbor_coordination.py`：邻域同步与冲突消解日志
- `src/milp_sim/dynamic_events.py`：动态新增/取消事件构造
- `src/milp_sim/auction_core.py`：分配引擎（竞拍、校对、撤回再拍卖、取消处理）
- `src/milp_sim/session.py`：统一服务层（init/reset/status/add/cancel/logs/plot）
- `src/milp_sim/simulator.py`：批处理仿真流程编排
- `src/milp_sim/visualization.py`：可视化绘图（文件导出 + ax 复用）
- `src/milp_sim/interactive_console.py`：实时命令行控制台（基于 `SimulationSession`）
- `src/milp_sim/gui_app.py`：Tkinter + Matplotlib GUI
- `src/milp_sim/main.py`：程序入口（批处理 + CLI + GUI）
- `src/milp_sim/assignment_cost_experiment.py`：前缀路程代价对比分配实验（纯 A*，输出 PNG + JSON）
- `src/milp_sim/path_quality_metrics.py`：平滑路径质量评估工具（单段路径指标 + 批量汇总）
- `docs/path_quality_evaluation.md`：平滑路径规划评估标准、阈值、场景与接手说明

## 2. 主流程说明
### 2.1 快速竞拍
1. 车辆基于当前序列末端对可行任务计算 `c_hat`
2. 每车仅提交最优任务出价
3. 同任务按低出价中标（平局按车辆 ID）
4. 任务进入 `tentative`

### 2.2 邻域协商
每个任务维护 `(winner, bid, status, version)`，车辆只与通信半径内邻居同步。
冲突消解顺序：
1. 状态优先级
2. version 更大
3. bid 更小
4. vehicle id 更小

### 2.3 高精度校对与撤回
1. 对 `tentative` 任务执行 A* 真实路径校对
2. 计算 `c_tilde` 和 `e_under`
3. 若 `e_under <= epsilon`：锁定任务（`locked`）
4. 若失败：任务 `withdrawn`，仅更新失败车对该任务代价 `c_hat <- c_tilde`，进入再拍卖
5. 若同车在更新后再次中标，则直接接受，不再因同一原因再次撤回

### 2.4 动态事件处理
- `add_task`：新增任务后触发局部再分配
- `cancel_task`：
  - 未锁定：直接标记 `canceled`
  - 已锁定未执行：从所属车辆任务序列移除并恢复容量，然后局部再分配

## 3. 如何运行
### 3.1 批处理（默认 Round 2 流程）
```bash
./run.sh
# 或
PYTHONPATH=src python -m milp_sim.main
```

### 3.2 交互式控制台（实时增删任务）
```bash
./run_console.sh
# 或
PYTHONPATH=src python -m milp_sim.main --interactive
```

### 3.3 GUI（Tkinter + Matplotlib）
```bash
./run_gui.sh
# 或
PYTHONPATH=src python -m milp_sim.main --gui
```

说明：
- 离线模式的主要操作都在 GUI 内完成，不走命令行交互。
- 在离线 GUI 中新增/取消任务、移动任务、修改障碍后，需要点击 `Re-auction Now` 才会重新计算分配结果。
- 若界面状态里出现 `offline_reallocation_pending=True`，说明当前展示的不是最终可评估结果。
- 离线 GUI 可通过 `Export Path Quality` 导出当前最终结果的路径质量评估。

### 3.3.1 离线校对差异增强场景（5 车 50 任务）
新增场景文件：`examples/scenario_offline_5v50_corridor.json`

特点：
- 5 台车辆
- 50 个任务点
- 9 个蛇形通道障碍（更容易放大快速估计与高精度校对差异）

运行方式：

```bash
./run_offline_gui.sh --scenario-file examples/scenario_offline_5v50_corridor.json
# 或
PYTHONPATH=src python -m milp_sim.main --gui --scenario-file examples/scenario_offline_5v50_corridor.json
```

### 3.3.2 在线 GUI 读取场景文件
在线模式现在同样支持通过场景文件启动（脚本默认读取 `examples/scenario_verification_demo.json`）：

```bash
./run_online_gui.sh --scenario-file examples/scenario_offline_5v50_corridor.json
# 或
PYTHONPATH=src python -m milp_sim.main --gui-online --scenario-file examples/scenario_offline_5v50_corridor.json
```

### 3.4 前缀路程代价对比分配实验
该实验固定一个手工场景，对比两种拍卖代价：
- `incremental_only`：只看当前末端到候选任务的 A* 路径代价
- `prefix_aware`：看已分配前缀累计路程 + 当前末端到候选任务的 A* 路径代价

当前固定场景包含：
- 3 台智能体
- 9 个任务点
- 4 个矩形障碍物
- 纯 A* 路径规划，不使用 Dubins，不依赖在线重分配

运行方式：

```bash
PYTHONPATH=src python -m milp_sim.assignment_cost_experiment
```

Windows PowerShell 示例：

```powershell
$env:PYTHONPATH = "src"
python -m milp_sim.assignment_cost_experiment
```

随机种子模式：
- 使用 `--seed N` 后，会生成基于该种子的随机复杂场景
- 随机场景默认包含 4 台智能体、12 个任务点、6 个随机障碍物
- 输出文件名会自动带上 `_seed_N` 后缀，便于保留多组结果

```bash
PYTHONPATH=src python -m milp_sim.assignment_cost_experiment --seed 7
PYTHONPATH=src python -m milp_sim.assignment_cost_experiment --seed 11
```

```powershell
$env:PYTHONPATH = "src"
python -m milp_sim.assignment_cost_experiment --seed 7
```

运行后会：
- 在终端打印两种模式下的任务分配、每车路程、系统总路程
- 在终端和 JSON 中额外输出执行时间指标（按最晚完成 agent 的完成时间统计，同时给出该 agent 的路径长度）
- 生成 `outputs/assignment_cost_comparison.png`
- 生成 `outputs/assignment_cost_comparison_summary.json`
- 带种子运行时生成 `outputs/assignment_cost_comparison_seed_<seed>.png`
- 带种子运行时生成 `outputs/assignment_cost_comparison_summary_seed_<seed>.json`

当前固定场景的实验结果应表现为：
- `incremental_only`：`V0 -> [T1, T3, T8, T7]`，`V1 -> [T5]`，`V2 -> [T0, T2, T4, T6]`
- `prefix_aware`：`V0 -> [T1, T3, T8, T7]`，`V1 -> [T2, T6]`，`V2 -> [T0, T4, T5]`

## 4. 交互式控制台命令
启动后输入 `help` 可查看命令。

- `status`
  - 查看当前车辆任务序列、剩余容量、系统总代价、任务状态统计
- `tasks [status]`
  - 列出全部任务，或按状态过滤，如 `tasks locked`
- `add x y demand [task_id]`
  - 在指定坐标新增任务（自动执行再分配）
  - 示例：`add 60 35 2`
- `add_random [demand]`
  - 随机新增一个合法任务并再分配
  - 示例：`add_random`、`add_random 3`
- `cancel task_id`
  - 取消指定任务并再分配
  - 示例：`cancel 12`
- `reset`
  - 以当前 seed 重建场景
- `undo`
  - 撤销上一步会改变状态的操作（加任务/取消任务/加障碍等）
- `plot [filename]`
  - 将当前状态保存到 `outputs/`
- `export_logs [prefix]`
  - 导出校对/协商日志到 `outputs/`
- `export_task_ops [filename]`
  - 导出任务点操作（新增/删除/移动）的可回放 JSON 到 `outputs/`
- `replay_task_ops <json_file>`
  - 读取 JSON，重置当前会话并按记录帧/时间回放任务点操作
- `logs [n]`
  - 打印最近 `n` 条校对/协商日志
- `quit` / `exit`

## 5. GUI 功能
- 左侧控制区：初始化/重置、Undo、保存快照、导出日志
- 左侧控制区支持路径质量导出：
  - `Export Path Quality`：导出当前离线最终结果的路径质量评估
  - 若处于 `offline_reallocation_pending=True`，会拒绝导出并提示先点击 `Re-auction Now`
- 左侧控制区支持任务操作脚本：
  - `Export Task Ops JSON`：导出本次会话中任务点新增/删除/移动操作（含 frame_idx/sim_time）
  - `Replay Task Ops JSON`：读取 JSON 并重置后回放，用于复现实验操作
- 新增任务（点击地图）：
  - 在 Add Task 区输入 `demand`、`task_id(可选)`
  - 点 `Start/Stop Click Add` 开启点击模式
  - 在地图上点击即添加任务并自动再分配
- 新增任务（输入坐标）：
  - 在 Add Task 区输入 `x`、`y`、`demand`（`task_id` 可选）
  - 点击 `Add At Coord`（在线模式按钮为 `Add At Coord + Replan`）
- 随机任务：可选 demand
- 取消任务：输入 task_id
- 多边形障碍绘制：
  - 点击 `Start/Stop Draw` 进入绘制模式
  - 在地图上逐点点击添加顶点（至少 3 点）
  - 可用 `Undo Point` / `Clear Points`
  - 点击 `Apply Obstacle` 应用障碍并重建 A* 规划器
- 中间地图：障碍、depot、车辆轨迹、任务状态着色
- 右侧状态区：车辆序列、容量、总时间、任务状态统计
- 右侧日志区：最近校对日志和协商日志
- 右侧任务区：任务列表摘要
- 统一错误弹窗：非法输入、越界、障碍冲突、重复 task_id 等

说明：障碍应用后会立即影响路径规划显示；若新障碍导致当前任务点或车辆点不可达/被覆盖，会拒绝应用并提示错误。
说明：支持 Undo（最多保留最近 30 步历史），可撤销加任务、随机加任务、取消任务、应用障碍。

## 6. 输出文件
- `outputs/03_round2_final_result.png`：Round 2 批处理最终图
- `outputs/coordination_log.txt`：批处理协商日志
- `outputs/verification_log.txt`：批处理校对日志
- `outputs/assignment_cost_comparison.png`：前缀路程代价对比分配实验结果图
- `outputs/assignment_cost_comparison_summary.json`：前缀路程代价对比实验摘要
- `outputs/assignment_cost_comparison_seed_*.png`：带随机种子的对比实验结果图
- `outputs/assignment_cost_comparison_summary_seed_*.json`：带随机种子的对比实验摘要
- `outputs/snapshot_*.png`：Session/GUI/CLI 快照
- `outputs/session_*_coordination_log.txt`：Session/GUI/CLI 协商日志导出
- `outputs/session_*_verification_log.txt`：Session/GUI/CLI 校对日志导出
- `outputs/path_quality/path_quality_*_per_vehicle.jsonl`：离线 GUI 路径质量逐车结果
- `outputs/path_quality/path_quality_*_summary.json`：离线 GUI 路径质量汇总结果

## 7. 参数调优建议
在 `src/milp_sim/config.py`：
- `verify_epsilon`：越小越容易触发撤回再拍卖
- `comm_radius`：通信半径，影响协商收敛
- `sync_stable_h`、`sync_rmax`：协商稳定轮数和最大轮次
- `dynamic_new_tasks`、`dynamic_cancel_tasks`：批处理模式动态事件数量
- `astar_resolution`：A* 分辨率（越小越精细，但更慢）
