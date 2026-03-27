# 平滑路径规划评估方案

本文档用于沉淀离线模式和在线模式的 Dubins 平滑路径评估标准，目标是让后续 agent 可以基于统一指标持续迭代，而不是依赖主观观察。

## 0. 推荐环境

统一使用 `conda` 的 `milp` 环境执行 GUI、评估导出和相关测试。

推荐命令：

```bash
conda run -n milp env PYTHONPATH=src python -m milp_sim.main --gui
```

已验证事项：

- `conda run -n milp env PYTHONPATH=src python -m unittest src.tests.test_path_quality_metrics`
- `conda run -n milp env PYTHONPATH=src python -c "from milp_sim.gui_app import OfflineGuiApp; ..."`

原因：

- 当前默认 `delivery` 环境缺少 `shapely`
- `milp` 环境已确认具备 `shapely` 和 `tkinter`

## 1. 适用范围

- 离线模式：任务分配完成后，每段执行路径的平滑性、可行性和效率评估。
- 在线模式：周期性重规划、多任务切换、动态事件插入后的路径平滑性和稳定性评估。
- 当前代码入口：
  - `src/milp_sim/dubins_path.py`
  - `src/milp_sim/path_postprocess.py`
  - `src/milp_sim/verification.py`
  - `src/milp_sim/session.py`
  - `src/milp_sim/config.py`

## 2. 评估目标

路径“平滑”不能只看视觉效果，必须同时满足以下四类目标：

1. 可行性：无碰撞、不过界、满足最小转弯半径。
2. 几何平滑性：航向连续，曲率受控，曲率跳变低。
3. 执行效率：不过度绕路，不以大幅增程换取局部平滑。
4. 在线稳定性：重规划后前缀尽量稳定，不出现明显抖动或左右摆。

## 3. 单段路径指标

单段路径指一次 `start_pose -> target_pose/task` 的规划结果。建议统一重采样后评估，默认采样步长取 `0.25m ~ 0.5m`。

### 3.1 硬约束指标

- `collision_free`
  - 路径是否与障碍物或边界冲突。
- `min_clearance`
  - 路径到障碍物/边界的最小净空。
- `p05_clearance`
  - 采样点净空的 5 分位数，用于识别长距离贴障。
- `curvature_violation_ratio`
  - 曲率超过 `1 / turn_radius` 的采样点占比。

任一硬约束失效，都应将该路径标记为失败样本。

### 3.2 几何平滑指标

- `max_initial_turn_delta_rad`
  - 起始航向与路径第一段方向的夹角。
- `heading_jump_p95_rad`
  - 相邻采样段航向变化绝对值的 95 分位数。
- `curvature_abs_mean`
  - 绝对曲率均值。
- `curvature_abs_p95`
  - 绝对曲率 95 分位数。
- `curvature_jump_p95`
  - 相邻曲率变化绝对值的 95 分位数。

离散定义：

```text
ds_i = ||p_{i+1} - p_i||
psi_i = atan2(y_{i+1} - y_i, x_{i+1} - x_i)
kappa_i = wrap(psi_{i+1} - psi_i) / ((ds_i + ds_{i+1}) / 2)
dkappa_i = |kappa_{i+1} - kappa_i|
```

### 3.3 执行效率指标

- `path_length`
- `astar_length`
- `straight_distance`
- `length_ratio = path_length / astar_length`
- `euclid_ratio = path_length / straight_distance`
- `sample_count`

解释：

- `length_ratio` 用于比较“平滑路径”相对现有 A* 参考路径是否显著增程。
- `euclid_ratio` 用于跨场景比较整体绕行程度。

### 3.4 规划器行为指标

- `dubins_ratio`
  - Dubins 段长度占比。
- `fallback_used`
  - 是否触发 A* 或其他兜底方案。
- `fallback_reason`
  - 回退原因。

## 4. 单次实验聚合指标

一个实验指一次离线完整求解，或一次在线完整仿真。聚合时建议同时输出均值和尾部分布。

- `success_rate`
- `fallback_rate`
- `mean_length_ratio`
- `p95_length_ratio`
- `mean_min_clearance`
- `p05_min_clearance`
- `mean_curvature_violation_ratio`
- `p95_curvature_violation_ratio`
- `mean_curvature_jump_p95`
- `mean_max_initial_turn_delta_rad`

在线模式额外输出：

- `online_completion_time`
- `online_extra_distance_ratio`
- `replan_count`
- `replan_path_delta_mean`
- `prefix_preservation_ratio_mean`
- `heading_flip_count`
- `task_oscillation_count`

## 5. 在线稳定性定义

### 5.1 前缀保留率

`prefix_preservation_ratio` 用于度量相邻两次重规划时，未来短时前缀是否保持稳定。

建议定义：

- 比较相邻两次规划结果在未来 `T=3s` 或前 `L=10m` 范围内的公共前缀长度。
- 比值越高，表示重规划后路径更稳定。

### 5.2 重规划路径差异

`replan_path_delta` 建议定义为：

- 对相邻两次规划结果统一重采样。
- 在未来 `T=3s` 或 `L=10m` 范围内逐点计算欧氏距离。
- 取这些距离的均值。

### 5.3 抖动计数

`heading_flip_count`：

- 在固定时间窗内，若航向变化符号频繁反复切换，视为抖动。
- 该值应尽可能低。

`task_oscillation_count`：

- 同一车辆在短时间内频繁切换当前目标任务的次数。

## 6. 基准场景集

建议固定一组 benchmark 场景，避免只在单一地图上过拟合。

- `open_sparse`
  - 开阔稀疏障碍，重点检查是否无意义绕路。
- `dense_obstacles`
  - 中高密障碍，重点检查贴障和平滑绕障质量。
- `narrow_passage`
  - 狭窄通道，重点检查曲率约束下的可通过性。
- `near_goal_high_heading_mismatch`
  - 目标近但末端航向差大，重点检查局部绕圈。
- `multi_task_chain`
  - 多任务串联，重点检查任务衔接处是否生硬。
- `online_dynamic`
  - 在线新增、取消、移动任务，重点检查重规划稳定性。

每类建议固定 `20~50` 个随机种子，单独保留一份“回归种子列表”。

## 7. 建议阈值

以下阈值不是最终真理，但足够作为第一版验收门槛：

- `collision_free_rate < 100%`：直接失败。
- `p05_min_clearance < safety_margin`：直接失败。
- `mean_curvature_violation_ratio > 1%`：直接失败。
- `fallback_rate > 15%`：标红。
- `p95_length_ratio > 1.35`：标红。
- `mean_max_initial_turn_delta_rad > pi/3`：标红。
- `replan_path_delta_mean > 1.5m`：在线模式标红。

## 8. 总评分

为了支持快速版本比较，定义总评分 `PQI`，范围建议为 `0~100`。

```text
PQI = 35 * Feasibility
    + 30 * Smoothness
    + 20 * Efficiency
    + 15 * OnlineStability
```

每个子项归一化到 `0~1`。

### 8.1 Feasibility

- 基于无碰撞率、净空表现、曲率超限率、fallback 惩罚。

### 8.2 Smoothness

- 基于初始转角、航向跳变、曲率大小、曲率跳变。

### 8.3 Efficiency

- 基于 `length_ratio`、`euclid_ratio`，在线模式可加入完成时间。

### 8.4 OnlineStability

- 基于 `replan_path_delta`、`prefix_preservation_ratio`、`heading_flip_count`、`task_oscillation_count`。

## 9. 输出数据规范

后续评估工具应至少输出两层数据：

1. `per_path.jsonl`
   - 每条路径一行，便于追溯异常样本。
2. `summary.json`
   - 场景级与全局聚合结果。

建议单条路径记录格式如下：

```json
{
  "mode": "offline",
  "scenario": "narrow_passage_seed_07",
  "vehicle_id": 2,
  "task_id": 11,
  "replan_index": 0,
  "path_length": 23.4,
  "astar_length": 19.8,
  "straight_distance": 17.2,
  "length_ratio": 1.182,
  "euclid_ratio": 1.360,
  "collision_free": true,
  "min_clearance": 1.05,
  "mean_clearance": 2.42,
  "p05_clearance": 1.21,
  "max_initial_turn_delta_rad": 0.91,
  "heading_jump_p95_rad": 0.18,
  "curvature_abs_mean": 0.19,
  "curvature_abs_p95": 0.43,
  "curvature_jump_p95": 0.27,
  "curvature_violation_ratio": 0.08,
  "dubins_ratio": 0.74,
  "fallback_used": false,
  "fallback_reason": ""
}
```

当前 GUI 首版导出是逐车级别：

- `*_per_vehicle.jsonl`
  - 每辆车一行，评估该车完整执行路线。
- `*_summary.json`
  - 对所有车辆结果做一次聚合。

这样做的原因是离线 GUI 当前稳定暴露的是每车最终 `route_points`，适合先形成一致、低风险的评估闭环。

## 10. 当前可复用工具

当前仓库已新增：

- `src/milp_sim/path_quality_metrics.py`
  - 单段路径评估与批量汇总工具。
- `src/tests/test_path_quality_metrics.py`
  - 基础指标测试，防止评估定义漂移。
- 离线 GUI 导出入口：
  - `Export Path Quality`
  - 输出到 `outputs/path_quality/`

## 11. GUI 操作约束

当前离线模式不是命令行交互，而是 Tk GUI。

- 离线 GUI 启动入口：`PYTHONPATH=src python -m milp_sim.main --gui`
- 在线 GUI 启动入口：`PYTHONPATH=src python -m milp_sim.main --gui-online`
- 若无图形显示环境，`run_offline_gui(...)` 会直接失败。

离线 GUI 中的关键行为：

- 新增任务、取消任务、移动任务、增删障碍后，离线分配不会自动重新求解。
- GUI 会进入 `offline_reallocation_pending=True` 状态。
- 必须点击左侧按钮 `Re-auction Now`，当前离线结果才会真正更新。
- 只有在 `Re-auction Now` 完成之后，`Export Path Quality` 才允许导出。

这点对评估很关键：

- 只有在 `pending` 清除之后导出的日志、截图、对比结果才可作为正式评估输入。
- 如果直接读取编辑后的界面状态而没有执行 `Re-auction Now`，得到的仍是旧分配结果或待更新状态。

## 12. 推荐实施顺序

1. 先把单条路径量化稳定下来。
2. 再做离线 benchmark 批跑。
3. 最后接入在线稳定性指标。

不要一开始同时改路径算法和评估口径，否则无法判断变化来自哪里。

## 13. 接手说明

如果其他 agent 接手，建议按下面顺序继续：

1. 阅读本文档。
2. 阅读 `src/milp_sim/path_quality_metrics.py` 的数据结构和评分函数。
3. 把路径生成结果接到 `evaluate_path_quality(...)`。
4. 将每次实验结果输出到 `outputs/path_quality/...`。
5. 只有在评估结果稳定后，再调整以下参数：
   - `dubins_sample_step`
   - `dubins_collision_margin`
   - `goal_heading_*`
   - `connector_*`
   - `online_max_initial_turn_rad`
   - `online_path_sample_step`
