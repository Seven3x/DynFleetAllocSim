# MILP Task Allocation Progress

## Round 1 Scope (Current)
- [x] 场景生成（地图边界、depot、不规则障碍、车辆与任务随机初始化）
- [x] A* 路径规划（高分辨率栅格 + 障碍膨胀）
- [x] 快速代价估计（距离 + 航向修正 + 走廊障碍密度）
- [x] 静态任务分配（按轮竞拍、冲突消解、任务序列生成）
- [x] 可视化（初始场景图 + 最终轨迹与任务序列图）

## Round 2 Scope (Pending)
- [x] 动态新增任务
- [x] 任务取消
- [x] 校对失败后撤回与再拍卖
- [x] 邻域协商日志

## Main Entry
- `PYTHONPATH=src python -m milp_sim.main`
- `./run.sh`
- `./run_console.sh` （交互式实时增删任务）
- `./run_gui.sh` （Tkinter + Matplotlib GUI）

## Output
- `outputs/01_initial_scene.png`
- `outputs/03_round2_final_result.png`
- `outputs/coordination_log.txt`
- `outputs/verification_log.txt`
- `outputs/snapshot_*.png`
- `outputs/session_*_coordination_log.txt`
- `outputs/session_*_verification_log.txt`

## Docs
- `ENGINE_GUIDE.md`（工程说明 + CLI/GUI 实时操作指南）

## GUI Extras
- [x] GUI 支持点击地图绘制多边形障碍并应用到实时规划器
- [x] GUI 支持点击地图新增任务（任务点击模式）
- [x] 支持 Undo（撤销最近一步状态变更）
