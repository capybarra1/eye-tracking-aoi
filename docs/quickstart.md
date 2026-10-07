# 眼动 AOI 切片标注工具

本地动态区域跟踪、人工关键帧修复、切片范围调整与导出工具。公开代码来自最近使用的标注工具副本，不包含参与者资料、真实录像、眼动或标注存档。

![合成场景下的实际标注界面](../assets/aoi-annotation.jpg)

## 合成演示启动

在仓库根目录运行，推荐 Python 3.12：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python demo.py
```

打开 http://127.0.0.1:18809/?view=simple 。程序在 `outputs/demo/` 生成六秒合成视频、十二切片和独立演示工作区；不读取原研究文件。已有演示进度保留。

1. 选择切片，在画面拖动区域或直线手柄。
2. 点击继续，观察当前切片的跟踪；随时暂停。
3. 调整切片起止，或设置首尾关键帧并生成预览。
4. 保存后导出标注存档或数据表，再复核。

直线模式的下方是平板近似区域，会包含背景。演示没有真实眼动数据，眼动指标计算需要自行提供匹配的数据与时间轴。

## 文件

- `scripts/supervised_aoi_server.py`：本地服务与界面操作。
- `scripts/supervised_aoi.py`：会话、逐帧记录、状态与关键帧。
- `scripts/aoi_horizontal.py`：直线分区跟踪。
- `scripts/aoi_range_review.py`：范围修复与备份。
- `scripts/aoi_slice_timing.py`：切片时间调整。
- `scripts/aoi_projects.py`：本地项目与标注包校验。
- `scripts/compute_supervised_aoi_metrics.py`：眼动指标计算入口，需要独立研究数据。

## 选定检查

```sh
.venv/bin/python -m unittest tests.test_supervised_aoi tests.test_supervised_preview tests.test_aoi_straight_partition tests.test_aoi_slice_timing tests.test_aoi_line_cache tests.test_aoi_manual_keyframes tests.test_aoi_scrub_preview tests.test_aoi_scene_anchor -q
```

## 范围

CPU/OpenCV 核心流程可运行；可选学习模型恢复依赖额外模型和配置，本包默认关闭且不带权重。缓存也是默认关闭。项目主要在 Mac 上使用，预处理包含 POSIX 文件锁，未声称兼容所有系统。标注完成仍需人工验收。

[完整案例](../README.md)
