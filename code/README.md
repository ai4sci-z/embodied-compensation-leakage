# 跨本体模仿学习中的示教补偿泄漏研究

本项目使用双相机图像与本体历史预测机械臂动作，研究控制延迟、机械回差和重力下垂补偿在多来源示教学习中的传递。

## 代码结构

- `portal_trace/policies/portal_policy.py`：策略网络。
- `portal_trace/policies/encoders.py`、`decoder.py`：视觉与状态编码、Transformer 动作解码。
- `course_tools/train_three_source_pool.py`：训练及模型加载。
- `course_tools/three_source_data.py`：数据采样、标签处理与归一化。
- `course_tools/evaluate_common_probes.py`：共同输入上的补偿测量与标签配对分析。
- `course_tools/evaluate_three_source.py`：不同目标身体上的自主执行。
- `configs/`：33 份训练配置、数据划分、归一化统计及评估设置。
- `results/`：实验结果表；`analysis/`：论文图表绘制脚本。

## 环境

训练与仿真使用 Linux、Python 3.12、PyTorch、MuJoCo 3.3.2 和 robosuite 1.5.2。依赖版本见 `environment/pro6000-pip-freeze.txt`，安装清单与兼容设置见 `environment/`；无显示环境使用 OSMesa 或 EGL。

仅重绘结果图时，在本目录执行：

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python analysis/build_formal_figures.py
python analysis/build_extra_figures.py
```

图像输出到 `results/figures/`。

## 模型权重

从 [Release](https://github.com/ai4sci-z/embodied-compensation-leakage/releases/tag/v1.0) 下载权重，按下方示例保存到 `../weights/`。共 33 个最终模型，每个权重含模型参数、网络配置、归一化统计和训练记录；`configs/checkpoints.json` 给出原始文件校验值。

加载单个策略：

```python
from course_tools.train_three_source_pool import load_three_source_checkpoint
policy = load_three_source_checkpoint(
    "../weights/L1-balanced-raw-seed-2026111801/policy-step-30000.pt",
    device="cpu",
)
```

在安装完整仿真依赖后，可运行一个正常目标场景：

```sh
python -m course_tools.evaluate_three_source   --checkpoint ../weights/L1-balanced-raw-seed-2026111801/policy-step-30000.pt   --rollout --device cpu --target normal --level 1 --seed-env 141294887   --render-backend osmesa --output course_results/example
```

目标代码 `normal`、`D2`、`D4hi`、`D3b` 分别对应正常、延迟、回差、下垂；`level` 为强度档位。全部模型、场景和配对设置见 `configs/evaluation/protocol.json`。

## 数据

结果 CSV 可直接用于重绘和统计。训练所用 HDF5 示教与完整轨迹另行保存，数据索引见 `configs/data/manifest.json`。运行训练或从示教重新构造共同测试输入时，需要将索引中的路径映射到数据所在目录。
