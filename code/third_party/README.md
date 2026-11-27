# 第三方代码

`himoe/` 包含 HiMoE-VLA 的动作模块，来源：

- 项目：https://github.com/ZhiyingDu/HiMoE-VLA
- 版本：`27a2c46932d8b6373ca0074eb997f299bcd4f6f5`
- 论文：HiMoE-VLA，arXiv:2512.05693
- 许可证：MIT，见 `himoe/LICENSE`

上游目录结构与代码保持原样，适配实现位于 `portal_trace/policies/himoe_adapter.py`。本课程实验的共享策略使用 ResNet18 编码器与 Transformer 动作解码器，配置见 `configs/training/`。
