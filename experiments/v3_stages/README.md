# QoPS v3：逐调用 AFIL 与三阶段完整预测

本入口适配冻结的第三版数据，分别训练网络、就绪和执行模型，按验证集选择方法，冻结后评价标准测试及泛化测试。原有矩阵式 AFIL 代码仍保留在 `experiments/run_comparison.py`。

## 方法分工

- 网络：AFIL 基础上下文、实体直接融合、实体门控；各自包含无身份、有身份两个版本。主网络方法在四种增强AFIL候选中按验证选择。另有完整上下文 MLP、梯度提升树和固定机理对照，全部报告结果。
- 就绪：固定机理、带机理特征的 MLP、机理加 MLP 残差、带机理特征的树；原始摘要 MLP 单列比较。
- 执行：基础执行时间除以物理能力，另训练 MLP 作近似对照。
- 总时延：选定的三个阶段逐调用相加；另报告直接总量 MLP 和三个独立 MLP 求和。

机理输入来自 t₀ 已知地图、队列、活动状态及配置。新增网络实体字段是数字能力、带宽上限、数字队列与活动观测。原始数据、仿真规则和标签没有修改。

## 目录布局

实验目录包含 `code/`、`data/`、`cache/`、`results/`、`evaluation/`、`reports/`。`data_source_hashes.json` 保存冻结源文件哈希。源数据保留各世界配置、运行索引、输入、标签和已知队列快照；字段与路径参见 `prepare.py`。

训练入口只打开 `train_x/y.pkl` 和 `validation_x/y.pkl`。测试数据由 `evaluate.py` 在选型完成、哈希冻结之后打开。缓存准备阶段会检查各分区标签和公式一致性，属于数据验收，不参与超参数选择。

## 安装

使用 Python 3.10 的独立环境。先安装官方 CUDA 12.8 版 PyTorch，再安装 `requirements.txt`。`environment_requirements.txt` 是服务器完整环境记录，含 Ubuntu 系统包；跨机器复现使用最小依赖清单。

```bash
python -m venv env
env/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
env/bin/python -m pip install -r code/requirements.txt
```

## 运行

服务器现用目录为 `/data/qops_lxy_20261118`。指定其他实验目录时设置 `QOPS_EXPERIMENT_ROOT`；目录中的代码来自本文件夹。

```bash
env/bin/python code/prepare.py --root data --out cache
export QOPS_EXPERIMENT_ROOT=/data/qops_lxy_20261118
bash code/run_server.sh 0
```

此部署入口只允许 RTX5880，拒绝物理 GPU2/6000D，并检查启动时没有已有计算进程。当前每个选定GPU同时至多两个独立训练进程，每进程CPU线程数为4，单进程显存上限为该卡总量的15%。两个进程运行不同候选，不共享模型或训练状态。原始数据、旧实验、其他用户的进程均不改动。

固定搜索：每个神经网络家族四组超参数、三个随机种子；树模型四组配置。共152个候选训练，随后六个能力/状态消融。每次训练最多100轮，连续12轮验证MAE不改善即提前停止。已成功写入 `result.json` 的任务会跳过，从而支持中断后继续。失败任务保留日志。

`run_server.sh` 依次执行验收、候选训练、验证选型、消融、冻结评价和独立复算。正常结束后运行：

```bash
env/bin/python code/report.py --root .
# 绘图额外需要 matplotlib；服务器已有个人环境可用于只读绘图。
python code/plot_results.py --root .
# 主流程结束且GPU0空闲后，检验独立推理、选型恢复与固定种子重新训练。
env/bin/python code/reproduce.py --root . --gpu 0
```

## 只做推理

```bash
env/bin/python code/predict.py --root . --inputs cache/test_x.pkl --out predictions.csv --seed 11
```

该命令只读取准备好的 t₀ 快照和模型，使用 CPU，输出网络、就绪、执行及总完成时延，单位为秒。若选中机理或残差方案，输入需要包含准备阶段由已知队列和地图计算的机理摘要。它不打开标签表。

`reproduce.py` 把CPU推理结果与保存的GPU评价逐条核对，再用最终网络模型的种子11重新训练。复现训练独立保存在 `reproducibility/`，不替换模型、不参与选型、不重新调参。脚本还在临时目录重跑选型，检查消融结果存在时选择仍保持一致。

## 实时查看

```bash
env/bin/python code/monitor.py --root .
```

每三秒显示当前任务、轮次、验证误差、完成数量和 GPU0 占用。新训练日志同时记录训练误差；前期已经结束的少数任务在逐轮历史中保存训练误差。关闭监控不会结束训练。

## 结果解释

- `result.json` 保存配置、字段清单、参数量、最佳轮次、环境和验证指标；`history.json` 保存逐轮训练过程。
- `selection.json` 保存所有家族的验证最优配置及最终三阶段选型。
- `evaluation/metrics.csv` 保存按调用加权、静态/动态、设备等权指标。每个神经网络的三个训练种子分别保存，汇总使用种子均值和标准差。
- `evaluation/context_slices.csv` 区分数字资源空闲/忙碌、物理队列空/非空条件。
- `evaluation_freeze.json` 冻结代码、数据清单、缓存和检查点；`independent_verification.json` 从逐调用预测复算指标并核验哈希。
- `network_paired_comparison.json` 按完整运行做配对重采样。参考AFIL依据验证选择，在所有测试条件下固定使用同一个参考版本。

基础AFIL与实体AFIL比较输入扩展的效果；直接融合与门控比较融合方式。当前实现是适配QoPS连续上下文的AFIL骨干，数值不能当作原论文在其原数据集上的复现结果。确定性执行公式与理想化t₀观测属于第三版的既定条件；论文结论按这些条件解释。
