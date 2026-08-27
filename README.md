# QoPS 仿真数据准备与预测实验

本项目用于研究 **Quality of Product-Service Prediction**：在传统 WS-DREAM QoS 预测基础上，为服务补充其承载实体的设备特征，验证这些信息能否提升产品服务网络指标 **响应时间（RT）** 和 **吞吐量（TP）** 的预测精度。

## 研究边界

- 场景暂定为一个 `120m × 80m` 的单层大型养老医院。
- WS-DREAM 原始用户、服务特征及 RT/TP 文件保持不变。
- 每个服务固定绑定一个实体，一个实体可以搭载多个服务。
- 第一版包含192个静态实体和64个动态实体。
- 第一版只使用实体类型、处理能力、网络接口和设备负载进行预测。
- 第一版没有随机噪声，只根据设备特征生成可复现的结构化扰动。
- 当前数据属于探索性半合成数据，参数尚不能解释为真实医院设备规律。

## 数据结构

生成器会建立以下数据表：

- `static_entities.csv`：静态实体、服务半径、调用成本、位置及性能属性；
- `dynamic_entities.csv`：动态实体、速度、电量、调用成本、位置及性能属性；
- `service_entity_mapping.csv`：服务与承载实体的固定映射；
- `user_hospital_locations.csv`：用户在医院平面中的仿真坐标；
- `service_entity_features.csv`：训练时按服务索引对齐的实体特征；
- `pilot_calls_audit.csv`：小样本调用及扰动审计信息，不直接作为模型输入。

完整仿真矩阵生成在 `data/simulated/full/`，该目录可由配置重建，不进入Git。

## 标签生成逻辑

三个核心属性为：

- `processing_capacity`：设备处理能力；
- `network_interface_type`：Ethernet、Wi-Fi 6、Wi-Fi 5或Cellular；
- `device_load`：设备当前综合负载。

先生成RT、TP惩罚分数：

```text
Z_RT = 0.4 × (1-capacity) + 0.4 × load + 0.2 × interface_rt
Z_TP = 0.3 × (1-capacity) + 0.3 × load + 0.4 × interface_tp
```

再以中等强度 `λ=0.30` 生成新标签：

```text
RT_PS = RT_WS × exp(0.30 × Z_RT)
TP_PS = TP_WS × exp(-0.30 × Z_TP)
```

因此设备状态较差时RT增大、TP下降，且所有标签保持正数。第一版不加入随机噪声。

## 环境

推荐使用 Python 3.10：

```bash
python -m pip install -r requirements.txt
```

运行前设置项目源码路径：

```bash
export PYTHONPATH="$PWD/src:$PWD"
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
```

## 生成数据

生成pilot和完整数据：

```bash
python -m qops_preparation.simulation --mode all
```

只生成小规模测试数据：

```bash
python -m qops_preparation.simulation --mode pilot
```

## 自动检查

```bash
python -m unittest tests.test_simulation -v
```

检查内容包括实体数量、映射完整性、可复现性、扰动方向、缺失值保持以及 `λ=0` 恒等性。

## 对照实验

基础AFIL不读取实体特征：

```bash
python -m experiments.run_comparison --model baseline --type rt --md 0.10 --seed 0 --epochs 50 --cuda 0
```

实体增强AFIL将实体类别、网络接口、处理能力和负载编码后附加到服务侧：

```bash
python -m experiments.run_comparison --model entity_concat --type rt --md 0.10 --seed 0 --epochs 50 --cuda 0
```

将 `--type rt` 改成 `--type tp` 即可预测吞吐量。两个模型必须使用相同密度、种子和训练轮数。

确认冒烟测试通过后，可在服务器前台串行运行四项首轮对照：

```bash
bash scripts/run_first_comparison.sh 0 50
```

第一个参数是执行前通过 `nvidia-smi` 选定的物理GPU编号，第二个参数是训练轮数。脚本不使用后台任务。

## 第一轮实验安排

1. 对RT和TP分别运行1 epoch冒烟测试；
2. 使用10%密度、seed 0、50 epoch运行RT基线与实体增强模型；
3. 以相同设置运行TP基线与实体增强模型；
4. 比较MAE和RMSE；
5. 只有确认直接融合有效后，再考虑静态/动态双分支、门控融合、FiLM、交叉网络或多任务学习。

## 注意事项

- 不要修改原始 WS-DREAM 文件；
- 不要将 `rt_penalty`、`tp_penalty` 或 multiplier 作为模型输入；
- 单次物理调用成本、服务半径、速度、电量和位置在第一版中只保存，不参与标签生成；
- 当前实验只能证明模型能够学习预设的结构化实体扰动，不能直接证明真实设备中的因果关系。

## 第二轮：实体信号检查与门控融合

先验证实体属性能否预测仿真标签中注入的扰动：

```bash
python tools/check_entity_predictability.py
```

残差门控模型保持基础AFIL主干不变，通过可学习门值控制实体信息的注入强度：

```text
h_fused = h_afil + sigmoid(Gate([h_afil, h_entity])) × h_entity
```

运行单项实验：

```bash
python -m experiments.run_comparison --model entity_gated --type rt --md 0.10 --seed 0 --epochs 50 --cuda 0
```

在已有baseline和entity_concat五种子结果的基础上，串行补跑门控模型：

```bash
bash scripts/run_gated_validation.sh 0 50
```
