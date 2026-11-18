"""Build a factual report exclusively from frozen evaluation artifacts."""
import argparse
import json
from pathlib import Path
import pandas as pd

LABELS = {
    'network_afil_base': 'AFIL：基础上下文',
    'network_afil_base_id': 'AFIL：基础上下文＋身份',
    'network_afil_full': 'AFIL：实体直接融合',
    'network_afil_full_id': 'AFIL：实体直接融合＋身份',
    'network_afil_gated': 'AFIL：实体门控融合',
    'network_afil_gated_id': 'AFIL：实体门控融合＋身份',
    'network_mlp_full': '网络 MLP：完整上下文',
    'network_tree_full': '网络梯度提升树：完整上下文',
    'ready_mlp_full': '就绪 MLP：摘要＋机理特征',
    'ready_residual_full': '就绪机理＋MLP 残差',
    'ready_mlp_raw': '就绪 MLP：原始摘要',
    'ready_tree_full': '就绪梯度提升树：摘要＋机理特征',
    'execution_mlp_raw': '执行 MLP',
    'total_mlp_raw': '直接预测总时延的 MLP',
    'mechanistic': '固定机理预测器',
    'selected_pipeline': '验证集选出的三阶段流程',
    'independent_mlp': '三个独立 MLP 求和',
}

def name(s):
    return LABELS.get(s, s)

def table(frame, split, stage):
    d = frame[(frame.split == split) & (frame.stage == stage) & (frame.group == 'all') & (frame.weighting == 'call')]
    lines = ['| 方法 | MAE，秒 | RMSE，秒 | P95绝对误差，秒 | 训练重复数 |',
             '|---|---:|---:|---:|---:|']
    for model, g in d.groupby('model', sort=True):
        sd = g.mae.std(ddof=1) if len(g) > 1 else 0
        lines.append(f'| {name(model)} | {g.mae.mean():.6f} ± {sd:.6f} | {g.rmse.mean():.6f} | {g.p95_absolute_error.mean():.6f} | {len(g)} |')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    metrics = pd.read_csv(root / 'evaluation/metrics.csv')
    selection = json.loads((root / 'selection.json').read_text())
    verification = json.loads((root / 'evaluation/independent_verification.json').read_text())
    paired = json.loads((root / 'evaluation/network_paired_comparison.json').read_text())
    prepared = json.loads((root / 'cache/preparation_report.json').read_text())
    search = pd.read_csv(root / 'validation_search.csv')
    directory = root / 'reports'
    directory.mkdir(exist_ok=True)
    standard = next(d for d in paired if d['split'] == 'test')
    net = selection['network']
    ready = selection['ready']
    lines = [
        '# QoPS 三阶段服务器实验结果',
        '',
        '## 完整预测流程',
        '',
        f"网络阶段（增强AFIL候选中按验证集选择）：{name('network_afil_' + net['variant'])}，嵌入维度 {net['width']}，学习率 {net['lr']}。",
        f"就绪阶段：验证集选择 `{ready['kind']}`，验证 MAE 为 {ready['mae']:.6f} 秒。",
        '执行阶段：`base_execution_time / processing_capacity`。逐调用将三个预测值相加，输出总完成时延。',
        '',
        f"标准测试上，增强网络模型 MAE 为 {standard['enhanced_mae']:.6f} 秒；较强基础 AFIL 对照为 {standard['base_mae']:.6f} 秒，相对变化为降低 {standard['relative_improvement']:.2%}。",
        f"按完整运行配对重采样的相对改善 95% 区间：[{standard['relative_improvement_ci95'][0]:.2%}, {standard['relative_improvement_ci95'][1]:.2%}]。",
        f"验证集预设的 10% 改善目标：{'达到' if selection['network_goal_10pct_met'] else '未达到'}；测试集独立报告实际变化。",
        '',
        '## 数据、选型与统计口径',
        '',
        '使用冻结的第三版正式数据；训练、验证和测试按完整运行隔离。完成记录用于时延回归，预热记录排除。网络、就绪和执行均使用请求发出时的可得信息。',
        '每个神经网络配置训练种子 11、22、33；按三种子平均验证 MAE 选择超参数，再冻结代码、配置、缓存及模型哈希后评价测试集。表中 MAE 为种子均值 ± 样本标准差，RMSE 与 P95 为各次训练指标的均值。树及确定性机理各计算一次。',
        'P95绝对误差是95%调用的绝对预测误差分位数。设备等权结果另保存在 metrics.csv；其中P95为各设备P95的平均，不能当作总体P95。相对改善的置信区间固定训练种子集合，按完整运行重采样1000次。',
        '网络主方法在实体直接融合、门控融合及其身份控制版本中选择；MLP、树和机理保持为对照，结果全部报告。基础AFIL保留特征交互骨干，并适配连续字段、缺失值和逐调用输入。原有用户/服务身份能力另设有身份对照；身份以世界及原编号组合，未见身份使用未知类别。这里比较的是AFIL在QoPS数据上的适配版本。',
        '',
        '| 数据部分 | 运行数 | 完成调用数 | 排除的记录数 | 机理回退次数 |',
        '|---|---:|---:|---:|---:|',
    ]
    runs = pd.DataFrame(prepared['runs'])
    for split, g in runs.groupby('split'):
        lines.append(f'| {split} | {len(g)} | {g.calls.sum()} | {g.excluded.sum()} | {g.fallbacks.sum()} |')
    labels = pd.read_pickle(root / 'cache/test_y.pkl')
    lines.extend(['', '## 标准测试真实时延量级', '', '| 阶段 | 均值，秒 | 中位数，秒 | P95时延，秒 |', '|---|---:|---:|---:|'])
    for key in labels.columns:
        values = labels[key]
        lines.append(f'| {key} | {values.mean():.6f} | {values.median():.6f} | {values.quantile(.95):.6f} |')
    lines.append('本表P95描述真实时延分布；下文P95绝对误差描述预测偏差，两者含义不同。')
    for stage, title in [('network_time', '网络'), ('ready_time', '就绪'), ('execution_time', '执行'), ('completion_time', '总完成')]:
        lines.extend(['', f'## 标准测试：{title}时延', '', table(metrics, 'test', stage)])
    lines.extend(['', '## 泛化与压力条件', '', '| 数据部分 | 基础AFIL MAE，秒 | 增强AFIL MAE，秒 | 相对改善 | 95%区间 |', '|---|---:|---:|---:|---|'])
    for d in paired:
        if d['split'] == 'validation':
            continue
        lo, hi = d['relative_improvement_ci95']
        lines.append(f"| {d['split']} | {d['base_mae']:.6f} | {d['enhanced_mae']:.6f} | {d['relative_improvement']:.2%} | [{lo:.2%}, {hi:.2%}] |")
    contexts = pd.read_csv(root / 'evaluation/context_slices.csv')
    context = contexts[(contexts.split == 'test') & (contexts.model.isin([standard['baseline'], standard['enhanced']])) & (contexts.stage == 'network_time')]
    lines.extend(['', '## 网络误差与数字资源状态', '', '| 方法 | t₀数字资源是否忙碌 | 调用数 | MAE，秒 | P95绝对误差，秒 |', '|---|---|---:|---:|---:|'])
    for (model, value), g in context.groupby(['model','value']):
        lines.append(f'| {name(model)} | {"忙碌" if value else "空闲"} | {int(g.n.iloc[0])} | {g.mae.mean():.6f} | {g.p95_absolute_error.mean():.6f} |')
    tails = pd.read_csv(root / 'evaluation/tail_risk.csv')
    tail = tails[(tails.split=='test') & (tails.model=='selected_pipeline')]
    lines.extend(['', '## 少量大误差的单独检查', '', '| 阶段 | P99绝对误差，秒 | P99.9绝对误差，秒 | 三次训练中最大误差，秒 | 误差超过60秒的比例 |', '|---|---:|---:|---:|---:|'])
    for stage, g in tail.groupby('stage'):
        lines.append(f'| {stage} | {g.p99_absolute_error.mean():.6f} | {g.p999_absolute_error.mean():.6f} | {g.max_absolute_error.max():.6f} | {g.fraction_over_60s.mean():.4%} |')
    lines.append('验证集追踪发现：物理队列快照为空时，数字队列中仍可能有尚未转入物理队列的用户请求。例如 L1_D3_high_standard_R04 的调用2396，在t₀时物理等待数为0；调用2394随后先获得确认并进入物理队列，其旋转和执行额外占用约141.105秒。现有输入保留数字队列长度，未保留其中用户请求的顺序与目标。该信息边界会造成少量较大就绪误差；本轮保持数据与输入规则不变，完整公布尾部误差。相关事件仅用于事后诊断。')
    lines.extend(['', '## 实验解释与论文使用', '',
                  '1. 基础AFIL已使用端点AP、可用带宽等网络上下文；新增实体信息是设备数字能力及数字队列/活动状态。移动、电量等物理状态没有被设定为网络惩罚项。基础AFIL与实体AFIL的差异检验这些新增信息的价值；直接融合与门控融合在相同字段下的比较才用于判断融合结构的作用。两种结果分别解释。',
                  '2. 机理预测器使用已知地图、完整已知队列和配置。就绪MLP、树和残差模型的完整输入包含同一机理结果；原始摘要MLP单独列出，以区分信息扩展和学习模块的作用。',
                  '3. 执行时间由当前数据的确定性公式生成。执行公式表现准确对应这一已知机制；MLP结果展示近似误差。',
                  '4. 三独立MLP求和与直接总量MLP作为分阶段建模的辅助对照。最终选中流程还利用机理推演，因此其收益不能全部解释为阶段拆分收益。',
                  '5. 数据集和规则保持不变。本报告检验第三版仿真假设下的预测流程；现场参数与遥测条件由后续实测、专家反馈进一步确定。',
                  '6. 历史第三版测试结果已在此前研究中浏览。本次实验使用现有基准的固定划分，所有本轮选择基于验证集；新数据上的独立确认可作为后续评价。',
                  '', '## 复算与文件', '',
                  f"独立复算通过 {verification['prediction_tables_recomputed']} 份逐调用预测表；检查非负输出、阶段求和、输入源哈希与冻结产物哈希。",
                  '- `selection.json`：最终方法与超参数选择。',
                  '- `validation_search.csv`：全部候选配置的验证结果。',
                  '- `results/`：模型、预处理器、配置、逐轮日志、验证预测。',
                  '- `evaluation/metrics.csv`：分阶段、设备类型、加权方式的完整指标。',
                  '- `evaluation/<split>/`：逐调用真实值与预测值。',
                  '- `evaluation/tail_risk.csv`、`worst_cases.csv`：尾部误差与最大误差调用清单。',
                  '- `evaluation/context_slices.csv`：按数字资源忙碌、物理队列非空分组的误差。',
                  '- `evaluation/network_paired_comparison.json`：配对运行置信区间。',
                  '- `evaluation/independent_verification.json`：独立验算结果。',
                  '- `environment_requirements.txt`：服务器依赖版本。',
                  '', '推理入口仅加载准备好的t₀输入与选定模型，执行 `predict.py --root <实验目录> --inputs <输入.pkl> --out <预测.csv>` 即可在CPU输出三阶段和总时延。机理摘要来自准备阶段的已知队列与地图推演。',
                  '', 'Overleaf实验章保持留空，本轮结果在本报告中交付。'])
    (directory / '实验结果与解读.md').write_text('\n'.join(lines) + '\n')
    aggregated = metrics.groupby(['split','model','stage','group','weighting']).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),p95_error_mean=('p95_absolute_error','mean'),seeds=('seed','nunique'),calls=('n','first')).reset_index()
    aggregated.to_csv(directory / '指标汇总.csv', index=False)
    search.to_csv(directory / '验证集完整选型记录.csv', index=False)
    print('REPORT_COMPLETE', directory)

if __name__ == '__main__':
    main()
