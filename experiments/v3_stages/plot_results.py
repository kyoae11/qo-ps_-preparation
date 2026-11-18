"""Plot audited metrics; no training or selection. Requires matplotlib."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
root=Path(args.root);out=root/'reports/figures';out.mkdir(parents=True,exist_ok=True)
allmetrics=pd.read_csv(root/'evaluation/metrics.csv')
metrics=allmetrics[(allmetrics.group=='all')&(allmetrics.weighting=='call')]
sel=json.loads((root/'selection.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
 plt.close(fig)
def summary(models,stage,split='test',metric='mae'):
 means=[];std=[]
 for model in models:
  values=metrics[(metrics.model==model)&(metrics.stage==stage)&(metrics.split==split)][metric]
  means.append(values.mean());std.append(values.std(ddof=1) if len(values)>1 else 0)
 return np.array(means),np.array(std)

models=['network_afil_base','network_afil_base_id','network_afil_full','network_afil_full_id','network_afil_gated','network_afil_gated_id','network_mlp_full','network_tree_full','mechanistic']
labels=['AFIL base','AFIL base + IDs','AFIL entity interaction','AFIL entity interaction + IDs','AFIL entity gate','AFIL entity gate + IDs','MLP, full context','Boosted trees, full context','Mechanistic']
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,metric,title in zip(axes,['mae','rmse','p95_absolute_error'],['MAE (s)','RMSE (s)','P95 absolute error (s)']):
 mu,sd=summary(models,'network_time',metric=metric);ax.barh(np.arange(len(models)),mu,xerr=sd,color=['#607d8b']*2+['#008c95']*4+['#8eaeb7']*2+['#c65516'],capsize=2)
 ax.set_yticks(np.arange(len(models)),labels);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
axes[0].invert_yaxis();fig.suptitle('Network prediction: held-out standard test runs\nBars = training-seed means; error bars = seed standard deviations',fontsize=12)
fig.tight_layout();save(fig,'network_controls')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,stage,models,labels in [
 (axes[0],'ready_time',['ready_mlp_raw','ready_mlp_full','ready_residual_full','ready_tree_full','mechanistic'],['MLP, summaries','MLP + mechanism feature','Mechanism + residual MLP','Trees + mechanism feature','Mechanistic']),
 (axes[1],'completion_time',['total_mlp_raw','independent_mlp','mechanistic','selected_pipeline'],['Direct-total MLP','Three independent MLPs','Mechanistic sum','Selected stage pipeline'])]:
 mu,sd=summary(models,stage);ax.barh(np.arange(len(models)),mu,color='#008c95');ax.set_yticks(np.arange(len(models)),labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('MAE (s), logarithmic scale');ax.set_title(stage.replace('_',' ').capitalize());ax.grid(axis='x',alpha=.2)
 for i,v in enumerate(mu):ax.annotate(f'{v:.3f}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
fig.suptitle('Standard test: readiness and total completion',fontsize=12);fig.tight_layout();save(fig,'ready_and_total')

paired=json.loads((root/'evaluation/network_paired_comparison.json').read_text());paired=[d for d in paired if d['split']!='validation']
splits=[d['split'] for d in paired];labels=[s.replace('test_unseen_','New ').replace('test_joint','Joint shift').replace('test','Standard').replace('stress','Stress').replace('_',' ') for s in splits]
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(paired));width=.34
ax.bar(x-width/2,[d['base_mae'] for d in paired],width,label='Validation-selected stronger AFIL base',color='#607d8b')
ax.bar(x+width/2,[d['enhanced_mae'] for d in paired],width,label='Validation-selected entity AFIL',color='#008c95')
ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_ylabel('Network MAE (s)');ax.set_title('Network prediction across evaluation conditions');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'network_generalization')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,title in zip(axes,['network_baseline','network'],['Stronger AFIL reference','Selected entity-aware AFIL']):
 for color,(seed,folder) in zip(['#008c95','#c65516','#596ab1'],sorted(sel[key]['folders'].items())):
  trial=root/'results'/folder
  history=pd.DataFrame(json.loads((trial/'history.json').read_text()))
  result=json.loads((trial/'result.json').read_text())
  ax.plot(history.epoch,history.train_standardized_mae*result['target_scale'],color=color,linestyle='--',alpha=.6,label=f'Seed {seed}: train')
  ax.plot(history.epoch,history.validation_mae,color=color,label=f'Seed {seed}: validation')
  ax.scatter([result['best_epoch']],[result['metrics']['mae']],color=color,s=24,zorder=3)
 ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Network MAE (s)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,ncol=2)
fig.suptitle('Recorded training process; dots mark selected checkpoints',fontsize=12)
fig.tight_layout();save(fig,'network_training_curves')
(out/'plot_sources.json').write_text(json.dumps({'source':'evaluation/metrics.csv, network_paired_comparison.json, results/*/history.json and result.json','matplotlib':matplotlib.__version__,'selection':'selection.json','statistics':'call-weighted; mean across trained seeds, not ensemble predictions','training_curve':'Online batch loss, before final nonnegative clipping; validation MAE after clipping. Both expressed in seconds.'},indent=2))
print('PLOTS_COMPLETE',out)
