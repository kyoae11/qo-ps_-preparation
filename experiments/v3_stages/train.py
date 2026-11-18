"""Train/validate only. Evaluation partitions are never opened here."""
import argparse,copy,json,os,pickle,random,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from modeling import *

def run(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache)
 if (out/'result.json').exists():return
 t=time.perf_counter();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 x=pd.read_pickle(cache/'train_x.pkl');v=pd.read_pickle(cache/'validation_x.pkl');yy=pd.read_pickle(cache/'train_y.pkl');vv=pd.read_pickle(cache/'validation_y.pkl')
 assert set(x.run_id).isdisjoint(v.run_id)
 target={'network':'network_time','ready':'ready_time','execution':'execution_time','total':'completion_time'}[a.stage]
 prep=Preprocessor(columns(x,a.stage,a.variant)).fit(x)
 y=yy[target].to_numpy();vy=vv[target].to_numpy();offset=x.mechanism_ready.to_numpy() if a.kind=='residual' else np.zeros(len(x));voffset=v.mechanism_ready.to_numpy() if a.kind=='residual' else np.zeros(len(v))
 train_target=y-offset;mean=float(train_target.mean());scale=max(float(train_target.std()),1e-8)
 device=torch.device('cuda:0' if a.gpu else 'cpu')
 if a.gpu:
  assert '5880' in torch.cuda.get_device_name(0), 'Only user-authorized RTX5880 GPUs may train'
  torch.cuda.set_per_process_memory_fraction(.15,0)
 def save_pred(p):
  df=v[['run_id','invocation_id','entity_id','entity_type','world_id']].copy();df['actual']=vy;df['pred']=p;df.to_csv(out/'validation_predictions.csv.gz',index=False)
 if a.kind=='tree':
  tx=prep.dense(x);vx=prep.dense(v);model=HistGradientBoostingRegressor(max_iter=300,learning_rate=a.lr,max_leaf_nodes=a.width,early_stopping=False,random_state=a.seed,l2_regularization=1.)
  model.fit(tx,(train_target-mean)/scale);p=np.maximum(0,model.predict(vx)*scale+mean+voffset);score=metrics(vy,p)
  with open(out/'model.pkl','wb') as f:pickle.dump(model,f)
  best_epoch=300;history=[];params=0
 else:
  if a.kind=='afil':
   xn,xc=prep.transform(x);vn,vc=prep.transform(v);inputs=[torch.from_numpy(xn).to(device),torch.from_numpy(xc).to(device)];vinputs=[torch.from_numpy(vn).to(device),torch.from_numpy(vc).to(device)];model=AFIL(prep,a.width,a.variant if a.architecture=='auto' else a.architecture).to(device)
  else:
   xx=prep.dense(x);vx=prep.dense(v);inputs=[torch.from_numpy(xx).to(device)];vinputs=[torch.from_numpy(vx).to(device)];model=MLP(xx.shape[1],a.width).to(device)
  ty=torch.tensor((train_target-mean)/scale,dtype=torch.float32,device=device);tv=torch.tensor(vy,dtype=torch.float32,device=device);vo=torch.tensor(voffset,dtype=torch.float32,device=device)
  opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);best=float('inf');bad=0;history=[];best_epoch=0
  def infer():
   model.eval()
   with torch.no_grad():return torch.cat([model(*[z[i:i+4096] for z in vinputs]) for i in range(0,len(v),4096)])*scale+mean+vo
  for epoch in range(1,a.epochs+1):
   model.train();perm=torch.randperm(len(x),device=device);total=0.;et=time.perf_counter()
   for idx in perm.split(a.batch):
    opt.zero_grad(set_to_none=True);pr=model(*[z[idx] for z in inputs]);loss=nn.functional.l1_loss(pr,ty[idx]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();total+=loss.item()*len(idx)
   raw=infer();p=raw.clamp_min(0);mae=(p-tv).abs().mean().item();history.append(dict(epoch=epoch,train_standardized_mae=total/len(x),validation_mae=mae,seconds=time.perf_counter()-et))
   if mae<best-1e-9:best=mae;bad=0;best_epoch=epoch;best_state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}
   else:bad+=1
   print(json.dumps(dict(stage=a.stage,kind=a.kind,variant=a.variant,seed=a.seed,epoch=epoch,train_mae=total/len(x)*scale,val_mae=mae,seconds=history[-1]["seconds"])),flush=True)
   if bad>=12:break
  model.load_state_dict(best_state);p=infer().clamp_min(0).cpu().numpy().astype(float);score=metrics(vy,p);torch.save(best_state,out/'model.pt');params=sum(z.numel() for z in model.parameters())
  # Check serialized weights recreate exactly the same validation predictions.
  model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True));p2=infer().clamp_min(0).cpu().numpy();assert np.allclose(p,p2,rtol=0,atol=1e-6)
 save_pred(p)
 with open(out/'preprocessor.pkl','wb') as f:pickle.dump(prep,f)
 result=dict(config=vars(a),target=target,target_mean=mean,target_scale=scale,metrics=score,best_epoch=best_epoch,parameters=params,elapsed_seconds=time.perf_counter()-t,features=prep.cols,torch=torch.__version__,python=sys.version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if a.gpu else 'cpu')
 (out/'history.json').write_text(json.dumps(history,indent=2));(out/'result.json').write_text(json.dumps(result,indent=2));print('TRAIN_COMPLETE',str(out),score,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stage',choices=['network','ready','execution','total'],required=True);ap.add_argument('--kind',choices=['afil','mlp','tree','residual'],required=True);ap.add_argument('--variant',default='full');ap.add_argument('--width',type=int,default=64);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--seed',type=int,default=11);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--architecture',default='auto');ap.add_argument('--gpu',action='store_true');run(ap.parse_args())
