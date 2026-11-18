"""Finite predeclared search with bounded independent workers per assigned GPU."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path

def jobs_for(stage,kind,variant):
 widths=[32,64] if kind=='afil' else ([15,31] if kind=='tree' else [64,128]);lrs=[.05,.1] if kind=='tree' else [.001,.0003]
 seeds=[11] if kind=='tree' else [11,22,33]
 return [dict(stage=stage,kind=kind,variant=variant,width=w,lr=lr,seed=s) for w in widths for lr in lrs for s in seeds]
def name(j):return f"{j['stage']}_{j['kind']}_{j['variant']}_w{j['width']}_lr{j['lr']}_s{j['seed']}"
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpus',default='1');ap.add_argument('--workers-per-gpu',type=int,choices=[1,2],default=1);ap.add_argument('--phase',choices=['search','ablation'],default='search');a=ap.parse_args();root=Path(a.root).resolve();code=Path(__file__).resolve().parent;gpus=a.gpus.split(',');assert '2' not in gpus, 'User prohibits GPU2/RTX6000D';assert len(gpus)<=2 and len(set(gpus))==len(gpus);workers=[g for g in gpus for _ in range(a.workers_per_gpu)];jobs=[]
 if a.phase=='search':
  # Interleave stages to obtain an end-to-end baseline early.
  specs=[('network','afil','base'),('network','afil','full'),('network','afil','gated'),('network','afil','base_id'),('network','afil','full_id'),('network','afil','gated_id'),('ready','mlp','full'),('ready','residual','full'),('ready','mlp','raw'),('network','mlp','full'),('execution','mlp','raw'),('total','mlp','raw'),('ready','tree','full'),('network','tree','full')]
  groups=[jobs_for(*s) for s in specs]
  for n in range(max(map(len,groups))):
   jobs.extend(g[n] for g in groups if n<len(g))
 else:
  select=json.loads((root/'selection.json').read_text());net=select['network'];
  for core in ['no_cap','no_state']:
   variant=core+('_id' if net['variant'].endswith('_id') else '')
   jobs.extend(dict(stage='network',kind='afil',variant=variant,width=net['width'],lr=net['lr'],seed=s,architecture=net['variant']) for s in [11,22,33])
 (root/f'{a.phase}_execution_schedule.json').write_text(json.dumps(dict(physical_gpus=gpus,workers_per_gpu=a.workers_per_gpu,cpu_threads_per_worker=4,training_candidates_unchanged=True),indent=2))
 (root/f'{a.phase}_jobs.json').write_text(json.dumps(jobs,indent=2))
 def worker(gpu,jobs):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  for j in jobs:
   out=root/'results'/name(j)
   if (out/'result.json').exists():continue
   out.mkdir(parents=True,exist_ok=True)
   cmd=[str(root/'env/bin/python'),str(code/'train.py'),'--cache',str(root/'cache'),'--out',str(out)]
   for k,v in j.items():cmd.extend(['--'+k,str(v)])
   if j['kind']!='tree':cmd.append('--gpu')
   with open(out/'console.log','a') as f:
    r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   print('JOB',gpu,name(j),'exit',r.returncode,flush=True)
   if r.returncode:
    (out/'failure.json').write_text(json.dumps(dict(exit_code=r.returncode,job=j)))
    raise RuntimeError('Training failed: '+str(out))
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as ex:
  futures=[ex.submit(worker,g,jobs[i::len(workers)]) for i,g in enumerate(workers)]
  for f in futures:f.result()
 print('PHASE_COMPLETE',a.phase,flush=True)
if __name__=='__main__':main()
