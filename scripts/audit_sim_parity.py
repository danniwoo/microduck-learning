"""No-training sim-to-sim diagnostic. Uses immutable model_4500 weights."""
import json, contextlib, io, copy, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch, mujoco
import infer_policy as ip
from checkpoint_review import make_cfg,BASE,RUN,ManagerBasedRlEnv,RslRlVecEnvWrapper,load_rl_cfg,load_runner_cls,MjlabOnPolicyRunner,configure_torch_backends
OUT=Path('logs/sim-parity-audit');OUT.mkdir(exist_ok=True)
OPTIONS=['timestep','integrator','solver','cone','iterations','ls_iterations','tolerance','ls_tolerance','impratio','disableflags','enableflags','jacobian','noslip_iterations','o_margin','gravity']
def val(x):
 return np.asarray(x).tolist()
def audit(m):
 return {'nq':m.nq,'nv':m.nv,'nu':m.nu,'options':{k:val(getattr(m.opt,k)) for k in OPTIONS},'bodies':{m.body(i).name.replace('robot/',''):{k:val(getattr(m,'body_'+k)[i]) for k in ['mass','inertia','ipos','iquat']} for i in range(m.nbody)},'joints':{m.joint(i).name.replace('robot/',''):{k:val(getattr(m,'jnt_'+k)[i]) for k in ['type','axis','pos','range','qposadr','dofadr']} for i in range(m.njnt)},'geoms':{m.geom(i).name.replace('robot/','') or f'unnamed{i}':{k:val(getattr(m,'geom_'+k)[i]) for k in ['type','size','pos','friction','condim','priority','contype','conaffinity','solref','solimp']} for i in range(m.ngeom)},'armature':m.dof_armature.tolist(),'actuator_names':[m.actuator(i).name for i in range(m.nu)]}
configure_torch_backends()
cfg=make_cfg(6);env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');robot=env.scene['robot']
ac=load_rl_cfg(BASE);wrap=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
runner=(load_runner_cls(BASE) or MjlabOnPolicyRunner)(wrap,asdict(ac),device='cuda:0')
runner.load(str(RUN/'model_4500.pt'),load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
with contextlib.redirect_stdout(io.StringIO()):
 cm,cd,cc,_=ip.load_mujoco_with_bam(ip.MICRODUCK_XML,ip.load_bam_model(200,7.4,None),.005,.1,6.)
 cp=ip.PolicyInference(cm,cd,walking_onnx_path=str(RUN/'model_4500_eval.onnx'),bam_ctrl=cc,new_cmd_obs=True,use_projected_gravity=True)
report={'cpu_original':audit(cm),'warp_compiled':audit(env.sim.mj_model),'clip_actions':ac.clip_actions,'checkpoint_sha256':hashlib.sha256((RUN/'model_4500.pt').read_bytes()).hexdigest()}
(OUT/'compiled-audit.json').write_text(json.dumps(report,indent=2))
mujoco.mj_saveModel(env.sim.mj_model,str(OUT/'warp-model.mjb'))
qinitial=np.array([[0,0,.125,1,0,0,0,*list(ip.DEFAULT_POSE+np.random.default_rng(s%3).uniform(-.01,.01,14))] for s in range(6)],dtype=np.float32)
env.reset(seed=20261025)
assert env.sim.data.qpos.shape==qinitial.shape
env.sim.data.qpos[:]=torch.tensor(qinitial,device='cuda:0');env.sim.data.qvel[:]=0
env.sim.forward();cmdterm=env.command_manager.get_term('twist');cmd=torch.zeros((6,3),device='cuda:0')
np.save(OUT/'initial-qpos.npy',qinitial)
export_max=0.;obs_diffs=[];data=[];saved_obs=[]
for step in range(1600):
 if step==100:cmd[3:,0]=.3;start=env.sim.data.qpos[:,:2].cpu().numpy().copy()
 cmdterm.command[:]=cmd;env.observation_manager.compute(update_history=True);obs=wrap.get_observations()
 with torch.inference_mode():action=policy(obs)
 if step%10==0:
  xx=obs['actor'].cpu().numpy();saved_obs.append(xx.copy())
  aa=np.concatenate([cp.ort_session.run(None,{cp.input_name:x[None]})[0] for x in xx])
  export_max=max(export_max,float(np.max(np.abs(aa-action.cpu().numpy()))))
  # Reconstruct CPU observations from the exact GPU state; forward updates kinematics/sensors.
  cd.qpos[:]=env.sim.data.qpos[0].cpu().numpy();cd.qvel[:]=env.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(cm,cd)
  cp.last_action=xx[0,34:48].copy();cp.command[:]=xx[0,48:]
  diff=cp.get_observations()-xx[0]
  obs_diffs.append([float(np.max(np.abs(diff[a:b]))) for a,b in [(0,3),(3,6),(6,20),(20,34),(34,48),(48,61)]])
 wrap.step(action)
 if step>=100:
  q=env.sim.data.qpos.cpu().numpy();v=env.sim.data.qvel.cpu().numpy()
  # Shared measurement: current freejoint pose/velocity, identical to CPU evaluator.
  vx=[float(cp.quat_rotate_inverse(x[3:7],y[:3])[0]) for x,y in zip(q,v)]
  data.append(vx)
 if step%400==0:print('GPU',step,'export max',export_max,flush=True)
report['export_max_abs_action_error']=export_max
report['obs_diff_max_same_qpos_qvel_after_cpu_forward']=np.max(obs_diffs,axis=0).tolist()
report['warp_trials']=[{'seed':i%3,'scenario':'stand' if i<3 else 'forward_030','net_cm':float(np.linalg.norm(q[i,:2]-start[i])*100),'vx':float(np.mean(np.asarray(data)[:,i]))} for i in range(6)]
np.save(OUT/'sample-observations.npy',np.concatenate(saved_obs))
(OUT/'compiled-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['cpu_original','warp_compiled']}),flush=True)
env.close()