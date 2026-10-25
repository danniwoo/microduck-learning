"""One-physics-step replay on identical states, controls and compiled model."""
import json,copy,contextlib,io
from pathlib import Path
from dataclasses import asdict
import numpy as np, torch,mujoco
from checkpoint_review import *
OUT=ROOT/'logs/sim-parity-audit'
configure_torch_backends();env=ManagerBasedRlEnv(cfg=make_cfg(1),device='cuda:0')
ac=load_rl_cfg(BASE);wrap=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions);runner=(load_runner_cls(BASE) or MjlabOnPolicyRunner)(wrap,asdict(ac),device='cuda:0')
runner.load(str(RUN/'model_4500.pt'),load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
env.reset(seed=20261025);env.sim.data.qpos[:]=torch.tensor(np.load(OUT/'initial-qpos.npy')[:1],device='cuda:0');env.sim.data.qvel[:]=0;env.sim.forward()
m=copy.copy(env.sim.mj_model);d=mujoco.MjData(m);rows=[]
for step in range(150):
 env.command_manager.get_term('twist').command[:]=torch.tensor([[.3 if step>=100 else 0,0,0]],device='cuda:0');env.observation_manager.compute(update_history=True)
 with torch.inference_mode():a=policy(wrap.get_observations())
 env.action_manager.process_action(a)
 for sub in range(4):
  env.action_manager.apply_action();env.scene.write_data_to_sim()
  for k in ['qpos','qvel','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied']:
   getattr(d,k)[:]=getattr(env.sim.data,k)[0].cpu().numpy()
  for k in ['dof_frictionloss','dof_damping']:
   getattr(m,k)[:]=getattr(env.sim.model,k)[0].cpu().numpy()
  mujoco.mj_step(m,d);env.sim.step();env.scene.update(dt=env.physics_dt)
  dv=d.qvel-env.sim.data.qvel[0].cpu().numpy();ncon_gpu=int(env.sim.data.nacon[0])
  rows.append({'step':step,'substep':sub,'qvel_max_abs':float(np.max(np.abs(dv))),'root_qvel_max_abs':float(np.max(np.abs(dv[:6]))),'cpu_contacts':int(d.ncon),'gpu_contacts':ncon_gpu})
 env.sim.forward()
 if step%50==0:print(step,rows[-1],flush=True)
(OUT/'one-step-physics.json').write_text(json.dumps(rows,indent=2));env.close()