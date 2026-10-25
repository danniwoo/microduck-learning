"""Paired CPU-only ablations; no retraining, no checkpoint writes."""
import json,contextlib,io,copy
from pathlib import Path
import numpy as np,mujoco
import infer_policy as ip
from bam.mujoco import MujocoController
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/sim-parity-audit';RUN=ROOT/'logs/rsl_rl/velocity/2026-10-25_18-12-33_rtx5060-walk'
ref=mujoco.MjModel.from_binary_path(str(OUT/'warp-model.mjb'))
initial=np.load(OUT/'initial-qpos.npy');rows=[]
variants=['fresh_observation_only','forward_only','friction_index_only','forward_and_friction','matched_all']
for variant in variants:
 for i in range(6):
  with contextlib.redirect_stdout(io.StringIO()):
   bam=ip.load_bam_model(200.,7.4,None)
   if variant=='matched_all':
    m=copy.copy(ref);d=mujoco.MjData(m);ctrl=MujocoController(bam,[m.actuator(j).name for j in range(m.nu)],m,d,vin_drop_gain=.1,vin_min=6.)
   else:
    scene=ip.MICRODUCK_XML.replace('scene.xml','scene_walk.xml') if variant.startswith('walk') else ip.MICRODUCK_XML
    m,d,ctrl,_=ip.load_mujoco_with_bam(scene,bam,.005,.1,6.)
    if variant in ['integrator_only','walk_plus_physics']:m.opt.integrator=ref.opt.integrator
    if variant in ['solver_iterations_only','walk_plus_physics']:
     m.opt.iterations=ref.opt.iterations;m.opt.ls_iterations=ref.opt.ls_iterations
   p=ip.PolicyInference(m,d,walking_onnx_path=str(RUN/'model_4500_eval.onnx'),bam_ctrl=ctrl,new_cmd_obs=True,use_projected_gravity=True)
   if variant=='matched_all':
    p.imu_ang_vel_id=m.sensor('robot/imu_ang_vel').id;p.trunk_base_id=m.body('robot/trunk_base').id;p._trunk_qpos_adr=0
   assert m.nq==21 and m.nv==20 and m.nu==14
   d.qpos[:]=initial[i];d.qvel[:]=0;ctrl.reset(d.qpos);p.set_position_targets(p.default_pose);mujoco.mj_forward(m,d);p.set_vel_cmd(0,0,0)
  # Diagnostic-local correction only. joint ID 1 maps to DOF 6 for a floating base.
  if variant in ['friction_index_only','forward_and_friction','matched_all']:
   import inspect,textwrap,types
   source=textwrap.dedent(inspect.getsource(MujocoController.update))
   assert source.count('[self.joint_indexes]')==1
   namespace={'np':np,'mujoco':mujoco}
   exec(source.replace('[self.joint_indexes]','[self.dof_indexes]'),namespace)
   ctrl.update=types.MethodType(namespace['update'],ctrl)
  if variant=='fresh_observation_only':
   p.get_base_ang_vel=lambda:d.qvel[3:6].astype(np.float32).copy()
   p.get_projected_gravity=lambda:p.quat_rotate_inverse(d.qpos[3:7].astype(np.float32),np.array([0,0,-1],dtype=np.float32))
  data=[];yaws=[];contacts=set();fell=False
  for step in range(1600):
   if step==100:
    with contextlib.redirect_stdout(io.StringIO()):p.set_vel_cmd(.3 if i>=3 else 0,0,0)
    start=d.qpos[:2].copy()
   a=p.infer();p.apply_action(a)
   for sub in range(4):
    ctrl.update();mujoco.mj_step(m,d)
    for ct in d.contact:
     for g in ct.geom:
      contacts.add(m.geom(g).name or f'body:{m.body(m.geom_bodyid[g]).name}')
   if variant in ['forward_only','forward_and_friction','matched_all']:mujoco.mj_forward(m,d)
   if step>=100:yaws.append(float(d.qvel[5]))
   if d.qpos[2]<.06 or d.xmat[p.trunk_base_id].reshape(3,3)[2,2]<.5:fell=True
   if step>=100:data.append(float(p.quat_rotate_inverse(d.qpos[3:7].astype(np.float32),d.qvel[:3].astype(np.float32))[0]))
  row={'variant':variant,'seed':i%3,'scenario':'stand' if i<3 else 'forward_030','net_cm':float(np.linalg.norm(d.qpos[:2]-start)*100),'vx':float(np.mean(data)),'yaw':float(np.mean(yaws)),'fell':fell,'contacts':sorted(contacts)}
  rows.append(row);print(json.dumps(row),flush=True);(OUT/'cpu-timing-friction-ablations.json').write_text(json.dumps(rows,indent=2))