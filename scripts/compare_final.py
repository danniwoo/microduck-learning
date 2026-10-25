"""Final-checkpoint screening: paired starts, fixed commands, CPU MuJoCo + BAM.
Outputs are evaluation evidence, not a claim of statistical training superiority.
"""
import contextlib,io,json,hashlib,time,csv
from pathlib import Path
import numpy as np
import mujoco
import infer_policy as ip
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/comparison-final-4999'; OUT.mkdir(exist_ok=True)
OWN=ROOT/'logs/rsl_rl/velocity/2026-10-25_18-12-33_rtx5060-walk/model_4999_eval.onnx'
WALK=ROOT/'policies/alpha_walking.onnx'; STAND=ROOT/'policies/alpha_stand.onnx'
POLICIES={'official_walking':(WALK,None),'official_switch':(WALK,STAND),'own_4999':(OWN,None)}
SCENARIOS={'stand':[0.,0.,0.], 'forward_015':[.15,0.,0.], 'forward_030':[.3,0.,0.], 'turn_075':[0.,0.,.75], 'walk_then_stop':[.3,0.,0.]}
rows=[];traces=[]
for name,(path,standing) in POLICIES.items():
 for scenario,initial_cmd in SCENARIOS.items():
  for seed in range(5):
   rng=np.random.default_rng(seed)
   with contextlib.redirect_stdout(io.StringIO()):
    bam=ip.load_bam_model(200.,7.4,None)
    model,data,ctrl,_=ip.load_mujoco_with_bam(ip.MICRODUCK_XML,bam,.005,.1,6.0)
    p=ip.PolicyInference(model,data,walking_onnx_path=str(path),standing_onnx_path=str(standing) if standing else None,bam_ctrl=ctrl,new_cmd_obs=True,use_projected_gravity=True)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'trunk_base_freejoint')
    qa=int(model.jnt_qposadr[jid]); va=int(model.jnt_dofadr[jid])
    data.qpos[qa:qa+7]=[0,0,.125,1,0,0,0]
    for i,j in enumerate(p.joint_qpos_indices):data.qpos[j]=p.default_pose[i]+rng.uniform(-.01,.01)
    ctrl.reset(data.qpos);p.set_position_targets(p.default_pose);mujoco.mj_forward(model,data)
    p.set_vel_cmd(0,0,0)
   metrics=[];fell=False;reason=None;cmd=np.zeros(3);start_xy=None;stop_xy=None;step=-1
   for step in range(1600): # 2 seconds settle + 30 commanded seconds
    if step==100 or (scenario=='walk_then_stop' and step==600):
     cmd=np.array(initial_cmd if step==100 else [0.,0.,0.])
     with contextlib.redirect_stdout(io.StringIO()):p.set_vel_cmd(*cmd)
     assert np.allclose(p.command[:3],cmd), (name,scenario,p.command,cmd)
     if step==100:start_xy=data.qpos[qa:qa+2].copy()
     else:stop_xy=data.qpos[qa:qa+2].copy()
    a=p.infer()
    if not np.isfinite(a).all():fell=True;reason='nonfinite_action';break
    p.apply_action(a)
    for _ in range(4):ctrl.update();mujoco.mj_step(model,data)
    z=float(data.qpos[qa+2]);upright=float(data.xmat[p.trunk_base_id].reshape(3,3)[2,2])
    if not np.isfinite(data.qpos).all():fell=True;reason='nonfinite_state';break
    if z<.06 or upright<.5:fell=True;reason='height_or_tilt';break
    if step>=100:
     v=p.quat_rotate_inverse(data.qpos[qa+3:qa+7].astype(np.float32),data.qvel[va:va+3].astype(np.float32))
     yaw=float(data.qvel[va+5]); t=(step-99)*.02
     metrics.append([t,float(v[0]),float(v[1]),yaw,float(np.linalg.norm(v[:2]-cmd[:2])),abs(yaw-cmd[2]),float(np.linalg.norm(v[:2]))])
     if step%10==0:traces.append([name,scenario,seed,t,*cmd,float(v[0]),float(v[1]),yaw,z,upright,*data.qpos[qa:qa+2],p.current_policy])
   arr=np.array(metrics)
   row={'policy':name,'scenario':scenario,'seed':seed,'fell':fell,'reason':reason,'survival_test_seconds':round(len(metrics)*.02,2),'final_policy':p.current_policy}
   if len(arr):
    row.update(mean_vx=float(arr[:,1].mean()),mean_vy=float(arr[:,2].mean()),mean_yaw=float(arr[:,3].mean()),xy_error=float(arr[:,4].mean()),yaw_mae=float(arr[:,5].mean()),planar_speed=float(arr[:,6].mean()),net_displacement=float(np.linalg.norm(data.qpos[qa:qa+2]-start_xy)))
    steady=arr[arr[:,0]>(15 if scenario=='walk_then_stop' else 5)]
    if len(steady):row.update(steady_vx=float(steady[:,1].mean()),steady_yaw=float(steady[:,3].mean()),steady_xy_error=float(steady[:,4].mean()),steady_planar_speed=float(steady[:,6].mean()))
    if stop_xy is not None:
     stopped=arr[arr[:,0]>10]
     row.update(stop_planar_speed=float(stopped[:,6].mean()),stop_yaw_mae=float(stopped[:,5].mean()),stop_net_displacement=float(np.linalg.norm(data.qpos[qa:qa+2]-stop_xy)))
   rows.append(row);print(json.dumps(row),flush=True)
report={'timestamp':time.time(),'protocol':{'scene':ip.MICRODUCK_XML,'bam':'M6 XL330','vin':7.4,'kp':200,'voltage_sag':.1,'action_delay':0,'physics_dt':.005,'control_hz':50,'settle_seconds':2,'test_seconds':30,'seeds':list(range(5)),'joint_initial_perturbation_rad':.01,'fall_definition':'trunk z < 0.06 m OR upright cosine < 0.5 OR nonfinite','scenarios':SCENARIOS,'walk_then_stop':'10 s forward 0.3 then 20 s zero; steady stop excludes first 5 s after stopping','official_switch_threshold':.05,'limitations':'Five paired initial-condition seeds, flat floor, fixed voltage/friction, no pushes or training DR; not independent training seeds or real-robot evidence. Metrics use surviving steps. Explicit command slot assertions.'},'policy_sha256':{str(v.relative_to(ROOT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in [OWN,WALK,STAND]},'trials':rows}
(OUT/'results.json').write_text(json.dumps(report,indent=2))
with (OUT/'traces.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['policy','scenario','seed','test_time_s','cmd_vx','cmd_vy','cmd_yaw','vx','vy','yaw_rate','trunk_z','upright','world_x','world_y','active_policy']);w.writerows(traces)
print('SAVED',OUT,flush=True)