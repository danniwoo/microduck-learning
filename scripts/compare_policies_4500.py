"""Paired CPU MuJoCo/BAM screening comparison; does not touch training."""
import contextlib,io,json,hashlib,time,csv
from pathlib import Path
import numpy as np
import mujoco
import infer_policy as ip
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/comparison-4500';OUT.mkdir(exist_ok=True)
POLICIES={'official_walking':ROOT/'policies/alpha_walking.onnx', 'own_4500':ROOT/'logs/rsl_rl/velocity/2026-10-25_18-12-33_rtx5060-walk/model_4500_eval.onnx'}
SCENARIOS={'stand':[0.,0.,0.], 'forward_015':[.15,0.,0.], 'forward_030':[.3,0.,0.], 'turn_075':[0.,0.,.75]}
rows=[];traces=[]
for name,path in POLICIES.items():
 for scenario,cmd in SCENARIOS.items():
  for seed in (0,1,2):
   rng=np.random.default_rng(seed)
   with contextlib.redirect_stdout(io.StringIO()):
    bam=ip.load_bam_model(200.,7.4,None)
    model,data,ctrl,_=ip.load_mujoco_with_bam(ip.MICRODUCK_XML,bam,.005,.1,6.0)
    p=ip.PolicyInference(model,data,walking_onnx_path=str(path),bam_ctrl=ctrl,new_cmd_obs=True,use_projected_gravity=True)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'trunk_base_freejoint')
    qa=int(model.jnt_qposadr[jid]); va=int(model.jnt_dofadr[jid])
    data.qpos[qa:qa+7]=[0,0,.125,1,0,0,0]
    for i,j in enumerate(p.joint_qpos_indices):data.qpos[j]=p.default_pose[i]+rng.uniform(-.01,.01)
    ctrl.reset(data.qpos);p.set_position_targets(p.default_pose);mujoco.mj_forward(model,data)
   metrics=[];fell=False;reason=None;last=None;first=None
   for step in range(1100): # 2 seconds settle, 20 seconds commanded test
    if step==100:
     with contextlib.redirect_stdout(io.StringIO()):p.set_vel_cmd(*cmd)
     first=data.qpos[qa:qa+3].copy()
    a=p.infer()
    if not np.isfinite(a).all():fell=True;reason='nonfinite_action';break
    p.apply_action(a)
    for _ in range(4):ctrl.update();mujoco.mj_step(model,data)
    z=float(data.qpos[qa+2]);upright=float(data.xmat[p.trunk_base_id].reshape(3,3)[2,2])
    if not np.isfinite(data.qpos).all():fell=True;reason='nonfinite_state';break
    if z<.06 or upright<.5:fell=True;reason='height_or_tilt';break
    if step>=100:
     v=p.quat_rotate_inverse(data.qpos[qa+3:qa+7].astype(np.float32),data.qvel[va:va+3].astype(np.float32))
     yaw=float(data.qvel[va+5])
     delta=float(np.mean((a-last)**2)) if last is not None else 0.
     metrics.append([float(v[0]),float(v[1]),yaw,z,delta])
     if step%10==0:traces.append([name,scenario,seed,(step-99)*.02,float(v[0]),float(v[1]),yaw,z,upright])
     last=a.copy()
   arr=np.array(metrics)
   row={'policy':name,'scenario':scenario,'seed':seed,'command':cmd,'fell':fell,'reason':reason,'survival_test_seconds':round(len(metrics)*.02,2)}
   if len(arr):
    row.update(mean_vx=float(arr[:,0].mean()),mean_vy=float(arr[:,1].mean()),mean_yaw=float(arr[:,2].mean()),vx_mae=float(np.abs(arr[:,0]-cmd[0]).mean()),xy_error=float(np.linalg.norm(arr[:,:2]-np.array(cmd[:2]),axis=1).mean()),yaw_mae=float(np.abs(arr[:,2]-cmd[2]).mean()),mean_trunk_z=float(arr[:,3].mean()),action_delta_mse=float(arr[1:,4].mean()) if len(arr)>1 else 0.)
   rows.append(row);print(json.dumps(row),flush=True)
report={'timestamp':time.time(),'protocol':{'scene':ip.MICRODUCK_XML,'bam':'M6 XL330','vin':7.4,'kp':200,'voltage_sag':.1,'delay':0,'physics_dt':.005,'control_hz':50,'settle_seconds':2,'test_seconds':20,'seeds':[0,1,2],'joint_initial_perturbation_rad':.01,'fall_definition':'trunk z < 0.06 m OR trunk upright cosine < 0.5 OR nonfinite','both_walking_only':True,'limitations':'Three paired initial-condition seeds, flat floor, fixed voltage/friction, no pushes; not independent training seeds or real-robot evidence. Speeds use only surviving test steps.'},'policy_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in POLICIES.items()},'trials':rows}
(OUT/'results.json').write_text(json.dumps(report,indent=2))
with (OUT/'traces.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['policy','scenario','seed','test_time_s','vx','vy','yaw_rate','trunk_z','upright_cosine']);w.writerows(traces)
print('SAVED',OUT,flush=True)