"""Read-only checkpoint review. Shared fixed mjlab profile for evaluation and play.
Never trains or saves weights; original registered training config is untouched.
"""
import argparse, json, hashlib, csv, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls,register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import run_play,PlayConfig
from mjlab.utils.torch import configure_torch_backends
BASE='Mjlab-Velocity-Flat-MicroDuck'
TASK='Mjlab-Velocity-Flat-MicroDuck-FixedReview'
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'logs/rsl_rl/velocity/2026-10-25_18-12-33_rtx5060-walk'
OUT=ROOT/'logs/checkpoint-review-fixed'
SCENARIOS=['stand','forward_015','forward_030','turn_075','walk_then_stop']
PROFILE={'name':'fixed-review-v1','backend':'mjlab MuJoCo Warp','pushes':False,'physics_dr':False,'sensor_noise':False,'imu_bias_deg':0,'encoder_bias':False,'observation_delay':0,'action_delay':0,'vin':7.4,'vin_sag':.1,'kp':200,'head_body_commands':'zero','seed':20261025,'paired_initial_states':True,'reset_joint_offset':[-.01,.01],'control_hz':50,'settle_s':2,'test_s':30,'fall':'z<.06 or upright cosine<.5 or nonfinite','ranking':'zero-fall gate, then mean of 5 dimensionless errors: stand displacement/.05, xy_error@.15/.15, xy_error@.30/.30, yaw_MAE@.75/.75, stop displacement/.05; lower better; equal task weights chosen before results'}
def make_cfg(n):
 c=load_env_cfg(BASE,play=True)
 c.seed=PROFILE['seed']; c.scene.num_envs=n; c.curriculum={}
 c.episode_length_s=120.; c.terminations={}
 keep={'reset_base','reset_robot_joints','expand_bam_friction_fields','reset_action_history'}
 c.events={k:v for k,v in c.events.items() if k in keep}
 c.events['reset_base'].params['pose_range']={'x':(0,0),'y':(0,0),'z':(.125,.125),'yaw':(0,0)}
 c.events['reset_base'].params['velocity_range']={k:(0,0) for k in ['x','y','z','roll','pitch','yaw']}
 c.events['reset_robot_joints'].params['position_range']=(-.01,.01)
 c.events['reset_robot_joints'].params['velocity_range']=(0,0)
 for g in c.observations.values():
  g.enable_corruption=False
  for t in g.terms.values():
   t.noise=None;t.delay_min_lag=0;t.delay_max_lag=0
   if 'max_angle_deg' in t.params:t.params['max_angle_deg']=0.
   if 'biased' in t.params:t.params['biased']=False
 for a in c.scene.entities['robot'].articulation.actuators:
  a.delay_min_lag=0;a.delay_max_lag=0;a.vin=7.4;a.vin_range=(7.4,7.4);a.vin_drop_gain_range=(.1,.1)
 tw=c.commands['twist'];tw.resampling_time_range=(3600,3600)
 tw.heading_command=False;tw.ranges.heading=None
 for k in ['rel_standing_envs','rel_heading_envs','rel_world_envs','rel_forward_envs','rel_turn_in_place_envs','init_velocity_prob']:setattr(tw,k,0.)
 tw.ranges.lin_vel_x=(0,0);tw.ranges.lin_vel_y=(0,0);tw.ranges.ang_vel_z=(0,0)
 for name in ['head_pose','body_pose']:
  c.commands[name].ranges=tuple((0.,0.) for _ in c.commands[name].ranges)
  c.commands[name].resampling_time_range=(3600,3600)
 c.viewer.distance=.8;c.viewer.elevation=-20.;c.viewer.lookat=(0,0,0)
 assert 'push_robot' not in c.events and not c.curriculum
 return c

def evaluate():
 configure_torch_backends();OUT.mkdir(exist_ok=True)
 (OUT/'protocol.json').write_text(json.dumps(PROFILE,indent=2))
 cfg=make_cfg(25); env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
 ac=load_rl_cfg(BASE);wrapped=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
 runner=(load_runner_cls(BASE) or MjlabOnPolicyRunner)(wrapped,asdict(ac),device='cuda:0')
 robot=env.scene['robot'];cmdterm=env.command_manager.get_term('twist')
 rows=[];traces=[];hashes={};initials={};reference=None
 for iteration in [4250,4500,4750,4999]:
  path=RUN/f'model_{iteration}.pt';hashes[str(iteration)]=hashlib.sha256(path.read_bytes()).hexdigest()
  runner.load(str(path),load_cfg={'actor':True},strict=True,map_location='cuda:0')
  policy=runner.get_inference_policy(device='cuda:0')
  env.reset(seed=PROFILE['seed'])
  initial=torch.cat([robot.data.root_link_pos_w,robot.data.root_link_quat_w,robot.data.joint_pos],dim=1).cpu().numpy()
  if reference is None:reference=initial.copy()
  assert np.array_equal(initial,reference),'Initial states differ between checkpoints'
  initials[str(iteration)]=hashlib.sha256(initial.tobytes()).hexdigest()
  cmd=torch.zeros((25,3),device='cuda:0');alive=np.ones(25,dtype=bool);fall_step=np.full(25,-1);data=[];start=None;stop=None
  for step in range(1600):
   if step==100:
    cmd[5:10,0]=.15;cmd[10:15,0]=.30;cmd[15:20,2]=.75;cmd[20:25,0]=.30
    start=robot.data.root_link_pos_w[:,:2].cpu().numpy().copy()
   if step==600:
    cmd[20:25]=0;stop=robot.data.root_link_pos_w[:,:2].cpu().numpy().copy()
   cmdterm.command[:]=cmd
   env.observation_manager.compute(update_history=True)  # refresh cached command observations
   obs=wrapped.get_observations()
   assert obs['actor'].shape==(25,61)
   assert torch.allclose(obs['actor'][:,-13:-10],cmd), (step,obs['actor'][5].tolist(),cmd[5].tolist(),cmdterm.command[5].tolist())
   assert torch.count_nonzero(obs['actor'][:,-10:])==0,'Head/body command is not zero'
   with torch.inference_mode():a=policy(obs)
   if not torch.isfinite(a).all():raise RuntimeError('Nonfinite action')
   wrapped.step(a)
   pos=robot.data.root_link_pos_w.cpu().numpy().copy();q=robot.data.root_link_quat_w.cpu().numpy()
   v=robot.data.root_link_lin_vel_b.cpu().numpy();yaw=robot.data.root_link_ang_vel_b[:,2].cpu().numpy();upright=1-2*(q[:,1]**2+q[:,2]**2)
   bad=(pos[:,2]<.06)|(upright<.5)|(~np.isfinite(pos).all(axis=1))
   fall_step[alive&bad]=step;alive&=~bad
   if step>=100:
    cm=cmd.cpu().numpy();xy=np.linalg.norm(v[:,:2]-cm[:,:2],axis=1);ye=np.abs(yaw-cm[:,2]);speed=np.linalg.norm(v[:,:2],axis=1)
    data.append(np.stack([v[:,0],v[:,1],yaw,xy,ye,speed,alive.astype(float)],axis=1))
    if step%10==0:
     for i in range(25):traces.append([iteration,SCENARIOS[i//5],i%5,(step-99)*.02,*cm[i],*v[i,:2],yaw[i],*pos[i],upright[i],bool(alive[i])])
   if step%400==0:print(iteration,'step',step,'alive',int(alive.sum()),flush=True)
  all_data=np.stack(data)
  for i in range(25):
   x=all_data[:,i];valid=x[:,6]>0;x=x[valid];s=SCENARIOS[i//5]
   row={'checkpoint':iteration,'scenario':s,'initial_condition':i%5,'fell':bool(fall_step[i]>=0),'first_fall_step':int(fall_step[i]),'survival_test_s':len(x)*.02}
   if len(x):
    for j,k in enumerate(['mean_vx','mean_vy','mean_yaw','xy_error','yaw_mae','planar_speed']):row[k]=float(x[:,j].mean())
    row['net_displacement']=float(np.linalg.norm(pos[i,:2]-start[i])) if alive[i] else None
    if s=='walk_then_stop':
     xx=all_data[500:,i];xx=xx[xx[:,6]>0];row['stop_planar_speed']=float(xx[:,5].mean()) if len(xx) else None
     row['stop_net_displacement']=float(np.linalg.norm(pos[i,:2]-stop[i])) if alive[i] else None
   rows.append(row)
  print('COMPLETE',iteration,flush=True)
  (OUT/'partial-results.json').write_text(json.dumps(rows,indent=2))
 report={'timestamp':time.time(),'protocol':PROFILE,'checkpoint_sha256':hashes,'initial_state_sha256':initials,'trials':rows}
 (OUT/'results.json').write_text(json.dumps(report,indent=2))
 with (OUT/'traces.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['checkpoint','scenario','initial_condition','time_s','cmd_vx','cmd_vy','cmd_yaw','vx','vy','yaw','world_x','world_y','world_z','upright','alive']);w.writerows(traces)
 np.save(OUT/'initial-states.npy',reference)
 env.close();print('SAVED',OUT,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['evaluate','play']);p.add_argument('--iteration',type=int,default=4999,choices=[4250,4500,4750,4999]);p.add_argument('--viewer',choices=['native','viser'],default='native');p.add_argument('--vx',type=float,default=0.15);p.add_argument('--vy',type=float,default=0.0);p.add_argument('--yaw',type=float,default=0.0);a=p.parse_args()
 if a.mode=='evaluate':evaluate()
 else:
  cfg=make_cfg(1)
  cfg.commands['twist'].ranges.lin_vel_x=(a.vx,a.vx)
  cfg.commands['twist'].ranges.lin_vel_y=(a.vy,a.vy)
  cfg.commands['twist'].ranges.ang_vel_z=(a.yaw,a.yaw)
  register_mjlab_task(TASK,cfg,cfg,load_rl_cfg(BASE),load_runner_cls(BASE))
  print('FIXED REVIEW: NO pushes, NO physics DR/noise/delay; zero head/body commands. No training.',flush=True)
  if a.viewer=='native':
   import mjlab.scripts.play as play_module
   from review_viewer import KeyboardReviewViewer
   KeyboardReviewViewer.checkpoint_label=f'model_{a.iteration}.pt'
   play_module.NativeMujocoViewer=KeyboardReviewViewer
  run_play(TASK,PlayConfig(checkpoint_file=str(RUN/f'model_{a.iteration}.pt'),num_envs=1,viewer=a.viewer))
if __name__=='__main__':main()