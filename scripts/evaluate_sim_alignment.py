"""Fixed, inference-only regression after CPU deployment alignment.

Run from project root: python scripts/evaluate_sim_alignment.py --backend warp
then --backend cpu. No optimizer, training or checkpoint writes.
"""
import argparse, contextlib, copy, csv, hashlib, io, json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch, mujoco
import infer_policy as ip
from checkpoint_review import (make_cfg, BASE, RUN, ROOT, ManagerBasedRlEnv,
    RslRlVecEnvWrapper, load_rl_cfg, load_runner_cls, MjlabOnPolicyRunner,
    configure_torch_backends)
from aligned_cpu_bam import AlignedBamController

SCENARIOS = [('stand', [0.,0.,0.]), ('forward', [.3,0.,0.]), ('turn', [0.,0.,.75])]
SEEDS = 3
STEPS, SETTLE, DT = 1600, 100, .02
CHECKPOINT = RUN/'model_4500.pt'
ONNX = RUN/'model_4500_eval.onnx'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def initials():
    return np.array([[0,0,.125,1,0,0,0,*(ip.DEFAULT_POSE + np.random.default_rng(i%SEEDS).uniform(-.01,.01,14))]
                     for i in range(len(SCENARIOS)*SEEDS)], dtype=np.float32)

def measure(q, v):
    vxvy = np.array([ip.PolicyInference.quat_rotate_inverse(None, a[3:7], b[:3])[:2] for a,b in zip(q,v)])
    upright = 1-2*(q[:,4]**2+q[:,5]**2)
    fallen = (q[:,2]<.06)|(upright<.5)|(~np.isfinite(q).all(axis=1))
    return vxvy, v[:,5], fallen

class Recorder:
    def __init__(self,n):
        self.fallen=np.zeros(n,bool);self.data=[];self.positions=[];self.traces=[]
    def add(self, step, q, v, commands):
        xyvel,yaw,bad=measure(q,v);self.fallen|=bad
        if step==SETTLE-1:self.start=q[:,:2].copy()
        if step>=SETTLE:
            self.data.append(np.column_stack([xyvel,yaw,np.linalg.norm(xyvel-commands[:,:2],axis=1),np.abs(yaw-commands[:,2])]))
            self.positions.append(q[:,:2].copy())
            if step%10==0:
                for i in range(len(q)):self.traces.append([i,(step-SETTLE+1)*DT,*commands[i],*q[i,:3],*xyvel[i],yaw[i],bool(self.fallen[i])])
    def save(self,out,name,indices=None):
        indices=list(range(len(self.fallen))) if indices is None else indices
        a=np.array(self.data);pos=np.array(self.positions)
        path=np.concatenate([self.start[None],pos],axis=0)
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=2).sum(axis=0)
        rows=[]
        for i,index in enumerate(indices):
            rows.append({'backend':name,'scenario':SCENARIOS[index//SEEDS][0],'seed':index%SEEDS,'fell':bool(self.fallen[i]),
                'mean_vx':float(a[:,i,0].mean()),'mean_vy':float(a[:,i,1].mean()),'mean_yaw':float(a[:,i,2].mean()),
                'xy_error':float(a[:,i,3].mean()),'yaw_mae':float(a[:,i,4].mean()),
                'net_cm':float(np.linalg.norm(pos[-1,i]-self.start[i])*100),
                'max_offset_cm':float(np.linalg.norm(pos[:,i]-self.start[i],axis=1).max()*100),'path_cm':float(lengths[i]*100)})
        return rows

def gpu(out):
    configure_torch_backends();q0=initials();n=len(q0)
    np.save(out/'initial-qpos.npy',q0)
    session=ip.create_policy_session(str(ONNX));input_name=session.get_inputs()[0].name
    all_rows=[]
    # Fresh environments prevent mutable actuator/physics state leaking between runs.
    for engine in ['warp_pt','warp_onnx']:
        env=ManagerBasedRlEnv(cfg=make_cfg(n),device='cuda:0')
        mujoco.mj_saveModel(env.sim.mj_model,str(out/'shared-model.mjb'))
        ac=load_rl_cfg(BASE);wrap=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
        runner=(load_runner_cls(BASE) or MjlabOnPolicyRunner)(wrap,asdict(ac),device='cuda:0')
        runner.load(str(CHECKPOINT),load_cfg={'actor':True},strict=True,map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        env.reset(seed=20261025);env.sim.data.qpos[:]=torch.tensor(q0,device='cuda:0');env.sim.data.qvel[:]=0;env.sim.forward()
        commands=np.zeros((n,3),np.float32);rec=Recorder(n)
        for step in range(STEPS):
            if step==SETTLE:commands[:]=np.repeat(np.array([c for _,c in SCENARIOS],np.float32),SEEDS,axis=0)
            env.command_manager.get_term('twist').command[:]=torch.tensor(commands,device='cuda:0')
            env.observation_manager.compute(update_history=True);obs=wrap.get_observations()
            assert np.array_equal(obs['actor'][:,-13:-10].cpu().numpy(),commands)
            assert torch.count_nonzero(obs['actor'][:,-10:])==0
            with torch.inference_mode():
                if engine=='warp_pt':action=policy(obs)
                else:
                    x=obs['actor'].cpu().numpy();actions=np.concatenate([session.run(None,{input_name:y[None]})[0] for y in x])
                    action=torch.tensor(actions,device='cuda:0')
                wrap.step(action)
            rec.add(step,env.sim.data.qpos.cpu().numpy(),env.sim.data.qvel.cpu().numpy(),commands)
            if step%400==0:print(engine,step,flush=True)
        rows=rec.save(out,engine);all_rows+=rows
        (out/f'{engine}.json').write_text(json.dumps(rows,indent=2))
        save_trace(out,engine,rec.traces);env.close()
    return all_rows

def save_trace(out,name,traces):
    with (out/f'{name}-traces.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['trial','time','cmd_vx','cmd_vy','cmd_yaw','x','y','z','vx','vy','yaw','fallen']);w.writerows(traces)

def cpu(out):
    ref=mujoco.MjModel.from_binary_path(str(out/'shared-model.mjb'));q0=np.load(out/'initial-qpos.npy');all_rows=[]
    for engine in ['cpu_original','cpu_aligned']:
        rows=[];traces=[]
        for i,q in enumerate(q0):
            with contextlib.redirect_stdout(io.StringIO()):
                bam=ip.load_bam_model(200,7.4,None)
                if engine=='cpu_original':m,d,ctrl,_=ip.load_mujoco_with_bam(ip.MICRODUCK_XML,bam,.005,.1,6.)
                else:
                    m=copy.copy(ref);d=mujoco.MjData(m)
                    ctrl=AlignedBamController(bam,[m.actuator(j).name for j in range(m.nu)],m,d,vin_drop_gain=.1,vin_min=6.)
                p=ip.PolicyInference(m,d,walking_onnx_path=str(ONNX),bam_ctrl=ctrl,new_cmd_obs=True,use_projected_gravity=True)
                if engine=='cpu_aligned':
                    p.imu_ang_vel_id=m.sensor('robot/imu_ang_vel').id;p.trunk_base_id=m.body('robot/trunk_base').id;p._trunk_qpos_adr=0
                assert m.nq==21 and m.nv==20 and m.nu==14
                assert [m.joint(m.actuator_trnid[j,0]).name.removeprefix('robot/') for j in range(m.nu)]==[ref.joint(ref.actuator_trnid[j,0]).name.removeprefix('robot/') for j in range(ref.nu)]
                d.qpos[:]=q;d.qvel[:]=0;ctrl.reset(d.qpos);p.set_position_targets(p.default_pose);mujoco.mj_forward(m,d);p.set_vel_cmd(0,0,0)
            rec=Recorder(1);command=np.zeros((1,3),np.float32)
            for step in range(STEPS):
                if step==SETTLE:
                    command[0]=SCENARIOS[i//SEEDS][1]
                    with contextlib.redirect_stdout(io.StringIO()):p.set_vel_cmd(*command[0])
                assert np.array_equal(p.get_observations()[-13:-10],command[0])
                p.apply_action(p.infer())
                for _ in range(4):ctrl.update();mujoco.mj_step(m,d)
                # Same placement as mjlab: refresh derived observations/forces after decimation.
                if engine=='cpu_aligned':mujoco.mj_forward(m,d)
                rec.add(step,d.qpos[None].copy(),d.qvel[None].copy(),command)
            row=rec.save(out,engine,[i])[0];rows.append(row);print(json.dumps(row),flush=True)
            traces.extend([[i,*r[1:]] for r in rec.traces])
            (out/f'{engine}.json').write_text(json.dumps(rows,indent=2))
        save_trace(out,engine,traces);all_rows+=rows
    return all_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['warp','cpu'],required=True)
    p.add_argument('--out',type=Path,default=ROOT/'logs/sim-alignment-4500-20261026');args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    # Refuse accidental overwriting of completed evaluation evidence.
    target=out/('warp_pt.json' if args.backend=='warp' else 'cpu_original.json')
    if target.exists():raise FileExistsError(f'Use a new --out directory: {target}')
    before={str(x):digest(x) for x in [CHECKPOINT,ONNX]}
    rows=gpu(out) if args.backend=='warp' else cpu(out)
    after={str(x):digest(x) for x in [CHECKPOINT,ONNX]};assert before==after
    protocol={'backend':args.backend,'checkpoint':4500,'training':False,'sha256_before':before,'sha256_after':after,
        'initial_state_sha256':digest(out/'initial-qpos.npy'),'scenarios':SCENARIOS,'seeds':list(range(SEEDS)),
        'settle_seconds':2,'test_seconds':30,'physics_dt':.005,'control_dt':DT,'pushes':False,'dr':False,'noise':False,
        'cpu_aligned':'shared compiled Warp model, corrected friction DOF indexing, installed GPU BAM formula, forward after decimation',
        'limitation':'Training-compatible friction is not a physical calibration. Same-model sim-to-sim check, not new training or official benchmark. Three initial conditions, no statistical population claim.'}
    (out/f'{args.backend}-protocol.json').write_text(json.dumps(protocol,indent=2));print('DONE',out,flush=True)
if __name__=='__main__':main()