import torch,json,numpy as np,mujoco,contextlib,io
from pathlib import Path
from tensordict import TensorDict
from rsl_rl.models import MLPModel
import infer_policy as ip
O=Path('logs/sim-parity-audit');R=Path('logs/rsl_rl/velocity/2026-10-25_18-12-33_rtx5060-walk')
ck=torch.load(R/'model_4500.pt',map_location='cpu',weights_only=False)
print('keys',list(ck));print('actor keys',list(ck.get('actor_state_dict',{})))
x=np.load(O/'sample-observations.npy')
s=ip.create_policy_session(str(R/'model_4500_eval.onnx'))
# Deterministic network with the exact checkpoint normalizer and MLP.
p=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=[512,256,128],activation='elu',obs_normalization=True)
state=ck['actor_state_dict'];weights={k:v for k,v in state.items() if not k.startswith('distribution.')}
p.load_state_dict(weights,strict=True);p.eval()
with torch.inference_mode():y=p(TensorDict({'actor':torch.tensor(x)},batch_size=[len(x)])).numpy()
z=np.concatenate([s.run(None,{s.get_inputs()[0].name:a[None]})[0] for a in x])
report={'samples':len(x),'cpu_torch_vs_onnx_max_abs':float(np.max(np.abs(y-z))),'cpu_torch_vs_onnx_mean_abs':float(np.mean(np.abs(y-z)))}
# A minimal constraint with a freejoint and one hinge establishes efc_id indexing.
spec='''<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/><body pos="0 0 .2"><joint name="hinge" axis="0 1 0" frictionloss=".1"/><geom type="capsule" size=".02 .1" mass=".1"/></body></body></worldbody></mujoco>'''
m=mujoco.MjModel.from_xml_string(spec);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
report['minimal_friction_constraint']={'hinge_joint_id':m.joint('hinge').id,'hinge_dof_id':int(m.joint('hinge').dofadr[0]),'efc_friction_ids':d.efc_id[d.efc_type==mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF].tolist()}
(O/'export-and-index-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))