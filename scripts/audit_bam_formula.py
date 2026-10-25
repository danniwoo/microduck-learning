import json,numpy as np,torch
from types import SimpleNamespace
from pathlib import Path
from bam.mjlab import BamActuator
import infer_policy as ip
bam=ip.load_bam_model(200,7.4,None)
motor=np.array([.1,.2,.2,-.2]);external=np.array([.05,.1,-.1,.1]);vel=np.zeros(4)
orig,_=bam.compute_frictions(motor,external,vel)
gpu=BamActuator._compute_friction_budget(SimpleNamespace(_bam_model=bam),torch.tensor(motor),torch.tensor(external),torch.ones(4,dtype=torch.float64)).numpy()
r={'motor_torque':motor.tolist(),'external_torque':external.tolist(),'speed':vel.tolist(),'cpu_friction_budget':orig.tolist(),'gpu_friction_budget':gpu.tolist(),'difference':(gpu-orig).tolist(),'note':'Same M6 parameters and same numerical inputs; GPU quadratic term lacks the CPU opposite-sign gate.'}
Path('logs/sim-parity-audit/bam-formula-proof.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))