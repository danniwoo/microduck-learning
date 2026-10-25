"""Keyboard controls for the fixed checkpoint viewer (never used in evaluation)."""
import json
from pathlib import Path
import numpy as np
import mujoco
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction

class KeyboardReviewViewer(NativeMujocoViewer):
    checkpoint_label='checkpoint'
    def __init__(self,*args,**kwargs):
        kwargs['enable_perturbations']=False
        super().__init__(*args,**kwargs)
    def _safe_key_callback(self,key):
        # The GLFW thread only queues; mutation happens on the simulation thread.
        if key in (265,264,263,262,32):
            self.request_action('DUCK_COMMAND',('duck_command',key))
            return
        if key==84: # T pauses, Space means zero command in this custom viewer
            self.request_toggle_pause()
            return
        super()._safe_key_callback(key)
    def _handle_custom_action(self,action,payload):
        if action==ViewerAction.CUSTOM and isinstance(payload,tuple) and payload[0]=='duck_command':
            key=payload[1];env=self.env.unwrapped
            term=env.command_manager.get_term('twist')
            cmd=term.command[0].detach().cpu().numpy().copy()
            if key==265:cmd[0]=np.clip(cmd[0]+.05,-.4,.4)
            elif key==264:cmd[0]=np.clip(cmd[0]-.05,-.4,.4)
            elif key==263:cmd[2]=np.clip(cmd[2]+.25,-1,1)
            elif key==262:cmd[2]=np.clip(cmd[2]-.25,-1,1)
            else:cmd[:]=0
            term.command[:]=term.command.new_tensor(cmd)
            term.vel_command_w[:]=term.command
            # Prevent a future resample/reset from restoring an old command.
            term.cfg.ranges.lin_vel_x=(float(cmd[0]),float(cmd[0]))
            term.cfg.ranges.lin_vel_y=(float(cmd[1]),float(cmd[1]))
            term.cfg.ranges.ang_vel_z=(float(cmd[2]),float(cmd[2]))
            obs=env.observation_manager.compute(update_history=True)
            assert np.allclose(obs['actor'][0,-13:-10].detach().cpu().numpy(),cmd)
            status={'checkpoint':self.checkpoint_label,'command':cmd.tolist(),'random_pushes':False,'observation_command':obs['actor'][0,-13:-10].detach().cpu().tolist()}
            Path('logs/review-keyboard-status.json').write_text(json.dumps(status))
            print('COMMAND',status,flush=True)
            return True
        return super()._handle_custom_action(action,payload)
    def _set_status_overlay(self,viewer):
        s=self.get_status()
        cmd=self.env.unwrapped.command_manager.get_command('twist')[0].detach().cpu().tolist()
        left='Checkpoint\nStatus\nPush / noise / DR\nForward vx\nTurn yaw\nMove\nTurn\nStop / Pause\nReset'
        right=f'{self.checkpoint_label}\n'+('PAUSED' if s.paused else 'RUNNING')+f'\nOFF\n{cmd[0]:+.2f} m/s\n{cmd[2]:+.2f} rad/s\nUp / Down\nLeft / Right\nSpace / T\nEnter'
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,mujoco.mjtGridPos.mjGRID_TOPLEFT.value,left,right))