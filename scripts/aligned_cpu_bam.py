"""Opt-in CPU evaluation compatibility with the installed training BAM rules.

Does not patch site-packages or change the original demo. Matching training's M6
rule is a compatibility choice, not a claim that it is the physically correct rule.
Controller equations adapted from bam.mujoco (Apache-2.0, Marc Duclusaud and
Gregoire Passault); friction budget delegates to the installed GPU implementation.
"""
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import mjlab_microduck.tasks  # initialize task registry before importing bam.mjlab
from bam.mujoco import MujocoController
from bam.mjlab import BamActuator


def dof_friction_force(data, nv):
    result = np.zeros(nv)
    mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    # MuJoCo efc_id identifies a velocity DOF, not a joint. A free joint has 6 DOFs.
    np.add.at(result, data.efc_id[mask], data.efc_force[mask])
    return result


def training_friction_budget(bam_model, motor, external, velocity):
    # Reuse the training implementation instead of maintaining a third formula.
    vel = torch.as_tensor(velocity, dtype=torch.float64)
    coefficient = torch.exp(-(vel.abs() / bam_model.dtheta_stribeck.value) ** bam_model.alpha.value)
    friction = BamActuator._compute_friction_budget(
        SimpleNamespace(_bam_model=bam_model),
        torch.as_tensor(motor, dtype=torch.float64),
        torch.as_tensor(external, dtype=torch.float64), coefficient,
    )
    return friction.numpy(), bam_model.friction_viscous.value


class AlignedBamController(MujocoController):
    """Correct DOF indexing and use the installed training friction formula."""
    def reset(self, qpos):
        super().reset(qpos)
        self.last_ts = self.mujoco_data.time

    def update(self):
        data, model = self.mujoco_data, self.mujoco_model
        q, velocity = data.qpos[self.qpos_indexes], data.qvel[self.dof_indexes]
        actuator = self.model.actuator
        original_vin = actuator.vin
        if self.vin_drop_gain is not None:
            voltage = original_vin - self.vin_drop_gain * np.abs(self._prev_motor_torque).sum()
            actuator.vin = max(voltage, self.vin_min) if self.vin_min is not None else voltage
        dt = data.time - self.last_ts
        self.last_ts = data.time
        try:
            control = actuator.compute_control(self.q_target, q, velocity, dt)
            torque = actuator.compute_torque(control, True, q, velocity)
        finally:
            actuator.vin = original_vin
        self._prev_motor_torque = np.atleast_1d(torque).copy()
        data.ctrl[self.act_indexes] = torque
        external = (-data.qfrc_bias + data.qfrc_constraint - dof_friction_force(data, model.nv))[self.dof_indexes]
        friction, damping = training_friction_budget(
            self.model, data.qfrc_actuator[self.dof_indexes], external, velocity)
        model.dof_frictionloss[self.dof_indexes] = friction
        model.dof_damping[self.dof_indexes] = damping