"""Regression: a floating-base joint ID is not its friction constraint DOF ID."""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import infer_policy as ip
from aligned_cpu_bam import AlignedBamController, dof_friction_force

class FloatingBaseFrictionTest(unittest.TestCase):
    def test_friction_is_removed_from_the_correct_dof_load(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
        <body pos="0 0 .2"><joint name="hinge" axis="0 1 0" frictionloss=".04"/>
        <geom type="capsule" size=".02 .1" mass=".1"/></body></body>
        </worldbody><actuator><motor name="motor" joint="hinge"/></actuator></mujoco>''')
        d=mujoco.MjData(m);bam=ip.load_bam_model(200,7.4,None)
        ctrl=AlignedBamController(bam,['motor'],m,d,vin_drop_gain=.1,vin_min=6)
        d.qvel[6]=.5;mujoco.mj_forward(m,d)
        self.assertEqual(m.joint('hinge').id,1)
        self.assertEqual(int(m.joint('hinge').dofadr[0]),6)
        forces=dof_friction_force(d,m.nv)
        self.assertGreater(abs(forces[6]),1e-8)
        self.assertEqual(forces[1],0.)
        # Independent reference: the constraint Jacobian maps forces to DOFs.
        jac=d.efc_J.reshape(d.nefc,m.nv)
        np.testing.assert_allclose(forces,jac.T@d.efc_force,atol=1e-12)
        expected=-d.qfrc_bias[6:7].copy()  # this fixture has no contacts or limits
        captured={}
        def capture(model,motor,external,velocity):
            captured['external']=external.copy()
            return np.array([.04]),0.
        with patch('aligned_cpu_bam.training_friction_budget',capture):ctrl.update()
        np.testing.assert_allclose(captured['external'],expected,atol=1e-12)
        self.assertEqual(bam.actuator.vin,7.4)
        self.assertTrue(np.isfinite(d.ctrl).all())

if __name__=='__main__':unittest.main()