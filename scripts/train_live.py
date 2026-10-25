"""Read-only Viser view of the actual PPO rollout environments.

Uses the standard mjlab training path; no evaluation environment or policy copy.
"""
import argparse
import copy
import json
import signal
import time
from pathlib import Path

import torch
import numpy as np
import mujoco
import viser
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.scripts import train
from mjlab.viewer.viser.scene import MjlabViserScene


class StopAtBoundary(Exception):
    pass


class LiveRunner(MjlabOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_server = viser.ViserServer(host="127.0.0.1", port=OPTIONS.port)
        self.live_server.gui.add_markdown(
            "# LIVE PPO TRAINING\nActual training environments, not checkpoint playback. "
            "All ducks share the policy being updated. Closing this page does not stop training."
        )
        self.live_info = self.live_server.gui.add_markdown("Initializing training...")
        self.live_scene = MjlabViserScene(self.live_server,
            copy.copy(self.env.unwrapped.sim.mj_model), self.env.num_envs)
        self.live_scene.camera_tracking_enabled = False
        self.live_scene.show_only_selected = False
        self.live_scene.site_groups_visible = [False] * 6
        self.live_scene.create_scene_gui(camera_distance=12.0, camera_elevation=65.0)
        self.live_server.gui.add_markdown(
            "**Compact display:** each duck is centered in its own tile so all 512 fit. "
            "Poses come from training; this display arrangement does not alter physics."
        )
        model = self.env.unwrapped.sim.mj_model
        trunks = [i for i in range(1, model.nbody)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("/")[-1] == "trunk_base"]
        assert len(trunks) == 1, f"Expected one robot trunk, found {trunks}"
        self.trunk_id = trunks[0]
        cols = int(np.ceil(np.sqrt(self.env.num_envs)))
        ids = np.arange(self.env.num_envs)
        self.tile_xy = np.column_stack((ids % cols, ids // cols)).astype(np.float32) * 0.6
        self.tile_xy -= self.tile_xy.mean(axis=0)
        self.live_frames = 0
        self.live_last = 0.0
        self.stop_requested = False
        self.status_path = Path(self.logger.log_dir) / "live-status.json"
        original_step = self.env.step

        def step(actions):
            result = original_step(actions)
            now = time.monotonic()
            if now - self.live_last >= 0.2:
                self.live_last = now
                # GPU readback and scene publication happen AFTER the real rollout step.
                # No simulation stepping, resets, action changes, or policy loading here.
                if self.live_frames == 0 or self.live_server.get_clients():
                    data = self.env.unwrapped.sim.data
                    positions = data.xpos.cpu().numpy().copy()
                    positions[:, :, :2] -= positions[:, self.trunk_id:self.trunk_id + 1, :2].copy()
                    positions[:, :, :2] += self.tile_xy[:, None, :]
                    self.live_scene.update_from_arrays(positions, data.xmat.cpu().numpy())
                    self.live_frames += 1
                state = dict(iteration=self.current_learning_iteration,
                    common_step_counter=self.env.unwrapped.common_step_counter,
                    num_envs=self.env.num_envs, frames=self.live_frames,
                    clients=len(self.live_server.get_clients()), timestamp=time.time())
                self.status_path.write_text(json.dumps(state))
                self.live_info.content = (
                    f"**Iteration:** {state['iteration']} / {OPTIONS.total}\n\n"
                    f"**Training ducks:** {state['num_envs']} | "
                    f"**Rollout step:** {state['common_step_counter']}\n\n"
                    "Display sampled at up to 5 FPS; physics and PPO continue independently."
                )
            return result

        self.env.step = step
        original_log = self.logger.log

        def log(**values):
            original_log(**values)
            if self.stop_requested:
                path = str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt")
                self.save(path)
                print(f"[LIVE] Saved complete iteration before stopping: {path}", flush=True)
                raise StopAtBoundary()

        self.logger.log = log
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        print(f"[LIVE] Actual {self.env.num_envs} training environments at http://localhost:{OPTIONS.port}", flush=True)

    def load(self, path, *args, **kwargs):
        info = super().load(path, *args, **kwargs)
        # Standard checkpoints label the iteration that has already completed.
        self.current_learning_iteration += 1
        print(f"[LIVE] Resumed at iteration {self.current_learning_iteration}; "
              f"curriculum steps={self.env.unwrapped.common_step_counter}; "
              f"optimizer states={len(self.alg.optimizer.state)}", flush=True)
        return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--run-name", default="rtx5060-walk")
    OPTIONS = parser.parse_args()
    cfg = train.TrainConfig.from_task("Mjlab-Velocity-Flat-MicroDuck")
    cfg.env.scene.num_envs = OPTIONS.envs
    cfg.agent.logger = "tensorboard"
    cfg.agent.run_name = OPTIONS.run_name
    cfg.agent.max_iterations = OPTIONS.total
    if OPTIONS.checkpoint:
        cp = OPTIONS.checkpoint.resolve()
        state = torch.load(cp, map_location="cpu", weights_only=False)
        for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"):
            assert key in state, f"Checkpoint missing {key}"
        assert "common_step_counter" in state["infos"]["env_state"]
        cfg.agent.resume = True
        cfg.agent.load_run = cp.parent.name
        cfg.agent.load_checkpoint = cp.name
        cfg.agent.max_iterations = OPTIONS.total - int(state["iter"]) - 1
        assert cfg.agent.max_iterations > 0, "This run has already reached the requested total"
        del state
    train.load_runner_cls = lambda task_id: LiveRunner
    try:
        train.launch_training("Mjlab-Velocity-Flat-MicroDuck", cfg)
    except StopAtBoundary:
        print("[LIVE] Training stopped safely at an iteration boundary.", flush=True)