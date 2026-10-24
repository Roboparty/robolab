# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.envs import DirectRLEnv
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import CircularBuffer

from .base_config import BaseEnvCfg
from .mdp.observations import feet_clearance, flatten_history, height_scan
from .mdp.rewards import _contact_forces


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg
    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_manager = RewardManager(self.cfg.reward, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        self.left_feet_scanner_cfg = SceneEntityCfg("left_feet_scanner")
        self.right_feet_scanner_cfg = SceneEntityCfg("right_feet_scanner")

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)

    def init_buffers(self):
        self.extras = {}

        self.episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.torch.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = CircularBuffer(
            max_len=self.cfg.robot.action_history_length, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.feet_body_names,
            preserve_order=self.cfg.robot.preserve_feet_order,
        )
        self.feet_cfg.resolve(self.scene)
        self.feet_asset_cfg = SceneEntityCfg(
            "robot", body_names=self.cfg.robot.feet_body_names,
            preserve_order=self.cfg.robot.preserve_feet_order,
        )
        self.feet_asset_cfg.resolve(self.scene)
        self.previous_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.init_obs_buffer()

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene_context.height_scanner.enable_height_scan:
                scan = height_scan(self)
                height_scan_noise_vec = torch.zeros_like(scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def _actor_observation_terms(self):
        data = self.robot.data
        return [
            data.root_ang_vel_b.torch * self.obs_scales.ang_vel,
            data.projected_gravity_b.torch * self.obs_scales.projected_gravity,
            self.command_generator.command * self.obs_scales.commands,
            (data.joint_pos.torch - data.default_joint_pos.torch) * self.obs_scales.joint_pos,
            (data.joint_vel.torch - data.default_joint_vel.torch) * self.obs_scales.joint_vel,
            self.action_buffer.buffer[:, -1] * self.obs_scales.actions,
        ]

    def compute_current_observations(self):
        actor_terms = self._actor_observation_terms()
        self.actor_obs_term_sizes = [term.shape[-1] for term in actor_terms]
        actor = torch.cat(actor_terms, dim=-1)

        force_type = self.cfg.robot.contact_force_type
        forces = _contact_forces(self.contact_sensor, force_type, history=True)
        contacts = forces[:, :, self.feet_cfg.body_ids].norm(dim=-1).amax(dim=1) > 1.0
        sensors = tuple(
            cfg for cfg in (self.left_feet_scanner_cfg, self.right_feet_scanner_cfg)
            if cfg.name in self.scene.sensors
        )
        critic_terms = [
            self.robot.data.root_lin_vel_b.torch * self.obs_scales.lin_vel,
            contacts.float(),
            _contact_forces(self.contact_sensor, force_type)[:, self.feet_cfg.body_ids].flatten(1),
            self.contact_sensor.data.current_air_time.torch[:, self.feet_cfg.body_ids].flatten(1),
            feet_clearance(
                self, self.feet_asset_cfg, sensors,
                replace_invalid_hits=self.cfg.robot.replace_invalid_height_hits,
            ),
            self.robot.data.joint_acc.torch,
            self.robot.actuators.applied_effort.torch,
        ]
        self.critic_obs_term_sizes = self.actor_obs_term_sizes + [term.shape[-1] for term in critic_terms]
        return actor, torch.cat([actor, *critic_terms], dim=-1)

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.is_rendering
        num_steps = 1 if self._physics_handles_decimation else self.cfg.decimation
        elapsed_dt = self.step_dt if self._physics_handles_decimation else self.physics_dt
        for _ in range(num_steps):
            self._sim_step_counter += self.cfg.decimation if self._physics_handles_decimation else 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=elapsed_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.sim.consume_reset_request():
            pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            pending[reset_env_ids] = False
            manual_reset_ids = pending.nonzero(as_tuple=False).squeeze(-1)
            if len(manual_reset_ids) > 0:
                self.reset_terminated[manual_reset_ids] = True
                self._reset_idx(manual_reset_ids)
        for recorder in self.video_recorders:
            recorder.step()

        self.obs_buf = self._get_observations()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w.torch[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.cfg.scene_context.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]

    def _append_action(self, actions):
        # CircularBuffer backfills the first push; some rewards need zero history after reset.
        self.previous_action.copy_(self.action_buffer.buffer[:, -1])
        self.action_buffer.append(actions)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._append_action(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos.torch

    def _apply_action(self) -> None:
        self.robot.actuators.target_command.set_position_index(value=self.actions)

    def _get_observations(self):
        actor, critic = self.compute_current_observations()
        actor_sizes = self.actor_obs_term_sizes.copy()
        critic_sizes = self.critic_obs_term_sizes.copy()
        if self.add_noise:
            actor += (2 * torch.rand_like(actor) - 1) * self.noise_scale_vec

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            scan = height_scan(self)
            critic = torch.cat([critic, scan], dim=-1)
            critic_sizes.append(scan.shape[-1])
            if self.add_noise and self.cfg.noise.add_height_scan_noise:
                scan += (2 * torch.rand_like(scan) - 1) * self.height_scan_noise_vec
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                actor = torch.cat([actor, scan], dim=-1)
                actor_sizes.append(scan.shape[-1])

        self.actor_obs_buffer.append(actor)
        self.critic_obs_buffer.append(critic)
        layout = self.cfg.robot.observation_history_layout
        return {
            "policy": flatten_history(self.actor_obs_buffer.buffer, actor_sizes, layout).clamp(-self.clip_obs, self.clip_obs),
            "critic": flatten_history(self.critic_obs_buffer.buffer, critic_sizes, layout).clamp(-self.clip_obs, self.clip_obs),
        }

    def _get_rewards(self):
        return self.reward_manager.compute(dt=self.step_dt)
    
    def _get_dones(self):
        net_contact_forces = self.contact_sensor.data.net_normal_forces_w_history.torch
        terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.robot.ground_contact_sensor_names:
            for name in self.cfg.robot.ground_contact_sensor_names:
                history = self.scene[name].data.normal_force_matrix_w_history
                if history is None or history.torch.shape[-2] == 0:
                    raise RuntimeError(f"Ground contact filter is empty: {name}")
                terminated_buf |= (history.torch.norm(dim=-1) > 1.0).flatten(1).any(dim=1)
        elif self.cfg.robot.terminate_contacts_body_names is not None:
            terminated_buf = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
        if self.cfg.robot.terminate_base_orientation is not None:
            gravity_z = torch.clamp(-self.robot.data.projected_gravity_b.torch[:, 2], -1.0, 1.0)
            terminated_buf |= torch.acos(gravity_z) > self.cfg.robot.terminate_base_orientation
        if self.cfg.robot.terminate_base_height is not None:
            terminated_buf |= self.robot.data.root_pos_w.torch[:, 2] < self.cfg.robot.terminate_base_height
        time_out_buf = self.episode_length_buf >= self.episode_length
        return terminated_buf, time_out_buf

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if len(env_ids) == 0:
            return
        
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner.reset(env_ids)

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.reset_time_outs

        self.command_generator.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.previous_action[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()
