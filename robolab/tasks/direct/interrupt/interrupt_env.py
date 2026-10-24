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

from isaaclab.markers import VisualizationMarkers
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import torch

from robolab.tasks.direct.base.base_env import BaseEnv


def guard_interrupt_targets(targets, lower_bound, upper_bound, cfg, max_outward_roll=None):
    """Constrain arm targets using the sampling bounds and roll-dependent clearance.

    Canonical coordinates have positive outward roll, inward yaw and elbow
    flexion on both arms. Small outward roll restricts yaw, backward pitch
    and elbow flexion. Forward-raised arms allow more yaw; overhead roll
    restricts negative yaw more strongly when the elbow bends in reverse.
    With a curriculum roll ceiling, first move roll outward within that
    window to avoid turning a small perturbation into a large elbow change.
    """
    if targets.ndim != 2 or targets.shape[-1] != 8:
        raise ValueError("Arm interruptions require a [batch, 8] target.")
    signs = targets.new_tensor(cfg.arm_joint_signs)
    clearance_cfg = cfg.arm_clearance
    canonical = (targets * signs).reshape(-1, 2, 4)
    bound_a = (lower_bound * signs).reshape(-1, 2, 4)
    bound_b = (upper_bound * signs).reshape(-1, 2, 4)
    lower = torch.minimum(bound_a, bound_b)
    upper = torch.maximum(bound_a, bound_b)
    canonical = canonical.clamp(lower, upper)
    pitch, roll, yaw, elbow = canonical.unbind(dim=-1)
    forward_yaw_margin = clearance_cfg.forward_yaw_margin * torch.sin(-pitch).clamp(min=0.0)
    if max_outward_roll is not None:
        if max_outward_roll.shape != roll.shape:
            raise ValueError("Arm roll ceilings must have shape [batch, 2].")
        no_requirement = lower[..., 1]
        pitch_roll = torch.where(
            pitch > clearance_cfg.backward_pitch_limit,
            clearance_cfg.roll_clearance + (pitch - clearance_cfg.backward_pitch_limit) / clearance_cfg.pitch_per_roll,
            no_requirement,
        )
        yaw_roll = torch.where(
            yaw > 0.0,
            torch.where(
                yaw > forward_yaw_margin,
                clearance_cfg.roll_clearance + yaw - forward_yaw_margin,
                clearance_cfg.roll_clearance * yaw / forward_yaw_margin.clamp(min=1e-6),
            ),
            torch.where(yaw < -forward_yaw_margin, (-yaw - forward_yaw_margin) / clearance_cfg.outward_yaw_per_roll, no_requirement),
        )
        flexion_roll = torch.where(
            elbow <= clearance_cfg.low_roll_elbow_flexion,
            clearance_cfg.roll_clearance * torch.sqrt(elbow.clamp(min=0.0) / clearance_cfg.low_roll_elbow_flexion),
            clearance_cfg.roll_clearance + (elbow - clearance_cfg.low_roll_elbow_flexion) / clearance_cfg.elbow_flexion_per_roll,
        )
        elbow_roll = torch.where(
            elbow > 0.0,
            flexion_roll,
            torch.where(
                elbow < 0.0,
                clearance_cfg.reverse_elbow_roll_start
                - elbow * (clearance_cfg.reverse_elbow_roll_span / clearance_cfg.reverse_elbow_range),
                no_requirement,
            ),
        )
        required_roll = torch.maximum(torch.maximum(pitch_roll, yaw_roll), elbow_roll)
        roll_ceiling = max_outward_roll.clamp(lower[..., 1], upper[..., 1])
        roll = torch.maximum(roll, torch.minimum(required_roll, roll_ceiling))
    outward = roll.clamp(min=0.0)
    clearance = (roll - clearance_cfg.roll_clearance).clamp(min=0.0)
    # Preserve the low-roll clearance slopes and take range caps from sampling bounds.
    pitch_max = (clearance_cfg.backward_pitch_limit + clearance_cfg.pitch_per_roll * clearance).clamp(max=upper[..., 0])
    elbow_max = (
        clearance_cfg.low_roll_elbow_flexion * (outward / clearance_cfg.roll_clearance).clamp(max=1.0).square()
        + clearance_cfg.elbow_flexion_per_roll * clearance
    ).clamp(max=upper[..., 3])
    elbow_min = (
        -clearance_cfg.reverse_elbow_range
        * ((roll - clearance_cfg.reverse_elbow_roll_start) / clearance_cfg.reverse_elbow_roll_span).clamp(min=0.0)
    ).clamp(min=lower[..., 3])
    elbow = torch.maximum(torch.minimum(elbow, elbow_max), elbow_min)
    yaw_min = (-clearance_cfg.outward_yaw_per_roll * outward - forward_yaw_margin).clamp(min=lower[..., 2])
    # Reverse elbow bends fold the wrist toward the torso in overhead poses.
    overhead_yaw_margin = (
        clearance_cfg.overhead_yaw_margin * (1.0 + elbow / clearance_cfg.reverse_elbow_yaw_scale).clamp(0.0, 1.0)
    )
    yaw_min = torch.maximum(
        yaw_min, ((roll - clearance_cfg.overhead_roll_start) * clearance_cfg.overhead_yaw_slope).clamp(max=0.0) - overhead_yaw_margin
    )
    # Retain inward-yaw protection when both arms are brought toward the midline.
    inward_yaw_margin = forward_yaw_margin * (outward / clearance_cfg.roll_clearance).clamp(max=1.0)
    yaw_max = (clearance + inward_yaw_margin).clamp(max=upper[..., 2])
    canonical = torch.stack(
        (
            torch.minimum(pitch, pitch_max),
            roll,
            torch.maximum(torch.minimum(yaw, yaw_max), yaw_min),
            elbow,
        ),
        dim=-1,
    )
    return canonical.reshape(-1, 8) * signs

class InterruptEnv(BaseEnv):
    def init_buffers(self):
        if self.cfg.interrupt.use_interrupt:
            self.interrupt_joint_cfg = SceneEntityCfg(name="robot", joint_names=self.cfg.interrupt.interrupt_joint_names, preserve_order=True)
            self.interrupt_joint_cfg.resolve(self.scene)
            all_indices = torch.arange(self.num_envs, device=self.device)
            perm = torch.randperm(self.num_envs, device=self.device)
            self.interrupt_indices = all_indices[perm[:int(self.num_envs * self.cfg.interrupt.interrupt_ratio)]]
            self.interrupt_mode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            self.interrupt_mode_mask[self.interrupt_indices] = True
            self.interrupt_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            self.interrupt_actions = torch.zeros(self.num_envs, len(self.cfg.interrupt.interrupt_joint_names), dtype=torch.float, device=self.device, requires_grad=False)
            self.interrupt_scale = torch.tensor(self.cfg.interrupt.interrupt_scale, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            self.interrupt_lower_bound = torch.tensor(self.cfg.interrupt.interrupt_lower_bound, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            self.interrupt_rad_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.interrupt_vis = VisualizationMarkers(self.cfg.interrupt_vis)
            self.interrupt_vis.set_visibility(True)
        
        super().init_buffers()

    def _actor_observation_terms(self):
        terms = super()._actor_observation_terms()
        if self.cfg.interrupt.use_interrupt:
            terms.append(self.interrupt_mask.unsqueeze(1))
        return terms

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)
        
        if self.cfg.interrupt.use_interrupt:
            tmp_interrupt_mask = self.interrupt_mask[env_ids]
            tmp_interrupt_mode_mask = self.interrupt_mode_mask[env_ids]
            # Clear reset state before assigning the next mode.  Curriculum
            # accounting still uses the pre-reset active environments.
            self.interrupt_mask[env_ids] = False
            self.update_interrupt_levels(
                env_ids[tmp_interrupt_mode_mask],
                env_ids[tmp_interrupt_mode_mask & tmp_interrupt_mask],
            )
            self.extras["log"].update({"Curriculum/interrupt_levels":  torch.mean(self.interrupt_rad_curriculum[self.interrupt_indices])})

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
        if self.cfg.interrupt.use_interrupt:
            self._refresh_interrupt_actions(env_ids[self.interrupt_mode_mask[env_ids]])

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
        if self.cfg.interrupt.use_interrupt:
            if self.common_step_counter % self.cfg.interrupt.interrupt_update_step == 0:
                self.interrupt_actions = self.uniform_interrupt_resample() / self.cfg.robot.action_scale
                self.common_step_counter %= self.cfg.interrupt.interrupt_update_step
            if is_rendering:
                robot_positions = self.robot.data.root_pos_w.torch
                marker_positions = robot_positions + torch.tensor([0.0, 0.0, 0.8], device=self.device)
                interrupt_list = (~self.interrupt_mask).int().cpu().numpy().tolist()
                self.interrupt_vis.visualize(marker_indices=interrupt_list, translations=marker_positions)

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
    
    def update_interrupt_levels(self, env_ids, interrupt_env_ids):
        if len(env_ids) == 0:
            return
        distance = torch.norm(self.robot.data.root_pos_w.torch[interrupt_env_ids, :2] - self.scene.env_origins[interrupt_env_ids, :2], dim=1)
        pass_distance = self.cfg.interrupt.curriculum_pass_distance
        if pass_distance is None:
            pass_distance = self.cfg.scene_context.terrain_generator.size[0] / 2
        curr_is_pass = distance > pass_distance
        curr_is_down = (
            distance < torch.norm(self.command_generator.command[interrupt_env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        curr_is_down *= ~curr_is_pass

        self.interrupt_rad_curriculum[interrupt_env_ids] = torch.where(
            curr_is_down,
            (self.interrupt_rad_curriculum[interrupt_env_ids] - 0.05).clip(min=0),
            torch.where(
                curr_is_pass,
                (self.interrupt_rad_curriculum[interrupt_env_ids] + 0.05).clip(max=self.cfg.interrupt.max_curriculum),
                self.interrupt_rad_curriculum[interrupt_env_ids]
            )
        )

        self.interrupt_mask[env_ids] = torch.rand(len(env_ids), device=self.device) <= 0.5

    def _refresh_interrupt_actions(self, env_ids):
        # Read the randomized new episode pose after writing and refreshing
        # simulator state, with the same action units as periodic resampling.
        self.interrupt_actions[env_ids] = (
            self.robot.data.joint_pos.torch[env_ids][:, self.interrupt_joint_cfg.joint_ids]
            - self.robot.data.default_joint_pos.torch[env_ids][:, self.interrupt_joint_cfg.joint_ids]
        ) / self.action_scale

    def uniform_interrupt_resample(self):
        cfg = self.cfg.interrupt
        joint_ids = self.interrupt_joint_cfg.joint_ids
        targets = self.interrupt_scale * torch.rand((self.num_envs, len(cfg.interrupt_joint_names)), device=self.device) + self.interrupt_lower_bound
        default = self.robot.data.default_joint_pos.torch[:, joint_ids]
        limits = self.robot.data.default_joint_pos_limits.torch[:, joint_ids]
        if cfg.arm_clearance is not None:
            lower = torch.maximum(self.interrupt_lower_bound, limits[..., 0])
            upper = torch.minimum(self.interrupt_lower_bound + self.interrupt_scale, limits[..., 1])
            return guard_interrupt_targets(targets, lower, upper, cfg) - default

        signs = targets.new_tensor(cfg.arm_joint_signs)
        arms = (targets * signs).reshape(-1, 2, 4)
        low_roll = arms[..., 1] < cfg.low_roll_threshold
        arms[..., 2] = torch.where(
            low_roll, arms[..., 2].clamp(*cfg.low_roll_yaw_range), arms[..., 2]
        )
        arms[..., 2:4] = torch.where((arms[..., 1] < 0.0).unsqueeze(-1), 0.0, arms[..., 2:4])
        targets = arms.reshape(-1, 8) * signs
        return (targets - default).clamp(limits[..., 0] - default, limits[..., 1] - default)

    def curriculum_interrupt_clipping_mean_rad(self):
        # clipping mean with curriculum
        noise_mean = self.interrupt_rad_curriculum.unsqueeze(-1) * (self.robot.data.joint_pos.torch[:, self.interrupt_joint_cfg.joint_ids] - self.robot.data.default_joint_pos.torch[:, self.interrupt_joint_cfg.joint_ids])

        # clipping action rate with curriculum by rad.
        interrupt_actions = torch.clamp(
            self.interrupt_actions,
            (- self.cfg.interrupt.interrupt_init_range * self.interrupt_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.robot.action_scale,
            (self.cfg.interrupt.interrupt_init_range * self.interrupt_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.robot.action_scale
        )
        if self.cfg.interrupt.arm_clearance is not None:
            joint_ids = self.interrupt_joint_cfg.joint_ids
            default = self.robot.data.default_joint_pos.torch[:, joint_ids]
            mean = default + noise_mean
            radius = self.cfg.interrupt.interrupt_init_range * self.interrupt_rad_curriculum.unsqueeze(-1)
            signs = default.new_tensor(self.cfg.interrupt.arm_joint_signs)
            canonical_mean = (mean * signs).reshape(-1, 2, 4)
            roll_ceiling = canonical_mean[..., 1] + radius
            limits = self.robot.data.default_joint_pos_limits.torch[:, joint_ids]
            lower = torch.maximum(self.interrupt_lower_bound, limits[..., 0])
            upper = torch.minimum(self.interrupt_lower_bound + self.interrupt_scale, limits[..., 1])
            target = guard_interrupt_targets(
                default + self.action_scale * interrupt_actions, lower, upper, self.cfg.interrupt, roll_ceiling
            )
            interrupt_actions = (target - default) / self.action_scale
        return interrupt_actions
    
    def random_switch_interrupt(self):
        switch_rand = torch.rand(self.num_envs, device=self.device)
        switch = switch_rand < self.cfg.interrupt.switch_prob
        self.interrupt_mask = torch.where(torch.logical_and(switch, self.interrupt_mode_mask), ~self.interrupt_mask, self.interrupt_mask)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._append_action(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        if self.cfg.interrupt.use_interrupt:
            self.random_switch_interrupt() 
            interrupt_action_clip = self.curriculum_interrupt_clipping_mean_rad()
            self.actions[:, self.interrupt_joint_cfg.joint_ids] = torch.where(
                self.interrupt_mask.unsqueeze(-1).expand(-1, len(self.interrupt_joint_cfg.joint_ids)),
                interrupt_action_clip,
                self.actions[:, self.interrupt_joint_cfg.joint_ids]
            )
            self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)

        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos.torch
