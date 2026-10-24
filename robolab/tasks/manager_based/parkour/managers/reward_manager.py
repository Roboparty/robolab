"""Reward manager for computing multiple reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardManager, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .manager_term_cfg import MultiRewardCfg


class MultiRewardManager(RewardManager):
    """Manager to computing multiple groups of reward signals for a given world.

    The reward manager is similar to the RewardManager class but it computes multiple
    groups of reward signals.

    The reward terms should be clustered in RewardGroups. Then the returned reward_buf will
    be in shape (num_envs, num_groups) where each column corresponds to the total reward.
    """

    def __init__(self, cfg: MultiRewardCfg, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self._episode_sums = dict()
        for group_name in self.__group_term_names.keys():
            for term_name in self.__group_term_names[group_name]:
                self._episode_sums["_".join([term_name])] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )
        self._reward_buf = dict()
        for group_name in self.__group_term_cfgs.keys():
            self._reward_buf[group_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # A logging buffer for reward term of each group in current step
        self._termwise_reward_buf: dict[str, dict[str, torch.Tensor]] = dict()
        for group_name in self.__group_term_cfgs.keys():
            self._termwise_reward_buf[group_name] = dict()
            for term_name in self.__group_term_names[group_name]:
                self._termwise_reward_buf[group_name][term_name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )

        # per-environment term weights: dict[group_name][term_name] -> tensor(num_envs,)
        self._per_env_term_weights: dict[str, dict[str, torch.Tensor]] = dict()
        for group_name in self.__group_term_cfgs.keys():
            self._per_env_term_weights[group_name] = dict()
            for term_name, term_cfg in zip(self.__group_term_names[group_name], self.__group_term_cfgs[group_name]):
                self._per_env_term_weights[group_name][term_name] = torch.full(
                    (self.num_envs,), float(term_cfg.weight), dtype=torch.float, device=self.device
                )

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<MultiRewardManager> contains {len(self.__group_term_names)} active groups.\n"
        msg += f"and {sum(len(terms) for terms in self.__group_term_names.values())} active reward terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Group Terms"
        table.field_names = ["Index", "Group", "Name", "Weight"]
        # set alignment of table columns
        table.align["Group"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        index = 0
        for group_name in self.__group_term_names.keys():
            for term_name, term_cfg in zip(self.__group_term_names[group_name], self.__group_term_cfgs[group_name]):
                table.add_row([index, group_name, term_name, term_cfg.weight])
                index += 1
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self):
        """Get the active reward terms."""
        return self.__group_term_names

    @property
    def num_rewards(self) -> int:
        """Get the number of reward groups."""
        return len(self.__group_term_names)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # resolve environment ids
        if env_ids is None:
            env_ids = range(self.num_envs)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # extras["Episode_Reward/" + key + "/max_episode_len_s"] = episodic_sum_avg / self._env.max_episode_length_s
            # extras["Episode_Reward/" + key + "/sum"] = episodic_sum_avg
            # extras["Episode_Reward/" + key + "/timestep"] = torch.mean(
            #     self._episode_sums[key][env_ids] / self._env.episode_length_buf[env_ids]
            # )
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the reward terms
        for group_class_term_cfg in self.__group_class_term_cfgs.values():
            for term_cfg in group_class_term_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> dict[str, torch.Tensor]:
        """
        Returns:
            A dict or reward signal with shape (num_envs,) for each reward group.
        """
        for group_name in self.__group_term_cfgs.keys():
            term_combine_method = self.__group_term_combine_methods.get(group_name, "sum")
            if term_combine_method == "sum":
                self._reward_buf[group_name][:] = 0.0
            elif term_combine_method == "prod":
                self._reward_buf[group_name][:] = dt
            else:
                raise ValueError(f"Invalid term combine method: {term_combine_method}")
        # iterate over all the reward groups and terms
        for group_name, terms_cfgs in self.__group_term_cfgs.items():
            term_combine_method = self.__group_term_combine_methods.get(group_name, "sum")
            for term_name, term_cfg in zip(self.__group_term_names[group_name], terms_cfgs):
                # skip if weight is zero (kind of a micro-optimization)
                # if all per-env weights are zero, skip whole term
                per_env_weights = self._per_env_term_weights.get(group_name, {}).get(term_name, None)
                if per_env_weights is not None and torch.allclose(per_env_weights, torch.zeros_like(per_env_weights)):
                    continue
                # fallback: if no per-env buffer, skip if term_cfg.weight == 0
                if per_env_weights is None and term_cfg.weight == 0.0:
                    continue
                # compute the term's value
                value = term_cfg.func(self._env, **term_cfg.params)
                # apply per-env weights when available, otherwise use scalar weight
                if per_env_weights is not None:
                    value = value * per_env_weights
                else:
                    value = value * term_cfg.weight
                # update the reward buffer
                if term_combine_method == "sum":
                    self._reward_buf[group_name] += value * dt
                elif term_combine_method == "prod":
                    self._reward_buf[group_name] *= value
                else:
                    raise ValueError(f"Invalid term combine method: {term_combine_method}")
                # update the termwise reward buffer
                self._termwise_reward_buf[group_name][term_name] = value  # (num_envs,)
                # update the episodic sum
                self._episode_sums["_".join([term_name])] += value * dt
        # return the reward buffer
        return self._reward_buf

    def get_term_cfg(self, term_name: str, group_name: str | None = None) -> RewardTermCfg:
        """Get the term configuration for a given term name and group name.

        Args:
            term_name: The name of the term.
            group_name: The name of the group. If None, the first group will be used.

        Returns:
            The term configuration.
        """
        if group_name is None:
            group_name = list(self.__group_term_names.keys())[0]
        if group_name not in self.__group_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self.__group_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        index = self.__group_term_names[group_name].index(term_name)
        return self.__group_term_cfgs[group_name][index]

    def get_per_env_term_weights(self, term_name: str, group_name: str | None = None) -> torch.Tensor:
        """Return the per-environment weight tensor for a term.

        If the manager does not have a per-env buffer for the term, this will return
        a tensor filled with the scalar term weight.
        """
        if group_name is None:
            group_name = list(self.__group_term_names.keys())[0]
        if group_name not in self.__group_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self.__group_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        per_env = self._per_env_term_weights.get(group_name, {}).get(term_name, None)
        if per_env is not None:
            return per_env
        # fallback to scalar weight from cfg
        index = self.__group_term_names[group_name].index(term_name)
        term_cfg = self.__group_term_cfgs[group_name][index]
        return torch.full((self.num_envs,), float(term_cfg.weight), dtype=torch.float, device=self.device)

    def set_term_weight_for_envs(
        self, term_name: str, env_ids: Sequence[int] | torch.Tensor, weights: float | Sequence[float] | torch.Tensor, group_name: str | None = None
    ) -> None:
        """Set the per-environment weights for a term on specified env indices.

        Args:
            term_name: name of the term
            env_ids: indices of environments to update
            weights: scalar or sequence/tensor with length equal to env_ids
            group_name: group name; if None uses first group
        """
        if group_name is None:
            group_name = list(self.__group_term_names.keys())[0]
        if group_name not in self.__group_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self.__group_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        # normalize env_ids to tensor
        if isinstance(env_ids, slice):
            env_idx = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_idx = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
        else:
            env_idx = env_ids.to(device=self.device)
        # ensure per-env buffer exists
        if group_name not in self._per_env_term_weights:
            self._per_env_term_weights[group_name] = dict()
        if term_name not in self._per_env_term_weights[group_name]:
            # initialize from scalar cfg
            index = self.__group_term_names[group_name].index(term_name)
            term_cfg = self.__group_term_cfgs[group_name][index]
            self._per_env_term_weights[group_name][term_name] = torch.full(
                (self.num_envs,), float(term_cfg.weight), dtype=torch.float, device=self.device
            )
        buf = self._per_env_term_weights[group_name][term_name]
        # prepare weights tensor
        if isinstance(weights, (float, int)):
            w = torch.full((env_idx.shape[0],), float(weights), dtype=torch.float, device=self.device)
        else:
            w = torch.as_tensor(weights, dtype=torch.float, device=self.device)
            if w.numel() != env_idx.numel():
                raise ValueError("weights length must match env_ids length")
        buf[env_idx] = w

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for group_name in self.__group_term_cfgs.keys():
            for term_name in self.__group_term_names[group_name]:
                # NOTE: there are some shitty conventions in feeding back to manager_live_visualizer.
                # You need to return a list[tuple[str, Iterable[float]]] where the first element is the name of the term
                terms.append(
                    (
                        f"{group_name}-{term_name}",
                        [self._termwise_reward_buf[group_name][term_name][env_idx].cpu().item()],
                    )
                )
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepare the reward group and each term in the groups for computation."""
        self.__group_term_names: dict[str, list[str]] = dict()
        self.__group_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self.__group_class_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self.__group_term_combine_methods: dict[str, str] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            groups_cfg_items = self.cfg.items()
        else:
            groups_cfg_items = self.cfg.__dict__.items()
        for group_name, group_cfg in groups_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms
            for term_name, term_cfg in group_cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # # check configs for the group specifically
                # if term_name == "combine_method":
                #     assert isinstance(
                #         term_cfg, str
                #     ), f"Configuration for the term '{term_name}' in group '{group_name}' is not of type str."
                #     self.__group_term_combine_methods[group_name] = term_cfg
                #     continue
                
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' in group '{group_name}' is not of type"
                        f" RewardTermCfg. Received: '{type(term_cfg)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add the term to the group
                if group_name not in self.__group_term_names:
                    self.__group_term_names[group_name] = list()
                    self.__group_term_cfgs[group_name] = list()
                    self.__group_class_term_cfgs[group_name] = list()
                self.__group_term_names[group_name].append(term_name)
                self.__group_term_cfgs[group_name].append(term_cfg)
                if isinstance(term_cfg.func, ManagerTermBase):
                    self.__group_class_term_cfgs[group_name].append(term_cfg)
