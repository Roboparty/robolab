# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train RoboLab tasks through the Isaac Lab RSL-RL workflow."""

import sys

import robolab.tasks  # noqa: F401

from isaaclab_rl.entrypoints.backends.train_rsl_rl import run


if __name__ == "__main__":
    run(sys.argv[1:])
