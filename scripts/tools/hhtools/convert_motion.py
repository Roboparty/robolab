# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert HHTools robot CSV/PKL exports to AMP (.pkl) or BeyondMimic (.npz).

Forward kinematics uses the training asset through MuJoCo without stepping
physics. Joint/body names and the xyzw quaternion convention are saved with
the motion so both training loaders can resolve their own coordinate order.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_robot(robot: str):
    from robolab.assets.robots.roboparty import RPO_CFG, RP1_24DOF_CFG

    cfg = {"rpo": RPO_CFG, "rp1": RP1_24DOF_CFG}[robot]
    asset_path = Path(cfg.spawn.asset_path).resolve()
    spec = mujoco.MjSpec.from_file(str(asset_path))
    spec.compiler.fusestatic = False
    if asset_path.suffix == ".urdf":
        spec.body(cfg.body_ordering[0]).add_freejoint(name="root")
    model = spec.compile()
    return model, list(cfg.joint_ordering), list(cfg.body_ordering)


def load_motion(path: Path):
    """Read the current HHTools export schemas; raw CSV needs a named header."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            metadata = {}
            rows = []
            for line in stream:
                if line.startswith("#"):
                    key, separator, value = line[1:].partition(":")
                    if separator:
                        metadata[key.strip()] = value.strip()
                elif line.strip():
                    rows.append(line)
        reader = csv.reader(rows)
        header = next(reader, [])
        root_columns = ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
        required = ["time", *root_columns]
        if len(header) != len(set(header)) or not set(required).issubset(header):
            raise ValueError("Export CSV from HHTools with comments and column headers enabled.")
        data = np.asarray(list(reader), dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(header):
            raise ValueError("CSV rows must match the column header.")
        names = [column.removeprefix("dof_") for column in header if column.startswith("dof_")]
        columns = [header.index(column) for column in root_columns]
        columns += [header.index(f"dof_{name}") for name in names]
        joint_q = data[:, columns]
        times = data[:, header.index("time")]
        fps = float(metadata["sample_rate"]) if "sample_rate" in metadata else None
    elif path.suffix.lower() == ".pkl":
        with path.open("rb") as stream:
            export = pickle.load(stream)
        if export.get("hhtools_export") not in {"retarget_v1", "r2r_v1"}:
            raise ValueError("Expected a robot PKL exported by HHTools.")
        robot = export["robot"]
        names = list(robot["dof_names"])
        joint_q = np.array(robot["joint_q"], dtype=np.float64, copy=True)
        order = robot["root_quat_format"]
        if order == "wxyz":
            joint_q[:, 3:7] = joint_q[:, [4, 5, 6, 3]]
        elif order != "xyzw":
            raise ValueError(f"Unsupported HHTools root_quat_format: {order}")
        fps = float(robot["sample_rate"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("HHTools sample_rate must be positive and finite.")
        times = np.arange(len(joint_q), dtype=np.float64) / fps
    else:
        raise ValueError("Input must be a HHTools .csv or .pkl robot export.")

    if joint_q.ndim != 2 or joint_q.shape[1] != 7 + len(names) or len(joint_q) < 2:
        raise ValueError("Motion needs at least two frames, seven root columns and one column per joint.")
    if not np.isfinite(joint_q).all() or not np.isfinite(times).all():
        raise ValueError("Motion contains NaN or infinite values.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("Motion timestamps must be strictly increasing.")
    if len(names) != len(set(names)):
        raise ValueError("Motion contains duplicate joint names.")
    if np.any(np.linalg.norm(joint_q[:, 3:7], axis=-1) < 1e-8):
        raise ValueError("Motion contains a zero root quaternion.")
    times = times - times[0]
    fps = 1.0 / np.median(np.diff(times)) if fps is None else fps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("HHTools sample_rate must be positive and finite.")
    return times, joint_q, names, fps


def resample_motion(times, joint_q, fps: float, keep_last: bool = True):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Output fps must be positive and finite.")
    # HHTools CSV timestamps are rounded to six decimals. Retain a final sample
    # that differs from the exact output grid only by that rounding error.
    count = int(np.floor((times[-1] + 1e-6) * fps)) + 1 if keep_last else int(round(times[-1] * fps))
    if count < 2:
        raise ValueError("Motion is shorter than one output frame interval.")
    output_times = np.minimum(np.arange(count, dtype=np.float64) / fps, times[-1])
    result = np.empty((count, joint_q.shape[1]), dtype=np.float64)
    for column in [0, 1, 2, *range(7, joint_q.shape[1])]:
        result[:, column] = np.interp(output_times, times, joint_q[:, column])
    result[:, 3:7] = Slerp(times, Rotation.from_quat(joint_q[:, 3:7]))(output_times).as_quat()
    return result


def angular_velocity(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """World angular velocity, repeating central differences at the endpoints."""
    count, bodies, _ = quaternions.shape
    indices = np.arange(count)
    before = np.maximum(indices - 1, 0)
    after = np.minimum(indices + 1, count - 1)
    q_before = Rotation.from_quat(quaternions[before].reshape(-1, 4))
    q_after = Rotation.from_quat(quaternions[after].reshape(-1, 4))
    rotvec = (q_after * q_before.inv()).as_rotvec().reshape(count, bodies, 3)
    velocity = rotvec * (fps / (after - before))[:, None, None]
    if count > 2:
        velocity[0], velocity[-1] = velocity[1], velocity[-2]
    return velocity


def convert_motion(model, joint_names, body_names, joint_q, source_names, fps):
    missing = set(joint_names) - set(source_names)
    extra = set(source_names) - set(joint_names)
    if missing or extra:
        raise ValueError(f"Joint names do not match the training robot; missing={sorted(missing)}, extra={sorted(extra)}")
    joint_pos = joint_q[:, [7 + source_names.index(name) for name in joint_names]]
    joint_ids = np.asarray([model.joint(name).id for name in joint_names])
    if not np.all(model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Only scalar revolute robot joints are supported.")
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError("Training asset must contain exactly one floating root joint.")
    root_address = model.jnt_qposadr[free_joints[0]]
    root_dof = model.jnt_dofadr[free_joints[0]]
    joint_addresses = model.jnt_qposadr[joint_ids]
    joint_dofs = model.jnt_dofadr[joint_ids]
    body_ids = np.asarray([model.body(name).id for name in body_names])
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)
    root_lin_vel = np.gradient(joint_q[:, :3], 1.0 / fps, axis=0)
    root_ang_vel = angular_velocity(joint_q[:, None, 3:7], fps)[:, 0]
    root_ang_vel_local = Rotation.from_quat(joint_q[:, 3:7]).inv().apply(root_ang_vel)
    data = mujoco.MjData(model)
    body_pos = np.empty((len(joint_q), len(body_names), 3))
    body_quat = np.empty((len(joint_q), len(body_names), 4))
    body_lin_vel = np.empty_like(body_pos)
    body_ang_vel = np.empty_like(body_pos)
    jac_pos, jac_rot = np.empty((3, model.nv)), np.empty((3, model.nv))
    for frame in range(len(joint_q)):
        data.qpos[root_address : root_address + 3] = joint_q[frame, :3]
        data.qpos[root_address + 3 : root_address + 7] = joint_q[frame, [6, 3, 4, 5]]
        data.qpos[joint_addresses] = joint_pos[frame]
        data.qvel[root_dof : root_dof + 3] = root_lin_vel[frame]
        data.qvel[root_dof + 3 : root_dof + 6] = root_ang_vel_local[frame]
        data.qvel[joint_dofs] = joint_vel[frame]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids][:, [1, 2, 3, 0]]
        # Isaac Lab body_pos_w/body_quat_w use link origins, but
        # body_lin_vel_w uses centers of mass. Preserve that mixed convention.
        for column, body_id in enumerate(body_ids):
            mujoco.mj_jacBodyCom(model, data, jac_pos, jac_rot, body_id)
            body_lin_vel[frame, column] = jac_pos @ data.qvel
            body_ang_vel[frame, column] = jac_rot @ data.qvel
    return {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
        "quaternion_order": "xyzw",
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("rpo", "rp1"), required=True)
    parser.add_argument("--input_file", "-f", type=Path, required=True)
    parser.add_argument("--output_file", "-o", type=Path, required=True, help="AMP .pkl or BeyondMimic .npz output.")
    parser.add_argument("--output_fps", type=float, help="Defaults to the source rate for AMP and 50 Hz for BeyondMimic.")
    parser.add_argument("--loop", choices=("clamp", "wrap"), default="clamp", help="AMP motion sampling mode.")
    args = parser.parse_args()
    suffix = args.output_file.suffix.lower()
    if suffix not in {".pkl", ".npz"}:
        parser.error("Output must end in .pkl (AMP) or .npz (BeyondMimic).")
    if args.input_file.resolve() == args.output_file.resolve():
        parser.error("Input and output must be different files.")
    times, joint_q, names, source_fps = load_motion(args.input_file)
    fps = args.output_fps if args.output_fps is not None else (50.0 if suffix == ".npz" else source_fps)
    joint_q = resample_motion(times, joint_q, fps, keep_last=suffix == ".pkl")
    model, joint_names, body_names = load_robot(args.robot)
    motion = convert_motion(model, joint_names, body_names, joint_q, names, fps)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".npz":
        with args.output_file.open("wb") as stream:
            np.savez_compressed(stream, **motion)
    else:
        amp_motion = {
            "fps": float(fps),
            "root_pos": motion["body_pos_w"][:, 0],
            "root_rot": motion["body_quat_w"][:, 0],
            "dof_pos": motion["joint_pos"],
            "loop_mode": int(args.loop == "wrap"),
            "key_body_pos": motion["body_pos_w"],
            "quaternion_order": motion["quaternion_order"],
            "joint_names": joint_names,
            "key_body_names": body_names,
        }
        with args.output_file.open("wb") as stream:
            pickle.dump(amp_motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.output_file}: {len(joint_q)} frames, {fps:g} Hz, {(len(joint_q) - 1) / fps:.3f} s")


if __name__ == "__main__":
    main()
