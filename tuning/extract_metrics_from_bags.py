#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd  # optional
except Exception:
    pd = None

try:
    from rosbags.rosbag2 import Reader
except Exception as e:
    raise SystemExit("Missing dependency 'rosbags'. Install with: pip install rosbags") from e

try:
    from rosbags.serde import deserialize_cdr as cdr_deserialize
except Exception:
    from rosbags.typesys import Stores, get_typestore
    _store = get_typestore(getattr(Stores, 'LATEST', Stores.ROS2_HUMBLE))

    def cdr_deserialize(rawdata, msgtype):
        return _store.deserialize_cdr(rawdata, msgtype)


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def get_ros_messages(reader, topic_name):
    connections = [c for c in reader.connections if c.topic == topic_name]
    if not connections:
        return
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        try:
            msg = cdr_deserialize(rawdata, connection.msgtype)
            yield timestamp * 1e-9, msg
        except Exception:
            continue


def extract_robot_trajectory(reader):
    pose_msgs = list(get_ros_messages(reader, '/go2/pose'))
    if pose_msgs:
        ts, traj = [], []
        for t, m in pose_msgs:
            ts.append(float(t))
            traj.append([float(m.pose.position.x), float(m.pose.position.y)])
        q = pose_msgs[-1][1].pose.orientation
        final_pose = [traj[-1][0], traj[-1][1], yaw_from_quat(q)]
        return ts, traj, final_pose, '/go2/pose'

    odom_msgs = list(get_ros_messages(reader, '/odom'))
    if odom_msgs:
        ts, traj = [], []
        for t, m in odom_msgs:
            ts.append(float(t))
            p = m.pose.pose.position
            traj.append([float(p.x), float(p.y)])
        q = odom_msgs[-1][1].pose.pose.orientation
        final_pose = [traj[-1][0], traj[-1][1], yaw_from_quat(q)]
        return ts, traj, final_pose, '/odom'

    return [], [], None, 'missing'


def extract_goal_pose(reader, goal_topic='/goal_pose', path_topic='/a_star/path'):
    goal_msgs = list(get_ros_messages(reader, goal_topic))
    if goal_msgs:
        _, gm = goal_msgs[-1]
        q = gm.pose.orientation
        return [gm.pose.position.x, gm.pose.position.y, yaw_from_quat(q)], 'goal_pose'

    path_msgs = list(get_ros_messages(reader, path_topic))
    for _, path_msg in reversed(path_msgs):
        if path_msg.poses:
            last = path_msg.poses[-1].pose
            q = last.orientation
            return [last.position.x, last.position.y, yaw_from_quat(q)], 'path_endpoint'

    return None, 'missing'


def extract_reference_points(reader, path_topic='/a_star/path', sample_step=2):
    path_msgs = list(get_ros_messages(reader, path_topic))
    if not path_msgs:
        return np.empty((0, 2), dtype=np.float64), 0

    ref = []
    for _, msg in path_msgs:
        for i, p in enumerate(msg.poses):
            if i % sample_step == 0:
                ref.append([float(p.pose.position.x), float(p.pose.position.y)])
    if not ref:
        return np.empty((0, 2), dtype=np.float64), len(path_msgs)
    return np.asarray(ref, dtype=np.float64), len(path_msgs)


def extract_mpc_solving_times_ms(reader, diag_topic='/mpc/diagnostics'):
    vals = []
    for _, msg in get_ros_messages(reader, diag_topic):
        data = getattr(msg, 'data', None)
        if data is None or len(data) < 3:
            continue
        try:
            v = float(data[2])
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def calculate_cte(robot_path, reference_points):
    if not robot_path or reference_points is None or len(reference_points) == 0:
        return float('nan')

    rp = np.asarray(reference_points, dtype=float)
    traj = np.asarray(robot_path, dtype=float)
    rp = np.unique(np.round(rp, 3), axis=0)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(rp)
        d, _ = tree.query(traj, k=1)
        return float(np.mean(d))
    except Exception:
        errs = []
        for p in traj:
            d = np.linalg.norm(rp - p, axis=1)
            errs.append(float(np.min(d)))
        return float(np.mean(errs)) if errs else float('nan')


def first_reach_time_sec(ts, traj, goal_xy, tol=0.5):
    if not ts or not traj or goal_xy is None:
        return float('nan')
    t0 = ts[0]
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    for t, p in zip(ts, traj):
        if math.hypot(p[0] - gx, p[1] - gy) <= tol:
            return float(t - t0)
    return float('nan')


def calculate_path_length(path_xy):
    if path_xy is None or len(path_xy) < 2:
        return float('nan')
    arr = np.asarray(path_xy, dtype=float)
    d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return float(np.sum(d))


def discover_runs(metrics_dir: Path):
    known_methods = ['baseline', 'bo_opti', 'bo_tuned', 'planner_params', 'copy_planner_params']
    method_alias = {
        'bo_tuned': 'bo_opti',
        'planner_params': 'bo_opti',
        'copy_planner_params': 'baseline',
    }
    world_alias = {
        'opne_world': 'open_world',
        'werehouse_env': 'warehouse',
        'werehouse': 'warehouse',
    }

    run_items = []
    for run_dir in sorted(metrics_dir.glob('*')):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        method = 'unknown'
        world = run_name
        for km in known_methods:
            if run_name.endswith(f'_{km}'):
                method = km
                world = run_name[:-len(f'_{km}')]
                break

        method = method_alias.get(method, method)
        world = world_alias.get(world.strip().lower(), world.strip().lower())

        for trial_dir in sorted(run_dir.glob('T*')):
            if not trial_dir.is_dir():
                continue
            m = re.match(r'T(\d+)', trial_dir.name)
            trial_idx = int(m.group(1)) if m else -1

            bag_dir_a = trial_dir / 'bag'
            if (bag_dir_a / 'metadata.yaml').exists() and any(bag_dir_a.glob('*.db3')):
                run_items.append((world, method, trial_idx, bag_dir_a))
                continue

            if (trial_dir / 'metadata.yaml').exists() and any(trial_dir.glob('*.db3')):
                run_items.append((world, method, trial_idx, trial_dir))
                continue

    return run_items


def _write_csv(rows, path: Path):
    if not rows:
        path.write_text('')
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v, nd=3):
    if isinstance(v, float):
        if math.isnan(v):
            return 'nan'
        return f'{v:.{nd}f}'
    return str(v)


def _print_table(rows, title):
    print(f'\n{title}')
    if not rows:
        print('(empty)')
        return
    keys = list(rows[0].keys())
    print(' | '.join(keys))
    for r in rows:
        print(' | '.join(_fmt(r[k]) for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--goal-tol', type=float, default=0.5)
    ap.add_argument('--outdir', type=Path, default=None)
    args = ap.parse_args()

    metrics_dir = args.repo_root / 'bags_recordings' / 'metrics_data'
    out_dir = args.outdir or (args.repo_root / 'bag_gp_tuning' / 'metrics_from_bags')
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(metrics_dir)
    print(f'Found {len(runs)} run bags in {metrics_dir}')

    per_run = []
    pooled_mpc_solves = defaultdict(list)

    for world, method, trial_idx, bag_dir in runs:
        try:
            with Reader(str(bag_dir)) as reader:
                ts, robot_path, final_pose, pose_source = extract_robot_trajectory(reader)
                goal_pose, goal_source = extract_goal_pose(reader)
                ref_pts, n_path_msgs = extract_reference_points(reader)
                mpc_solve_ms = extract_mpc_solving_times_ms(reader)

                success = 0
                dist_final = float('nan')
                if final_pose is not None and goal_pose is not None:
                    dist_final = float(np.linalg.norm(np.array(final_pose[:2]) - np.array(goal_pose[:2])))
                    success = int(dist_final <= args.goal_tol)

                cte = calculate_cte(robot_path, ref_pts)
                t_goal = first_reach_time_sec(ts, robot_path, goal_pose[:2] if goal_pose else None, tol=args.goal_tol)
                path_len = calculate_path_length(robot_path)

                solve_mean = float(np.mean(mpc_solve_ms)) if mpc_solve_ms.size else float('nan')
                solve_p95 = float(np.percentile(mpc_solve_ms, 95)) if mpc_solve_ms.size else float('nan')
                solve_max = float(np.max(mpc_solve_ms)) if mpc_solve_ms.size else float('nan')

                if mpc_solve_ms.size:
                    pooled_mpc_solves[(world, method)].extend(mpc_solve_ms.tolist())

                per_run.append({
                    'World': world,
                    'Method': method,
                    'Trial': trial_idx,
                    'Success': success,
                    'SR_run (%)': 100.0 if success else 0.0,
                    'CE (m)': cte,
                    'Time-to-goal (s)': t_goal,
                    'Path length (m)': path_len,
                    'Final dist to goal (m)': dist_final,
                    'MPC solve mean (ms)': solve_mean,
                    'MPC solve p95 (ms)': solve_p95,
                    'MPC solve max (ms)': solve_max,
                    'MPC solve samples': int(mpc_solve_ms.size),
                    'pose_source': pose_source,
                    'goal_source': goal_source,
                    'n_path_msgs': int(n_path_msgs),
                    'n_ref_points': int(ref_pts.shape[0]),
                    'bag_dir': str(bag_dir),
                })
        except Exception as e:
            print(f'Skipping {bag_dir}: {e}')

    per_run.sort(key=lambda r: (r['World'], r['Method'], r['Trial']))

    grouped = defaultdict(list)
    for r in per_run:
        grouped[(r['World'], r['Method'])].append(r)

    agg_rows = []
    for (world, method), g in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(g)
        n_succ = sum(int(x['Success']) for x in g)
        sr = 100.0 * n_succ / n if n > 0 else float('nan')

        ce_success = [x['CE (m)'] for x in g if x['Success'] == 1 and not _isnan(x['CE (m)'])]
        ce_all = [x['CE (m)'] for x in g if not _isnan(x['CE (m)'])]
        ce_mean = float(np.mean(ce_success)) if ce_success else (float(np.mean(ce_all)) if ce_all else float('nan'))

        tg_success = [x['Time-to-goal (s)'] for x in g if x['Success'] == 1 and not _isnan(x['Time-to-goal (s)'])]

        pooled = np.asarray(pooled_mpc_solves.get((world, method), []), dtype=np.float64)
        pooled_mean = float(np.mean(pooled)) if pooled.size else float('nan')
        pooled_p95 = float(np.percentile(pooled, 95)) if pooled.size else float('nan')
        pooled_max = float(np.max(pooled)) if pooled.size else float('nan')

        agg_rows.append({
            'World': world,
            'Method': method,
            'SR (%)': sr,
            'CE (m)': ce_mean,
            'Runs': n,
            'Successes': n_succ,
            'Time-to-goal mean (s)': float(np.mean(tg_success)) if tg_success else float('nan'),
            'Time-to-goal std (s)': float(np.std(tg_success, ddof=1)) if len(tg_success) > 1 else float('nan'),
            'MPC solve mean (ms)': pooled_mean,
            'MPC solve p95 (ms)': pooled_p95,
            'MPC solve max (ms)': pooled_max,
            'MPC solve samples': int(pooled.size),
        })

    per_run_path = out_dir / 'per_run_metrics.csv'
    agg_path = out_dir / 'aggregated_metrics.csv'
    latex_path = out_dir / 'aggregated_metrics_latex.txt'

    _write_csv(per_run, per_run_path)
    _write_csv(agg_rows, agg_path)

    if pd is not None:
        per_df = pd.DataFrame(per_run)
        agg_df = pd.DataFrame(agg_rows)
        print('\nPer-run metrics:')
        print(per_df.to_string(index=False))
        print('\nAggregated metrics:')
        print(agg_df.to_string(index=False))
    else:
        _print_table(per_run, 'Per-run metrics:')
        _print_table(agg_rows, 'Aggregated metrics:')

    with open(latex_path, 'w') as f:
        for row in agg_rows:
            ce = 'N/A' if _isnan(row['CE (m)']) else f"{row['CE (m)']:.2f}"
            mpc_mean = 'N/A' if _isnan(row['MPC solve mean (ms)']) else f"{row['MPC solve mean (ms)']:.1f}"
            f.write(
                f"{row['World']} & {row['Method']} & {row['SR (%)']:.0f} & {ce} & {mpc_mean} \\\\n"
            )

    print(f'\nSaved: {per_run_path}')
    print(f'Saved: {agg_path}')
    print(f'Saved: {latex_path}')


if __name__ == '__main__':
    main()
