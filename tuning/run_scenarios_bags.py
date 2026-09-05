#!/usr/bin/env python3
"""
Run and record rosbags for a sequence of scenarios (single environment per run).

This runner reuses helpers from `bayesian_mpc_tuner.py`:
- `SCENARIOS`
- `SimulationManager`
- `RosbagRecorder`
- `PerformanceMonitor`

It launches sim per scenario, records one bag per scenario, stops recording when
all goals are reached (or timeout), then fully kills sim before next scenario.

Usage examples:
  python3 tuning/run_scenarios_bags.py
  python3 tuning/run_scenarios_bags.py --gui
  python3 tuning/run_scenarios_bags.py --names open_square,warehouse_loop
  python3 tuning/run_scenarios_bags.py --world warehouse.world
  python3 tuning/run_scenarios_bags.py --world default.sdf --world-pkg go2_sim
  python3 tuning/run_scenarios_bags.py --outdir /tmp/bags --duration 120
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path


GOAL_REACHED_RADIUS = 0.5


def load_tuner_module(path: Path):
    spec = importlib.util.spec_from_file_location("bayes_tuner", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scenario_matches_world(scenario: dict, world: str | None, world_pkg: str | None) -> bool:
    if world is None:
        return True
    if scenario.get("world") != world:
        return False
    if world_pkg is not None and scenario.get("world_pkg") != world_pkg:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Run scenarios and record rosbags")
    ap.add_argument(
        "--tuner",
        type=str,
        default=str(Path(__file__).parent / "bayesian_mpc_tuner.py"),
        help="Path to bayesian_mpc_tuner.py to reuse helpers",
    )
    ap.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated scenario names to run (default: all)",
    )
    ap.add_argument(
        "--world",
        type=str,
        default=None,
        help="Run only scenarios that use this world file (e.g. warehouse.world)",
    )
    ap.add_argument(
        "--world-pkg",
        type=str,
        default=None,
        help="Optional world package filter when used with --world (e.g. sim_worlds)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Root output directory for bags (default: Go2_navigation/bags_recordings)",
    )
    ap.add_argument("--gui", action="store_true", help="Launch Gazebo + RViz GUI")
    ap.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override per-scenario max duration (seconds)",
    )
    ap.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip extra cleanup wait between scenarios",
    )
    args = ap.parse_args()

    tuner_path = Path(args.tuner)
    if not tuner_path.exists():
        raise SystemExit(f"tuner file not found: {tuner_path}")

    mod = load_tuner_module(tuner_path)

    scenarios = getattr(mod, "SCENARIOS", None)
    if scenarios is None:
        raise SystemExit("SCENARIOS not found in tuner module")

    SimulationManager = getattr(mod, "SimulationManager")
    RosbagRecorder = getattr(mod, "RosbagRecorder")

    
    bags_root = Path(__file__).parent.parent / "bags_recordings"

    # Default destination depends on selected world, unless overridden by --outdir
    # or BAGS_OUTPUT_DIR. Keep folder names aligned with existing repository layout.
    world_to_subdir = {
        "indoor_office.world": "office",
        "warehouse.world": "werehouse_env",
        "default.sdf": "open",
    }
    default_subdir = world_to_subdir.get(args.world, "office") if args.world else "office"
    default_bags = bags_root / default_subdir
    results_dir = Path(os.environ.get("BAGS_OUTPUT_DIR", str(default_bags))).expanduser()
    planner_delay_sec = getattr(mod, "PLANNER_DELAY_SEC", 30)
    cleanup_wait_sec = getattr(mod, "CLEANUP_WAIT_SEC", 5)
    default_timeout = getattr(mod, "SCENARIO_TIMEOUT", 120)

    out_root = Path(args.outdir) if args.outdir else results_dir
    out_root.mkdir(parents=True, exist_ok=True)

    selected_names = None
    if args.names:
        selected_names = {n.strip() for n in args.names.split(",") if n.strip()}

    # Build final run list using name + world filters.
    run_scenarios = []
    for sc in scenarios:
        name = sc.get("name", "unnamed")
        if selected_names and name not in selected_names:
            continue
        if not scenario_matches_world(sc, args.world, args.world_pkg):
            continue
        run_scenarios.append(sc)

    if not run_scenarios:
        msg = "No scenarios matched the provided filters"
        details = []
        if args.names:
            details.append(f"names={args.names}")
        if args.world:
            details.append(f"world={args.world}")
        if args.world_pkg:
            details.append(f"world_pkg={args.world_pkg}")
        if details:
            msg += f" ({', '.join(details)})"
        raise SystemExit(msg)

    print(f"Output root: {out_root}")
    if args.world:
        print(f"World filter: {args.world}  pkg={args.world_pkg or '*'}")
    print(f"Scenarios to run ({len(run_scenarios)}): {[s.get('name') for s in run_scenarios]}")

    sim = SimulationManager(gui=args.gui)
    bag = None

    try:
        for sc in run_scenarios:
            name = sc.get("name", "unnamed")
            print(f"\n== Running scenario: {name} ==")

            # One folder per scenario; suffix to avoid overwrite.
            base_dir = out_root / f"scenario_{name}"
            trial_dir = base_dir
            suffix = 1
            while trial_dir.exists():
                trial_dir = Path(f"{base_dir}_{suffix}")
                suffix += 1

            bag_dir = trial_dir / "rosbag"
            bag = RosbagRecorder(bag_dir)

            try:
                params_yaml = getattr(mod, "BASE_PARAMS", None)
                params_yaml = Path(params_yaml) if params_yaml else None
                if params_yaml is None:
                    raise RuntimeError("BASE_PARAMS not found in tuner module")

                sim.launch(params_yaml, sc)

                time.sleep(3)
                sim.spawn_obstacles(sc)

                wait_after = max(planner_delay_sec - 3, 5)
                print(f"  waiting {wait_after}s for planner to stabilise...")
                time.sleep(wait_after)

                print(f"  recording bag to: {bag_dir}")
                bag.start()

                duration = args.duration or sc.get("timeout") or default_timeout
                goals = sc.get("goals", [[sc.get("goal_x", 0.0), sc.get("goal_y", 0.0)]])
                print(f"  max duration: {duration}s | goals: {goals}")

                try:
                    if mod.rclpy.ok():
                        mod.rclpy.shutdown()
                    mod.rclpy.init()
                    monitor = mod.PerformanceMonitor()

                    current_idx = 0
                    monitor.start(goals[0][0], goals[0][1])
                    start_t = time.time()
                    end_t = start_t + duration
                    last_log = 0.0
                    x = y = gx = gy = 0.0

                    while time.time() < end_t:
                        mod.rclpy.spin_once(monitor, timeout_sec=0.1)
                        now = time.time()

                        if monitor.trajectory:
                            _, x, y, _ = monitor.trajectory[-1]
                            gx, gy = goals[current_idx]
                            dist = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

                            if dist < GOAL_REACHED_RADIUS:
                                print(
                                    f"    [nav] goal {current_idx + 1}/{len(goals)} reached "
                                    f"(pos=({x:.2f},{y:.2f}))"
                                )
                                current_idx += 1
                                if current_idx >= len(goals):
                                    print("    all goals reached -> stop recording")
                                    break
                                monitor.publish_goal(goals[current_idx][0], goals[current_idx][1])
                                print(
                                    f"    [nav] next goal {current_idx + 1}/{len(goals)}: "
                                    f"{goals[current_idx]}"
                                )

                        if now - last_log >= 5.0:
                            rem = int(max(0, end_t - now))
                            print(
                                f"    ...{rem}s remaining  pos=({x:.2f},{y:.2f})  "
                                f"target=({gx:.2f},{gy:.2f})",
                                flush=True,
                            )
                            last_log = now

                    monitor.stop()
                    try:
                        monitor.destroy_node()
                    except Exception:
                        pass
                    mod.rclpy.shutdown()

                except Exception as e:
                    print(f"  monitor error -> fallback timed recording: {e}")
                    start_t = time.time()
                    while time.time() - start_t < duration:
                        rem = int(duration - (time.time() - start_t))
                        print(f"    ...{rem}s remaining", end="\r", flush=True)
                        time.sleep(1)

                finally:
                    print("\n  stopping bag and shutting down simulation")
                    bag.stop()
                    sim.kill()
                    bag = None
                    if not args.no_cleanup:
                        time.sleep(cleanup_wait_sec)

            except Exception as e:
                print(f"  scenario {name} error: {e}")
                try:
                    bag.stop()
                except Exception:
                    pass
                sim.kill()
                bag = None
                if not args.no_cleanup:
                    time.sleep(cleanup_wait_sec)

        print("\nAll selected scenarios completed.")

    except KeyboardInterrupt:
        print("\nInterrupted -> stopping current recording and exiting")
        if bag is not None:
            try:
                bag.stop()
            except Exception:
                pass
        sim.kill()


if __name__ == "__main__":
    main()
