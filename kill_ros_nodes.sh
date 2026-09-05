#!/usr/bin/env bash
set -euo pipefail

# Collect ROS-related processes and avoid killing this script shell.
self=$$
parent=$PPID
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a pids=()

add_pids() {
  local pid
  while IFS= read -r pid; do
    [ -n "$pid" ] && pids+=("$pid")
  done
}


# Common process names of the G1 stack.
for name in \
  ros2 rviz2 rqt \
  robot_state_publisher static_transform_publisher \
  mujoco_sim; do
  add_pids < <(pgrep -x "$name" || true)
done

# Catch ROS-installed executables that may not use common names.
add_pids < <(pgrep -f '/opt/ros/' || true)
# Catch workspace-built ROS executables.
add_pids < <(pgrep -f "${workspace_root}/install/" || true)
add_pids < <(pgrep -f "${workspace_root}/build/" || true)

# Catch the stack's launch and its Python nodes, which pgrep -x misses
# because they run as `python3 <path>`.
add_pids < <(pgrep -f 'g1_a_star_mpc.launch.py' || true)
add_pids < <(pgrep -f 'g1_sim.launch.py' || true)
add_pids < <(pgrep -f 'mujoco_sim' || true)
add_pids < <(pgrep -f 'a_star_node|mpc_node|odom_to_pose_node|setpoint_to_cmd_vel_node' || true)
add_pids < <(pgrep -f 'lidar_filter_node|goal_relay_node|mission_runner_node' || true)

# Unique + filter out this script shell and parent.
pids=$(
  printf '%s\n' "${pids[@]:-}" \
  | grep -Ev '^\s*$' \
  | sort -u \
  | grep -Ev "^(${self}|${parent})$" \
  || true
)

if [ -z "${pids:-}" ]; then
  echo "No ROS processes found."
  exit 0
fi

echo "Stopping ROS processes:"
printf '%s\n' "$pids"

# Try TERM first.
while IFS= read -r pid; do
  kill "$pid" 2>/dev/null || true
done <<< "$pids"

sleep 1

# Force kill any remaining.
while IFS= read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done <<< "$pids"

echo "Done."
