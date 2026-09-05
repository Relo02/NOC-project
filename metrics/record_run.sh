#!/usr/bin/env bash
# Registra una missione del G1 per l'analisi offline dei due pannelli.
#
#   ./metrics/record_run.sh [nome]
#
# Poi, in un altro terminale, si lancia lo stack e si mandano i goal da RViz.
# La griglia di occupazione NON viene registrata: e' grossa e i pannelli la
# ricalcolano comunque dal campo di costo.
set -e
NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/metrics/bags/$NAME"
mkdir -p "$(dirname "$OUT")"
echo "registro in $OUT  (Ctrl-C per fermare)"
exec ros2 bag record -o "$OUT" \
    /robot_pose /lidar/points_filtered /a_star/path /mpc/predicted_path \
    /mpc/next_setpoint /mpc/diagnostics /global_goal /cmd_vel /odom \
    /tf /tf_static
