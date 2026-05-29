#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ws_root="$repo_root/environment/ros_ws/svo"

if [[ ! -d /opt/ros/noetic ]]; then
  echo "missing /opt/ros/noetic" >&2
  echo "run this inside the svo:noetic container" >&2
  exit 1
fi

export ROS_DISTRO="${ROS_DISTRO:-noetic}"
set +u
source /opt/ros/noetic/setup.bash
set -u

if [[ ! -d "$ws_root/src/rpg_svo_pro_open" ]]; then
  echo "workspace not initialized" >&2
  echo "run: bash scripts/setup_svo_workspace.sh" >&2
  exit 1
fi

catkin build --workspace "$ws_root"
