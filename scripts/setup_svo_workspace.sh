#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ws_root="$repo_root/environment/ros_ws/svo"
src_dir="$ws_root/src"
upstream_dir="$repo_root/methods/svo/upstream"
deps_file="$repo_root/methods/svo/dependencies_https.yaml"

mkdir -p "$src_dir"

if [[ ! -d /opt/ros/noetic ]]; then
  echo "missing /opt/ros/noetic" >&2
  echo "run this inside the svo:noetic container" >&2
  exit 1
fi

export ROS_DISTRO="${ROS_DISTRO:-noetic}"
set +u
source /opt/ros/noetic/setup.bash
set -u

if [[ ! -f "$src_dir/CMakeLists.txt" ]]; then
  catkin_init_workspace "$src_dir"
fi

catkin config --workspace "$ws_root" --extend /opt/ros/noetic \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3

if [[ ! -e "$src_dir/rpg_svo_pro_open" ]]; then
  ln -s "$upstream_dir" "$src_dir/rpg_svo_pro_open"
fi

if [[ ! -d "$src_dir/catkin_simple" ]]; then
  (
    cd "$src_dir"
    vcs import < "$deps_file"
  )
fi

dbow2_cmake="$src_dir/dbow2_catkin/CMakeLists.txt"
if [[ -f "$dbow2_cmake" ]]; then
  sed -i 's|git@github.com:dorian3d/DBoW2.git|https://github.com/dorian3d/DBoW2.git|g' "$dbow2_cmake"
fi

touch "$src_dir/minkindr/minkindr_python/CATKIN_IGNORE"

vocab_dir="$src_dir/rpg_svo_pro_open/svo_online_loopclosing/vocabularies"
if [[ ! -f "$vocab_dir/ORBvoc.dbow2" ]]; then
  (
    cd "$vocab_dir"
    bash ./download_voc.sh
  )
fi

echo "workspace ready: $ws_root"
