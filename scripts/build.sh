#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/humble/setup.bash

cd "${WS_ROOT}"

rosdep install --from-paths src --ignore-src -r -y

colcon build

echo "Build complete. Source the workspace with: source install/setup.bash"
