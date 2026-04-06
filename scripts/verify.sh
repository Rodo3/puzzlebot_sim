#!/usr/bin/env bash
# Verification script for the puzzlebot_sim ROS 2 workspace.
# Run from the repo root: bash scripts/verify.sh

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WS_ROOT}"

PASS=0
FAIL=0
WARN=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()      { echo -e "  ${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
fail()    { echo -e "  ${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); }
warn()    { echo -e "  ${YELLOW}[WARN]${NC} $1"; WARN=$((WARN+1)); }
section() { echo -e "\n=== $1 ==="; }

# ─── 1. REPO STRUCTURE ───────────────────────────────────────────────────────
section "Repo structure"

for f in \
    src/puzzlebot_description/package.xml \
    src/puzzlebot_description/CMakeLists.txt \
    src/puzzlebot_description/urdf/puzzlebot.urdf \
    src/puzzlebot_description/rviz/puzzlebot_rviz.rviz \
    src/puzzlebot_bringup/package.xml \
    src/puzzlebot_bringup/launch/puzzlebot_launch.py \
    src/homework_01_transforms/package.xml \
    src/homework_01_transforms/homework_01_transforms/joint_state_publisher.py \
    src/puzzlebot_tf_tools/package.xml \
    src/shared_utils/package.xml \
    .gitignore \
    README.md \
    Makefile \
    scripts/build.sh \
    scripts/run_rviz.sh \
    scripts/source_ros.sh; do
  [[ -f "${f}" ]] && ok "${f}" || fail "${f} missing"
done

for f in docker/ .devcontainer/ .github/ scripts/enter_container.sh; do
  [[ ! -e "${f}" ]] && ok "Removed: ${f}" || fail "${f} should not exist"
done

# ─── 2. MESH FILES ───────────────────────────────────────────────────────────
section "Mesh files"

for m in \
    src/puzzlebot_description/meshes/Puzzlebot_Jetson_Lidar_Edition_Base.stl \
    src/puzzlebot_description/meshes/Puzzlebot_Wheel.stl \
    src/puzzlebot_description/meshes/Puzzlebot_Caster_Wheel.stl; do
  [[ -f "${m}" ]] && ok "${m}" || fail "${m} missing"
done

# ─── 3. SCRIPT PERMISSIONS ───────────────────────────────────────────────────
section "Script permissions (executable)"

for s in scripts/build.sh scripts/source_ros.sh scripts/run_rviz.sh scripts/verify.sh; do
  [[ -x "${s}" ]] && ok "${s}" || fail "${s} not executable"
done

# ─── 4. URDF MESH PATHS ──────────────────────────────────────────────────────
section "URDF mesh paths"

if grep -q "package://puzzlebot_description/meshes/" src/puzzlebot_description/urdf/puzzlebot.urdf; then
  ok "URDF uses package://puzzlebot_description/meshes/"
else
  fail "URDF still references old package name"
fi

if ! grep -q "package://puzzlebot_sim" src/puzzlebot_description/urdf/puzzlebot.urdf 2>/dev/null; then
  ok "URDF has no stale puzzlebot_sim references"
else
  fail "URDF contains stale package://puzzlebot_sim reference"
fi

# ─── 5. LAUNCH FILE REFERENCES ───────────────────────────────────────────────
section "Launch file package references"

LAUNCH="src/puzzlebot_bringup/launch/puzzlebot_launch.py"
grep -q "puzzlebot_description" "${LAUNCH}" \
  && ok "Launch references puzzlebot_description" || fail "Launch missing puzzlebot_description"
grep -q "homework_01_transforms" "${LAUNCH}" \
  && ok "Launch references homework_01_transforms" || fail "Launch missing homework_01_transforms"
if ! grep -q "package='puzzlebot_sim'" "${LAUNCH}" 2>/dev/null; then
  ok "Launch has no stale puzzlebot_sim node ref"
else
  fail "Launch still uses package='puzzlebot_sim'"
fi

# ─── 6. COLCON BUILD ─────────────────────────────────────────────────────────
section "colcon build (clean + rebuild)"

source /opt/ros/humble/setup.bash

rm -rf build/ install/ log/
if colcon build > /tmp/colcon_build.log 2>&1; then
  ok "colcon build succeeded"
  BUILT_PKGS=$(grep -oP '\d+(?= packages finished)' /tmp/colcon_build.log)
  [[ "${BUILT_PKGS}" == "5" ]] \
    && ok "All 5 packages built" \
    || warn "Expected 5 packages, got '${BUILT_PKGS}'"
else
  fail "colcon build failed — see /tmp/colcon_build.log"
  tail -20 /tmp/colcon_build.log
fi

# ─── 7. ROS PACKAGE DISCOVERY ────────────────────────────────────────────────
section "ROS package discovery"

source /opt/ros/humble/setup.bash
source "${WS_ROOT}/install/setup.bash"

for pkg in puzzlebot_description puzzlebot_bringup homework_01_transforms \
           puzzlebot_tf_tools shared_utils; do
  ros2 pkg list 2>/dev/null | grep -q "^${pkg}$" \
    && ok "ros2 pkg: ${pkg}" \
    || fail "ros2 pkg not found: ${pkg}"
done

# ─── 8. EXECUTABLES ──────────────────────────────────────────────────────────
section "ROS executables"

ros2 pkg executables homework_01_transforms 2>/dev/null | grep -q "joint_state_publisher" \
  && ok "homework_01_transforms: joint_state_publisher found" \
  || fail "homework_01_transforms: joint_state_publisher missing"

# ─── 9. PYTHON IMPORTS ───────────────────────────────────────────────────────
section "Python dependency imports"

for dep in rclpy tf2_ros transforms3d numpy; do
  python3 -c "import ${dep}" 2>/dev/null \
    && ok "import ${dep}" \
    || fail "import ${dep} failed — run: pip3 install ${dep}"
done

# ─── 10. MAKEFILE TARGETS ────────────────────────────────────────────────────
# Run last — make clean destroys install/
section "Makefile targets (non-GUI)"

make help > /dev/null 2>&1 && ok "make help" || fail "make help"
make build > /dev/null 2>&1 && ok "make build" || fail "make build"
make clean > /dev/null 2>&1 && [[ ! -d build/ ]] \
  && ok "make clean (removed build/)" || fail "make clean"
make source 2>&1 | grep -q "source" && ok "make source" || fail "make source"

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}PASS${NC}: ${PASS}   ${RED}FAIL${NC}: ${FAIL}   ${YELLOW}WARN${NC}: ${WARN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ${FAIL} -eq 0 ]]; then
  echo -e "  ${GREEN}All checks passed.${NC}"
  exit 0
else
  echo -e "  ${RED}${FAIL} check(s) failed. Fix the issues above.${NC}"
  exit 1
fi
