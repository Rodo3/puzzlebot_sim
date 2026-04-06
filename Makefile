.PHONY: build clean source rviz help

WS_ROOT := $(shell pwd)

help:
	@echo "Available commands:"
	@echo "  make build   Build all ROS 2 packages with colcon"
	@echo "  make clean   Remove build/, install/, and log/ directories"
	@echo "  make source  Print the command to source the workspace"
	@echo "  make rviz    Launch Puzzlebot simulation in RViz"

build:
	bash -c "source /opt/ros/humble/setup.bash && colcon build"

clean:
	rm -rf build/ install/ log/

source:
	@echo "Run: source /opt/ros/humble/setup.bash && source $(WS_ROOT)/install/setup.bash"

rviz:
	bash -c "source /opt/ros/humble/setup.bash && source $(WS_ROOT)/install/setup.bash && ros2 launch puzzlebot_bringup puzzlebot_launch.py"
