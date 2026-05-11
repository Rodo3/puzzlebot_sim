# puzzlebot_bringup

Owns launch files and runtime configuration.

Put here:
- Launch profiles for Gazebo, real robot, mapping, localization, and navigation.
- YAML parameter files.
- Small smoke-test or mock tools used to start the stack.

Do not put here:
- Algorithm implementations.
- Robot model files.
- Long-running production nodes, unless they are simple bringup utilities.

Launch files should choose between simulation and physical hardware by wiring
different input sources, not by duplicating algorithm code.
