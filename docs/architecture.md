# Workspace Architecture

## Why a Multi-Package Workspace?

A single monolithic package works for small scripts but breaks down when:
- Multiple assignments share a robot model
- Team members work on different features in parallel
- CI needs to build and test components independently

This workspace separates concerns into focused packages that can be built, tested, and versioned independently.

## Package Responsibilities

### `puzzlebot_description`
Robot model only. Contains the URDF, STL meshes, and RViz configuration.  
**No Python code. No logic. No launch files.**  
Anything that needs to know the robot's geometry depends on this package.

### `puzzlebot_bringup`
Entry point for launching the simulation. Contains launch files that wire together
nodes from other packages. This is where you choose what to run and with what arguments.

### `homework_01_transforms`
Homework 1 deliverable. Implements a ROS 2 node that:
- Publishes TF frames: `map → odom → base_footprint`
- Simulates a circular differential drive trajectory
- Publishes wheel joint states for robot_state_publisher

### `puzzlebot_tf_tools`
Reusable TF utilities to be shared across homework packages. When you write a helper
(e.g., a quaternion utility, a pose listener) that more than one homework needs, it goes here.

### `shared_utils`
General-purpose Python helpers with no ROS dependency assumption. Math, geometry,
file I/O helpers, etc.

## Dependency Graph

```
puzzlebot_bringup
├── puzzlebot_description   (URDF + meshes)
└── homework_01_transforms  (TF publisher node)
    └── puzzlebot_tf_tools  (reusable TF helpers, optional)
        └── shared_utils    (pure Python helpers, optional)
```

## Scalability

Each new homework assignment becomes a new package under `src/`:
- Self-contained
- Its own dependencies
- Its own tests
- Launched via a new launch file in `puzzlebot_bringup`

This keeps the repository clean and makes it easy to find code for any given week.
