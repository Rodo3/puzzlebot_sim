# puzzlebot_msgs

Owns custom ROS interfaces shared by multiple packages.

Put here:
- `.msg`, `.srv`, and `.action` definitions.
- Interface-only changes that multiple packages depend on.

Do not put here:
- Node implementations.
- Algorithm code.
- Package-specific private data structures.

Keep custom messages minimal. Prefer standard ROS messages unless a custom type
clearly improves the interface.
