"""Centralizes default topic names and rate limits (Hz) for the bridge."""

DEFAULT_TOPICS = {
    # Inbound (ROS → WebSocket)
    'odom':                   '/odom',
    'scan':                   '/scan',
    'map':                    '/map',
    'cmd_vel':                '/cmd_vel',
    'cmd_vel_in':             '/cmd_vel_in',
    'cmd_vel_steering':       '/cmd_vel_steering',   # steering controller output (pre-DOM)
    'dom_state':              '/dom/state',           # dynamic_obstacle_manager FSM state
    'augmented_map':          '/augmented_map',       # map + injected dynamic obstacles
    'camera':                 '/camera/image/compressed',
    'voice_command':          '/voice/command',
    'voice_confidence':       '/voice/confidence',
    'voice_status':           '/voice/status',
    'voice_ranked_predictions': '/voice/ranked_predictions',
    'voice_inference_time':   '/voice/inference_time_ms',
    # Outbound (WebSocket → ROS) — control topics
    'cmd_vel_out':            '/cmd_vel',          # teleop; override in Gazebo to /model/puzzlebot/cmd_vel
    'goal_pose':              '/goal_pose',
    'navigate_to_waypoint':   '/navigate_to_waypoint',
    'slam_reset':             '/slam/reset',
}

# Maximum publish rate to WebSocket clients (Hz).
# Voice topics are event-driven so they have no rate limit (0 = unlimited).
RATE_LIMITS_HZ = {
    'odom':                     10.0,
    'cmd_vel':                  10.0,
    'cmd_vel_in':               10.0,
    'cmd_vel_steering':         10.0,
    'scan':                      5.0,
    'map':                       1.0,
    'augmented_map':             1.0,
    'dom_state':                 5.0,
    'camera':                   10.0,
    'voice_command':             0.0,
    'voice_confidence':          0.0,
    'voice_status':              0.0,
    'voice_ranked_predictions':  0.0,
    'voice_inference_time':      0.0,
}

WEBSOCKET_HOST_DEFAULT = '0.0.0.0'
WEBSOCKET_PORT_DEFAULT = 8000
