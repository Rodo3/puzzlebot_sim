#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

/**
 * Pure-pursuit steering controller.
 * Subscribes to /odom (filtered pose) and /planned_path.
 * Publishes Twist to /cmd_vel_in (obstacle avoidance node sits downstream).
 *
 * Pure pursuit chooses a look-ahead point on the path and computes the
 * curvature needed to reach it, giving smooth curved trajectories.
 */
class SteeringControllerNode : public rclcpp::Node
{
public:
  SteeringControllerNode() : Node("steering_controller_node")
  {
    declare_parameter("lookahead_distance",  0.30);   // metres
    declare_parameter("max_linear_vel",      0.30);   // m/s
    declare_parameter("max_angular_vel",     1.50);   // rad/s
    declare_parameter("goal_tolerance",      0.10);   // metres — stop when reached
    declare_parameter("control_frequency",  20.0);   // Hz
    declare_parameter("path_timeout_sec",    3.0);   // s — clear path if no update

    lookahead_     = get_parameter("lookahead_distance").as_double();
    max_v_         = get_parameter("max_linear_vel").as_double();
    max_w_         = get_parameter("max_angular_vel").as_double();
    goal_tol_      = get_parameter("goal_tolerance").as_double();
    path_timeout_  = get_parameter("path_timeout_sec").as_double();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&SteeringControllerNode::odom_cb, this, std::placeholders::_1));

    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      "/planned_path", 1, std::bind(&SteeringControllerNode::path_cb, this, std::placeholders::_1));

    sub_cancel_ = create_subscription<std_msgs::msg::Bool>(
      "/navigation/cancel", 10, std::bind(&SteeringControllerNode::cancel_cb, this, std::placeholders::_1));

    pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_in", 10);
    pub_goal_reached_ = create_publisher<std_msgs::msg::Bool>("/navigation/goal_reached", 10);

    double hz = get_parameter("control_frequency").as_double();
    timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / hz)),
      std::bind(&SteeringControllerNode::control_loop, this));

    RCLCPP_INFO(get_logger(), "steering_controller_node started (pure pursuit, lookahead=%.2f m)", lookahead_);
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    robot_x_  = msg->pose.pose.position.x;
    robot_y_  = msg->pose.pose.position.y;
    // Extract yaw from quaternion
    auto & q  = msg->pose.pose.orientation;
    robot_th_ = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                           1.0 - 2.0*(q.y*q.y + q.z*q.z));
    have_pose_ = true;
  }

  void path_cb(const nav_msgs::msg::Path::SharedPtr msg)
  {
    last_path_time_ = get_clock()->now();
    path_           = msg->poses;
    goal_reached_   = false;

    // Al recibir ruta nueva, empezar desde el waypoint más cercano al robot.
    // Esto evita que el controlador intente ir hacia puntos detrás del robot
    // cuando el A* replaneó mientras el robot estaba en movimiento.
    path_idx_ = 0;
    if (have_pose_ && path_.size() > 1) {
      double best_dist = std::numeric_limits<double>::max();
      for (int i = 0; i < static_cast<int>(path_.size()); ++i) {
        double dx = path_[i].pose.position.x - robot_x_;
        double dy = path_[i].pose.position.y - robot_y_;
        double d  = std::hypot(dx, dy);
        if (d < best_dist) {
          best_dist = d;
          path_idx_ = i;
        }
      }
      // No ir más allá del penúltimo punto para que el goal final sea correcto
      path_idx_ = std::min(path_idx_, static_cast<int>(path_.size()) - 2);
    }

    RCLCPP_INFO(get_logger(), "New path received: %zu waypoints (starting at idx %d)",
                path_.size(), path_idx_);
  }

  void cancel_cb(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data) return;
    path_.clear();
    path_idx_ = 0;
    goal_reached_ = true;
    pub_cmd_->publish(geometry_msgs::msg::Twist());
    RCLCPP_INFO(get_logger(), "Navigation cancelled — path cleared");
  }

  void control_loop()
  {
    geometry_msgs::msg::Twist cmd;

    // If no path update has arrived recently, the planner may have crashed.
    // Clear the stale path and stop rather than following indefinitely.
    // Only applies once a path has been received (last_path_time_ > 0).
    if (!path_.empty() && !goal_reached_ && last_path_time_.nanoseconds() > 0) {
      double age = (get_clock()->now() - last_path_time_).seconds();
      if (age > path_timeout_) {
        RCLCPP_WARN(get_logger(), "Path timeout (%.1f s > %.1f s) — stopping", age, path_timeout_);
        path_.clear();
      }
    }

    if (!have_pose_ || path_.empty() || goal_reached_) {
      pub_cmd_->publish(cmd);  // zero velocity
      return;
    }

    // Check if final goal reached
    const auto & goal = path_.back().pose.position;
    double dist_to_goal = std::hypot(goal.x - robot_x_, goal.y - robot_y_);
    if (dist_to_goal < goal_tol_) {
      goal_reached_ = true;
      path_.clear();
      RCLCPP_INFO(get_logger(), "Goal reached (dist=%.3f m)", dist_to_goal);
      std_msgs::msg::Bool reached;
      reached.data = true;
      pub_goal_reached_->publish(reached);
      pub_cmd_->publish(cmd);
      return;
    }

    // Advance path_idx_ past points already within lookahead
    while (path_idx_ < static_cast<int>(path_.size()) - 1) {
      double dx = path_[path_idx_].pose.position.x - robot_x_;
      double dy = path_[path_idx_].pose.position.y - robot_y_;
      if (std::hypot(dx, dy) > lookahead_) break;
      ++path_idx_;
    }

    // Look-ahead point
    double lx = path_[path_idx_].pose.position.x;
    double ly = path_[path_idx_].pose.position.y;

    // Transform look-ahead point to robot frame
    double dx = lx - robot_x_;
    double dy = ly - robot_y_;
    double local_x =  dx * std::cos(robot_th_) + dy * std::sin(robot_th_);
    double local_y = -dx * std::sin(robot_th_) + dy * std::cos(robot_th_);

    double ld2 = local_x * local_x + local_y * local_y;
    if (ld2 < 1e-6) {
      pub_cmd_->publish(cmd);
      return;
    }

    // Pure pursuit curvature: κ = 2y / L²
    double curvature = 2.0 * local_y / ld2;
    double v = max_v_;
    double w = std::clamp(v * curvature, -max_w_, max_w_);

    // Reducir velocidad lineal proporcionalmente al ángulo de giro requerido.
    // A 0° error → velocidad máxima. A 90° error → velocidad mínima (20%).
    // Esto evita que el robot choque cuando recibe una ruta que gira bruscamente.
    double heading_err = std::atan2(local_y, local_x);
    double abs_err = std::abs(heading_err);
    double speed_scale = 1.0;
    if (abs_err > 0.35) {  // > ~20 grados
      // Escala lineal: 20° → 1.0,  90° → 0.20
      speed_scale = 1.0 - 0.8 * ((abs_err - 0.35) / (M_PI / 2.0 - 0.35));
      speed_scale = std::max(0.20, std::min(1.0, speed_scale));
    }
    v *= speed_scale;

    // Freno adicional cerca del goal final
    if (dist_to_goal < 0.40)
      v = std::min(v, max_v_ * (dist_to_goal / 0.40));

    cmd.linear.x  = std::max(0.0, v);
    cmd.angular.z = w;
    pub_cmd_->publish(cmd);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     sub_path_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr     sub_cancel_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr  pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr        pub_goal_reached_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<geometry_msgs::msg::PoseStamped> path_;
  int    path_idx_{0};
  bool   goal_reached_{false};
  bool   have_pose_{false};

  double robot_x_{0.0}, robot_y_{0.0}, robot_th_{0.0};
  double lookahead_, max_v_, max_w_, goal_tol_, path_timeout_;
  rclcpp::Time last_path_time_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SteeringControllerNode>());
  rclcpp::shutdown();
  return 0;
}
