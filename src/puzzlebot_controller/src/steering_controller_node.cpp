#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cmath>
#include <vector>

static double norm_angle(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

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

    lookahead_  = get_parameter("lookahead_distance").as_double();
    max_v_      = get_parameter("max_linear_vel").as_double();
    max_w_      = get_parameter("max_angular_vel").as_double();
    goal_tol_   = get_parameter("goal_tolerance").as_double();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&SteeringControllerNode::odom_cb, this, std::placeholders::_1));

    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      "/planned_path", 1, std::bind(&SteeringControllerNode::path_cb, this, std::placeholders::_1));

    pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_in", 10);

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
    path_       = msg->poses;
    path_idx_   = 0;
    goal_reached_ = false;
    RCLCPP_INFO(get_logger(), "New path received: %zu waypoints", path_.size());
  }

  void control_loop()
  {
    geometry_msgs::msg::Twist cmd;

    if (!have_pose_ || path_.empty() || goal_reached_) {
      pub_cmd_->publish(cmd);  // zero velocity
      return;
    }

    // Check if final goal reached
    const auto & goal = path_.back().pose.position;
    double dist_to_goal = std::hypot(goal.x - robot_x_, goal.y - robot_y_);
    if (dist_to_goal < goal_tol_) {
      goal_reached_ = true;
      RCLCPP_INFO(get_logger(), "Goal reached (dist=%.3f m)", dist_to_goal);
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

    // Slow down when heading angle error is large
    double heading_err = std::atan2(local_y, local_x);
    if (std::abs(heading_err) > M_PI / 4.0)
      v *= 0.5;

    cmd.linear.x  = v;
    cmd.angular.z = w;
    pub_cmd_->publish(cmd);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     sub_path_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr  pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<geometry_msgs::msg::PoseStamped> path_;
  int    path_idx_{0};
  bool   goal_reached_{false};
  bool   have_pose_{false};

  double robot_x_{0.0}, robot_y_{0.0}, robot_th_{0.0};
  double lookahead_, max_v_, max_w_, goal_tol_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SteeringControllerNode>());
  rclcpp::shutdown();
  return 0;
}
