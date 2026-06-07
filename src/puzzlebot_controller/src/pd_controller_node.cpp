#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cmath>

static double norm_angle(double a)
{
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

/**
 * PD controller for a differential-drive robot.
 *
 * Subscribes to /odom and /goal_pose.
 * Publishes Twist to /cmd_vel_in (obstacle_avoidance sits downstream).
 *
 * Control law (robot-frame heading error):
 *   e_dist  = euclidean distance to goal
 *   e_ang   = atan2(Δy, Δx) − θ_robot   (normalised to [-π, π])
 *
 *   v = Kp_lin * e_dist   (clamped, zeroed when |e_ang| > pi/2)
 *   ω = Kp_ang * e_ang  +  Kd_ang * (e_ang − e_ang_prev) / dt
 *
 * Debug topic /controller/debug publishes
 *   [e_dist, e_ang, v_cmd, omega_cmd]
 * so you can watch all four signals live in rqt_plot while tuning.
 */
class PDControllerNode : public rclcpp::Node
{
public:
  PDControllerNode() : Node("pd_controller_node")
  {
    declare_parameter("Kp_linear",      0.5);
    declare_parameter("Kp_angular",     2.0);
    declare_parameter("Kd_angular",     0.1);
    declare_parameter("goal_tolerance", 0.05);   // metres
    declare_parameter("max_linear_vel", 0.20);   // m/s  — conservative for real robot
    declare_parameter("max_angular_vel", 1.20);  // rad/s
    declare_parameter("control_freq",   20.0);   // Hz

    Kp_lin_   = get_parameter("Kp_linear").as_double();
    Kp_ang_   = get_parameter("Kp_angular").as_double();
    Kd_ang_   = get_parameter("Kd_angular").as_double();
    goal_tol_ = get_parameter("goal_tolerance").as_double();
    max_v_    = get_parameter("max_linear_vel").as_double();
    max_w_    = get_parameter("max_angular_vel").as_double();
    double hz = get_parameter("control_freq").as_double();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&PDControllerNode::odom_cb, this, std::placeholders::_1));

    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      std::bind(&PDControllerNode::goal_cb, this, std::placeholders::_1));

    sub_cancel_ = create_subscription<std_msgs::msg::Bool>(
      "/navigation/cancel", 10,
      std::bind(&PDControllerNode::cancel_cb, this, std::placeholders::_1));

    sub_goal_reached_ = create_subscription<std_msgs::msg::Bool>(
      "/navigation/goal_reached", 10,
      std::bind(&PDControllerNode::cancel_cb, this, std::placeholders::_1));

    pub_cmd_   = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_in", 10);
    pub_debug_ = create_publisher<std_msgs::msg::Float32MultiArray>("/controller/debug", 10);
    pub_goal_reached_ = create_publisher<std_msgs::msg::Bool>("/navigation/goal_reached", 10);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / hz)),
      std::bind(&PDControllerNode::control_loop, this));

    last_time_ = now();

    RCLCPP_INFO(get_logger(),
      "pd_controller_node started — Kp_lin=%.2f Kp_ang=%.2f Kd_ang=%.3f",
      Kp_lin_, Kp_ang_, Kd_ang_);
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    robot_x_ = msg->pose.pose.position.x;
    robot_y_ = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;
    robot_th_ = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    have_pose_ = true;
  }

  void goal_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    goal_x_       = msg->pose.position.x;
    goal_y_       = msg->pose.position.y;
    have_goal_    = true;
    goal_reached_ = false;
    prev_e_ang_   = 0.0;   // reset derivative on new goal
    RCLCPP_INFO(get_logger(), "New goal: (%.2f, %.2f)", goal_x_, goal_y_);
  }

  void cancel_cb(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data) return;
    have_goal_ = false;
    goal_reached_ = true;
    was_active_ = false;
    prev_e_ang_ = 0.0;
    RCLCPP_INFO(get_logger(), "Goal cleared by navigation event");
  }

  void control_loop()
  {
    geometry_msgs::msg::Twist cmd;  // zero-initialised

    auto now_t = now();
    double dt  = (now_t - last_time_).seconds();
    last_time_ = now_t;

    if (dt <= 0.0 || dt > 0.5) {
      // Reloj inestable: manda un stop de cortesía y cede el bus
      if (was_active_) { was_active_ = false; pub_cmd_->publish(cmd); }
      return;
    }

    if (!have_pose_ || !have_goal_ || goal_reached_) {
      // Sin goal activo: NO publicar — deja que teleop (u otro nodo) mande /cmd_vel
      // Se envía un único stop al perder el goal para frenar de forma segura.
      if (was_active_) { was_active_ = false; pub_cmd_->publish(cmd); }
      return;
    }

    was_active_ = true;

    double dx      = goal_x_ - robot_x_;
    double dy      = goal_y_ - robot_y_;
    double e_dist  = std::hypot(dx, dy);

    if (e_dist < goal_tol_) {
      goal_reached_ = true;
      have_goal_ = false;
      RCLCPP_INFO(get_logger(), "Goal reached (dist=%.3f m)", e_dist);
      std_msgs::msg::Bool reached;
      reached.data = true;
      pub_goal_reached_->publish(reached);
      publish_cmd(cmd, e_dist, 0.0, 0.0, 0.0);
      return;
    }

    double e_ang  = norm_angle(std::atan2(dy, dx) - robot_th_);

    // Derivative term on angular error
    double de_ang = (dt > 0.0) ? norm_angle(e_ang - prev_e_ang_) / dt : 0.0;
    prev_e_ang_   = e_ang;

    // Control outputs
    double v = Kp_lin_ * e_dist;
    double w = Kp_ang_ * e_ang + Kd_ang_ * de_ang;

    // Don't drive forward when pointing away from the goal (> 90°) — turn first.
    if (std::abs(e_ang) > M_PI_2)
      v = 0.0;

    v = std::clamp(v, 0.0,   max_v_);
    w = std::clamp(w, -max_w_, max_w_);

    cmd.linear.x  = v;
    cmd.angular.z = w;

    publish_cmd(cmd, e_dist, e_ang, v, w);
  }

  void publish_cmd(const geometry_msgs::msg::Twist & cmd,
                   double e_dist, double e_ang, double v, double w)
  {
    pub_cmd_->publish(cmd);

    std_msgs::msg::Float32MultiArray dbg;
    dbg.data = {
      static_cast<float>(e_dist),
      static_cast<float>(e_ang),
      static_cast<float>(v),
      static_cast<float>(w)
    };
    pub_debug_->publish(dbg);
  }

  // Subscriptions / publishers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr             sub_cancel_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr             sub_goal_reached_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr           pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr    pub_debug_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                 pub_goal_reached_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_time_;

  // Robot state
  double robot_x_{0.0}, robot_y_{0.0}, robot_th_{0.0};
  bool   have_pose_{false};

  // Goal state
  double goal_x_{0.0}, goal_y_{0.0};
  bool   have_goal_{false};
  bool   goal_reached_{false};

  // Controller state
  double prev_e_ang_{0.0};
  bool   was_active_{false};   // true mientras hay goal activo en curso

  // Gains & limits
  double Kp_lin_, Kp_ang_, Kd_ang_;
  double goal_tol_, max_v_, max_w_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PDControllerNode>());
  rclcpp::shutdown();
  return 0;
}
