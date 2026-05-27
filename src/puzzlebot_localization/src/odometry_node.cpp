#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>
#include <string>

class OdometryNode : public rclcpp::Node
{
public:
  OdometryNode() : Node("odometry_node"), x_(0.0), y_(0.0), theta_(0.0),
                   vr_(0.0), vl_(0.0), have_input_(false)
  {
    declare_parameter("wheel_radius", 0.05);
    declare_parameter("wheel_separation", 0.18);
    declare_parameter("input_source", "encoders");
    declare_parameter("odom_topic", "odom_raw");
    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_footprint");
    declare_parameter("publish_tf", false);
    declare_parameter("input_timeout", 0.5);
    declare_parameter("left_joint_name", "wheel_l_joint");
    declare_parameter("right_joint_name", "wheel_r_joint");

    r_ = get_parameter("wheel_radius").as_double();
    l_ = get_parameter("wheel_separation").as_double();
    input_source_ = get_parameter("input_source").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    input_timeout_ = get_parameter("input_timeout").as_double();
    left_joint_name_ = get_parameter("left_joint_name").as_string();
    right_joint_name_ = get_parameter("right_joint_name").as_string();

    if (input_source_ == "joint_states") {
      sub_joint_states_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10,
        std::bind(&OdometryNode::joint_states_cb, this, std::placeholders::_1));
    } else if (input_source_ == "encoders") {
      // micro-ROS siempre publica BEST_EFFORT — usar SensorDataQoS para compatibilidad
      auto enc_qos = rclcpp::SensorDataQoS();
      sub_r_ = create_subscription<std_msgs::msg::Float32>(
        "velocity_enc_r", enc_qos,
        [this](const std_msgs::msg::Float32::SharedPtr msg) {
          vr_ = msg->data;
          mark_input();
        });

      sub_l_ = create_subscription<std_msgs::msg::Float32>(
        "velocity_enc_l", enc_qos,
        [this](const std_msgs::msg::Float32::SharedPtr msg) {
          vl_ = msg->data;
          mark_input();
        });
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Unknown input_source '%s'; falling back to encoder topics",
        input_source_.c_str());
      input_source_ = "encoders";
      auto enc_qos = rclcpp::SensorDataQoS();
      sub_r_ = create_subscription<std_msgs::msg::Float32>(
        "velocity_enc_r", enc_qos,
        [this](const std_msgs::msg::Float32::SharedPtr msg) {
          vr_ = msg->data;
          mark_input();
        });
      sub_l_ = create_subscription<std_msgs::msg::Float32>(
        "velocity_enc_l", enc_qos,
        [this](const std_msgs::msg::Float32::SharedPtr msg) {
          vl_ = msg->data;
          mark_input();
        });
    }

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&OdometryNode::update, this));

    last_time_ = now();
    last_input_time_ = last_time_;
    RCLCPP_INFO(
      get_logger(),
      "odometry_node started (input=%s, topic=%s, r=%.3f m, l=%.3f m, publish_tf=%s)",
      input_source_.c_str(), odom_topic_.c_str(), r_, l_, publish_tf_ ? "true" : "false");
  }

private:
  void mark_input()
  {
    last_input_time_ = now();
    have_input_ = true;
  }

  void joint_states_cb(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    bool found_left = false;
    bool found_right = false;
    double left_velocity = 0.0;
    double right_velocity = 0.0;

    const size_t count = std::min(msg->name.size(), msg->velocity.size());
    for (size_t i = 0; i < count; ++i) {
      if (msg->name[i] == left_joint_name_) {
        left_velocity = msg->velocity[i];
        found_left = true;
      } else if (msg->name[i] == right_joint_name_) {
        right_velocity = msg->velocity[i];
        found_right = true;
      }
    }

    if (!found_left || !found_right) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "JointState missing '%s' or '%s'",
        left_joint_name_.c_str(), right_joint_name_.c_str());
      return;
    }

    vl_ = left_velocity;
    vr_ = right_velocity;
    mark_input();
  }

  void update()
  {
    auto current_time = now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;
    if (dt <= 0.0 || dt > 1.0) {
      publish(0.0, 0.0, current_time);
      return;
    }

    if (!have_input_) {
      publish(0.0, 0.0, current_time);
      return;
    }

    const bool stale_input =
      input_timeout_ > 0.0 &&
      (current_time - last_input_time_).seconds() > input_timeout_;
    const double vr = stale_input ? 0.0 : vr_;
    const double vl = stale_input ? 0.0 : vl_;

    // Convert encoder angular velocities [rad/s] → linear wheel velocities [m/s]
    double vR = vr * r_;
    double vL = vl * r_;

    double v     = (vR + vL) / 2.0;
    double omega = (vR - vL) / l_;

    // Integración de punto medio (arco de círculo exacto para drive diferencial).
    // Euler puro acumula error O(delta_theta²) en curvas; el punto medio lo reduce a O(delta_theta³).
    double delta_d     = v * dt;
    double delta_theta = omega * dt;
    double theta_mid   = theta_ + delta_theta * 0.5;

    x_     += delta_d * std::cos(theta_mid);
    y_     += delta_d * std::sin(theta_mid);
    theta_ += delta_theta;

    // Normalise angle to [-π, π]
    while (theta_ >  M_PI) theta_ -= 2.0 * M_PI;
    while (theta_ < -M_PI) theta_ += 2.0 * M_PI;

    publish(v, omega, current_time);
  }

  void publish(double v, double omega, const rclcpp::Time & stamp)
  {
    // Publish odometry
    auto odom = nav_msgs::msg::Odometry();
    odom.header.stamp    = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id  = base_frame_;

    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;

    tf2::Quaternion q;
    q.setRPY(0, 0, theta_);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x  = v;
    odom.twist.twist.angular.z = omega;

    pub_odom_->publish(odom);

    if (!publish_tf_) {
      return;
    }

    // Broadcast odom → base TF only when this node is the selected odometry owner.
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp    = stamp;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id  = base_frame_;
    tf_msg.transform.translation.x = x_;
    tf_msg.transform.translation.y = y_;
    tf_msg.transform.rotation      = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  // Subscriptions
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_r_, sub_l_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
  // Publisher
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_time_, last_input_time_;

  // State
  double x_, y_, theta_;
  double vr_, vl_;   // raw encoder angular velocities [rad/s]
  double r_, l_;     // wheel radius, wheel separation [m]
  bool have_input_;
  bool publish_tf_;
  double input_timeout_;
  std::string input_source_;
  std::string odom_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string left_joint_name_;
  std::string right_joint_name_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdometryNode>());
  rclcpp::shutdown();
  return 0;
}
