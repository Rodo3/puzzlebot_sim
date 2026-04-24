#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>

class OdometryNode : public rclcpp::Node
{
public:
  OdometryNode() : Node("odometry_node"), x_(0.0), y_(0.0), theta_(0.0),
                   vr_(0.0), vl_(0.0)
  {
    declare_parameter("wheel_radius", 0.05);
    declare_parameter("wheelbase",    0.18);

    r_ = get_parameter("wheel_radius").as_double();
    l_ = get_parameter("wheelbase").as_double();

    sub_r_ = create_subscription<std_msgs::msg::Float32>(
      "/velocity_enc_r", 10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) { vr_ = msg->data; });

    sub_l_ = create_subscription<std_msgs::msg::Float32>(
      "/velocity_enc_l", 10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) { vl_ = msg->data; });

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/odom_raw", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&OdometryNode::update, this));

    last_time_ = now();
    RCLCPP_INFO(get_logger(), "odometry_node started (r=%.3f m, l=%.3f m)", r_, l_);
  }

private:
  void update()
  {
    auto current_time = now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    // Convert encoder angular velocities [rad/s] → linear wheel velocities [m/s]
    double vR = vr_ * r_;
    double vL = vl_ * r_;

    double v     = (vR + vL) / 2.0;
    double omega = (vR - vL) / l_;

    // Discrete integration
    double delta_d     = v * dt;
    double delta_theta = omega * dt;

    x_     += delta_d * std::cos(theta_);
    y_     += delta_d * std::sin(theta_);
    theta_ += delta_theta;

    // Normalise angle to [-π, π]
    while (theta_ >  M_PI) theta_ -= 2.0 * M_PI;
    while (theta_ < -M_PI) theta_ += 2.0 * M_PI;

    // Publish odometry
    auto odom = nav_msgs::msg::Odometry();
    odom.header.stamp    = current_time;
    odom.header.frame_id = "odom";
    odom.child_frame_id  = "base_footprint";

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

    // Broadcast odom → base_footprint TF
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp    = current_time;
    tf_msg.header.frame_id = "odom";
    tf_msg.child_frame_id  = "base_footprint";
    tf_msg.transform.translation.x = x_;
    tf_msg.transform.translation.y = y_;
    tf_msg.transform.rotation      = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  // Subscriptions
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_r_, sub_l_;
  // Publisher
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_time_;

  // State
  double x_, y_, theta_;
  double vr_, vl_;   // raw encoder angular velocities [rad/s]
  double r_, l_;     // wheel radius, wheelbase [m]
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdometryNode>());
  rclcpp::shutdown();
  return 0;
}
