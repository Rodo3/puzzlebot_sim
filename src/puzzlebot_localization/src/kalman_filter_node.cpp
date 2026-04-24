#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <array>
#include <cmath>

// Simple 3x3 matrix operations (row-major)
using Mat3 = std::array<double, 9>;

static Mat3 mat_add(const Mat3 & A, const Mat3 & B) {
  Mat3 C; for (int i = 0; i < 9; ++i) C[i] = A[i] + B[i]; return C;
}
static Mat3 mat_mul(const Mat3 & A, const Mat3 & B) {
  Mat3 C{};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        C[r*3+c] += A[r*3+k] * B[k*3+c];
  return C;
}
static Mat3 mat_transpose(const Mat3 & A) {
  return {A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8]};
}
static Mat3 mat_inv3(const Mat3 & A) {
  // Analytic 3x3 inverse
  double det = A[0]*(A[4]*A[8]-A[5]*A[7])
             - A[1]*(A[3]*A[8]-A[5]*A[6])
             + A[2]*(A[3]*A[7]-A[4]*A[6]);
  double id = 1.0 / det;
  return {
     id*(A[4]*A[8]-A[5]*A[7]), id*(A[2]*A[7]-A[1]*A[8]), id*(A[1]*A[5]-A[2]*A[4]),
     id*(A[5]*A[6]-A[3]*A[8]), id*(A[0]*A[8]-A[2]*A[6]), id*(A[2]*A[3]-A[0]*A[5]),
     id*(A[3]*A[7]-A[4]*A[6]), id*(A[1]*A[6]-A[0]*A[7]), id*(A[0]*A[4]-A[1]*A[3])
  };
}
static double norm_angle(double a) {
  while (a >  M_PI) a -= 2.0*M_PI;
  while (a < -M_PI) a += 2.0*M_PI;
  return a;
}

class KalmanFilterNode : public rclcpp::Node
{
public:
  KalmanFilterNode() : Node("kalman_filter_node")
  {
    declare_parameter("process_noise_x",     0.01);
    declare_parameter("process_noise_y",     0.01);
    declare_parameter("process_noise_theta", 0.005);
    declare_parameter("meas_noise_x",        0.05);
    declare_parameter("meas_noise_y",        0.05);
    declare_parameter("meas_noise_theta",    0.01);
    declare_parameter("initial_covariance",  0.1);

    double ic = get_parameter("initial_covariance").as_double();
    x_  = {0.0, 0.0, 0.0};
    P_  = {ic,0,0, 0,ic,0, 0,0,ic};

    double qx  = get_parameter("process_noise_x").as_double();
    double qy  = get_parameter("process_noise_y").as_double();
    double qt  = get_parameter("process_noise_theta").as_double();
    Q_ = {qx,0,0, 0,qy,0, 0,0,qt};

    double rx  = get_parameter("meas_noise_x").as_double();
    double ry  = get_parameter("meas_noise_y").as_double();
    double rt  = get_parameter("meas_noise_theta").as_double();
    R_ = {rx,0,0, 0,ry,0, 0,0,rt};

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom_raw", 10,
      std::bind(&KalmanFilterNode::odom_cb, this, std::placeholders::_1));

    sub_aruco_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/aruco/poses", 10,
      std::bind(&KalmanFilterNode::aruco_cb, this, std::placeholders::_1));

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    last_time_ = now();
    RCLCPP_INFO(get_logger(), "kalman_filter_node started");
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    auto t = rclcpp::Time(msg->header.stamp);
    double dt = (t - last_time_).seconds();
    last_time_ = t;
    if (dt <= 0.0 || dt > 1.0) { publish(); return; }

    double v   = msg->twist.twist.linear.x;
    double w   = msg->twist.twist.angular.z;
    double th  = x_[2];

    double delta_d  = v * dt;
    double delta_th = w * dt;

    // Prediction
    x_[0] += delta_d * std::cos(th);
    x_[1] += delta_d * std::sin(th);
    x_[2]  = norm_angle(x_[2] + delta_th);

    // Jacobian F
    Mat3 F = {1,0,-delta_d*std::sin(th),
              0,1, delta_d*std::cos(th),
              0,0,1};
    P_ = mat_add(mat_mul(mat_mul(F, P_), mat_transpose(F)), Q_);

    publish();
  }

  void aruco_cb(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->poses.empty()) return;

    // Use first detected marker as a direct pose measurement
    const auto & p = msg->poses[0];
    tf2::Quaternion q(p.orientation.x, p.orientation.y,
                      p.orientation.z, p.orientation.w);
    double z[3] = {p.position.x, p.position.y, tf2::getYaw(q)};

    // H = I (direct measurement), S = P + R, K = P * S^-1
    Mat3 S = mat_add(P_, R_);
    Mat3 K = mat_mul(P_, mat_inv3(S));

    double inn[3] = {z[0]-x_[0], z[1]-x_[1], norm_angle(z[2]-x_[2])};
    x_[0] += K[0]*inn[0] + K[1]*inn[1] + K[2]*inn[2];
    x_[1] += K[3]*inn[0] + K[4]*inn[1] + K[5]*inn[2];
    x_[2]  = norm_angle(x_[2] + K[6]*inn[0] + K[7]*inn[1] + K[8]*inn[2]);

    // P = (I - K) P
    Mat3 IK = {1-K[0],-K[1],-K[2], -K[3],1-K[4],-K[5], -K[6],-K[7],1-K[8]};
    P_ = mat_mul(IK, P_);

    publish();
  }

  void publish()
  {
    tf2::Quaternion q;
    q.setRPY(0, 0, x_[2]);

    auto odom = nav_msgs::msg::Odometry();
    odom.header.stamp    = now();
    odom.header.frame_id = "odom";
    odom.child_frame_id  = "base_footprint";
    odom.pose.pose.position.x  = x_[0];
    odom.pose.pose.position.y  = x_[1];
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    // Fill covariance diagonal
    odom.pose.covariance[0]  = P_[0];
    odom.pose.covariance[7]  = P_[4];
    odom.pose.covariance[35] = P_[8];
    pub_odom_->publish(odom);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = odom.header;
    tf_msg.child_frame_id = "base_footprint";
    tf_msg.transform.translation.x = x_[0];
    tf_msg.transform.translation.y = x_[1];
    tf_msg.transform.rotation      = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_aruco_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::array<double, 3> x_;
  Mat3 P_, Q_, R_;
  rclcpp::Time last_time_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}
