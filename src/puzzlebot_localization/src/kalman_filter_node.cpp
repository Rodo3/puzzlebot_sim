#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <array>
#include <cmath>
#include <string>

/**
 * kalman_filter_node — Filtro de Kalman Extendido (EKF) para localización del Puzzlebot.
 *
 * POSICIÓN EN EL PIPELINE:
 *   odometry_node → /odom_raw ─────────────────┐
 *   aruco_node    → /aruco/pose ────────────────┤→ [ESTE NODO] → /odom → steering_controller
 *   slam_node     → /scan_match/pose ───────────┘              → TF odom→base_footprint
 *
 * FUNCIÓN:
 *   Fusiona tres fuentes de información para estimar la pose del robot (x, y, θ):
 *
 *   1. PREDICCIÓN  (odometría /odom_raw):
 *      Modelo cinemático diferencial: integra velocidad lineal y angular.
 *      ZUPT: cuando el robot está quieto (|v|+|ω| < threshold), no crece P.
 *
 *   2. CORRECCIÓN PRIMARIA  (ArUco /aruco/pose):
 *      Medición absoluta de pose; inicializa el estado en la primera detección.
 *      Floor de ruido (meas_noise_*) para no confiar demasiado en solvePnP.
 *
 *   3. CORRECCIÓN SECUNDARIA  (scan match /scan_match/pose):
 *      Refinamiento adicional desde slam_node cuando el matcher es confiable.
 *      Solo fusiona si ArUco ya inicializó el estado (no puede bootstrapear solo).
 *
 * TOPICS SUSCRITOS:
 *   /odom_raw        (nav_msgs/Odometry)                     — odometría cruda
 *   /aruco/pose      (geometry_msgs/PoseWithCovarianceStamped) — pose absoluta ArUco
 *   /scan_match/pose (geometry_msgs/PoseWithCovarianceStamped) — corrección de SLAM
 *
 * TOPICS PUBLICADOS:
 *   /odom                    (nav_msgs/Odometry)  — pose filtrada del robot
 *   /localization/status     (std_msgs/String)    — "INITIALIZING" | "OK" | "LOST"
 *   /localization/covariance (std_msgs/Float32)   — trace(P_xy) actual en m²
 *   TF odom→base_footprint                        — necesario para TF tree
 *
 * PARÁMETROS CLAVE:
 *   init_from_aruco      [true]  — espera el primer ArUco para inicializar
 *   zupt_speed_threshold [0.02]  — umbral de velocidad para ZUPT
 *   p_max_xy / p_max_theta       — techo de covarianza para evitar explosión de P
 */

// Operaciones matriciales 3×3 (almacenamiento row-major)
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
    declare_parameter("meas_noise_theta",    0.07);
    declare_parameter("initial_covariance",  0.1);
    declare_parameter("initial_x",     0.0);
    declare_parameter("initial_y",     0.0);
    declare_parameter("initial_theta", 0.0);
    declare_parameter("init_from_aruco", true);
    declare_parameter("zupt_speed_threshold", 0.02);
    declare_parameter("p_max_xy",    1.0);
    declare_parameter("p_max_theta", 2.0);
    declare_parameter("scan_match_noise_x",     0.04);
    declare_parameter("scan_match_noise_y",     0.04);
    declare_parameter("scan_match_noise_theta", 0.03);

    // ── Fase 2: gating por covarianza actual del EKF ─────────────────────────
    // Cuando trace(P_xy) = P[0]+P[4] es pequeña (ArUco activo, pose confiable),
    // se infla R_scan para reducir el peso del scan matching.
    // Cuando trace(P_xy) es grande (sin ArUco, deriva acumulada), R_scan es normal.
    //
    // R_scan_eff = R_scan_base * (1 + alpha * low_cov_thresh / (trace_P + eps))
    //   → trace_P << low_cov_thresh : R_scan_eff ≈ R_scan_base * (1 + alpha) → gating fuerte
    //   → trace_P >> high_cov_thresh: R_scan_eff ≈ R_scan_base               → sin gating
    //
    // Ajuste recomendado:
    //   low_cov_thresh:  0.05 m²  → trace_P < 0.05: ArUco activo y confiable
    //   high_cov_thresh: 0.30 m²  → trace_P > 0.30: pérdida de ArUco o drift acumulado
    //   gate_alpha:      5.0      → factor máximo de inflación de R cuando P es pequeña
    declare_parameter("scan_match_gate_alpha",       5.0);
    declare_parameter("scan_match_low_cov_thresh",   0.05);
    declare_parameter("scan_match_high_cov_thresh",  0.30);

    // ── Fase 4: Mahalanobis gate ─────────────────────────────────────────────
    // Rechaza correcciones de scan matching cuya innovación sea estadísticamente
    // incompatible con el estado actual del EKF.
    // d² = inn[0]²/S[0] + inn[1]²/S[4] + inn[2]²/S[8]
    // Si d > mahal_gate → corrección rechazada (outlier).
    // Valor por defecto 3.5 ≈ 3.5σ: conservador, rechaza saltos grandes sin
    // bloquear correcciones pequeñas válidas.
    declare_parameter("scan_match_mahal_gate", 3.5);
    // Umbral de trace(P_xy) para publicar estado "LOST" en /localization/status.
    // Por encima de este valor la localización se considera perdida.
    declare_parameter("loc_lost_thresh", 0.80);

    double ic = get_parameter("initial_covariance").as_double();
    x_  = {get_parameter("initial_x").as_double(),
           get_parameter("initial_y").as_double(),
           get_parameter("initial_theta").as_double()};
    P_  = {ic,0,0, 0,ic,0, 0,0,ic};
    init_from_aruco_ = get_parameter("init_from_aruco").as_bool();
    initialized_     = !init_from_aruco_;

    double qx  = get_parameter("process_noise_x").as_double();
    double qy  = get_parameter("process_noise_y").as_double();
    double qt  = get_parameter("process_noise_theta").as_double();
    Q_ = {qx,0,0, 0,qy,0, 0,0,qt};

    double rx  = get_parameter("meas_noise_x").as_double();
    double ry  = get_parameter("meas_noise_y").as_double();
    double rt  = get_parameter("meas_noise_theta").as_double();
    R_ = {rx,0,0, 0,ry,0, 0,0,rt};

    double smx = get_parameter("scan_match_noise_x").as_double();
    double smy = get_parameter("scan_match_noise_y").as_double();
    double smt = get_parameter("scan_match_noise_theta").as_double();
    R_scan_ = {smx,0,0, 0,smy,0, 0,0,smt};

    zupt_threshold_      = get_parameter("zupt_speed_threshold").as_double();
    p_max_xy_            = get_parameter("p_max_xy").as_double();
    p_max_theta_         = get_parameter("p_max_theta").as_double();
    sm_gate_alpha_       = get_parameter("scan_match_gate_alpha").as_double();
    sm_low_cov_thresh_   = get_parameter("scan_match_low_cov_thresh").as_double();
    sm_high_cov_thresh_  = get_parameter("scan_match_high_cov_thresh").as_double();
    sm_mahal_gate_       = get_parameter("scan_match_mahal_gate").as_double();
    loc_lost_thresh_     = get_parameter("loc_lost_thresh").as_double();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom_raw", 10,
      std::bind(&KalmanFilterNode::odom_cb, this, std::placeholders::_1));

    sub_aruco_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/aruco/pose", 10,
      std::bind(&KalmanFilterNode::aruco_cb, this, std::placeholders::_1));

    sub_scan_match_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/scan_match/pose", 10,
      std::bind(&KalmanFilterNode::scan_match_cb, this, std::placeholders::_1));

    pub_odom_   = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    pub_loc_status_ = create_publisher<std_msgs::msg::String>("/localization/status", 10);
    pub_loc_cov_    = create_publisher<std_msgs::msg::Float32>("/localization/covariance", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Publica estado inicial antes de que llegue el primer ArUco
    status_timer_ = create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&KalmanFilterNode::publish_status, this));

    last_time_ = now();
    if (init_from_aruco_) {
      RCLCPP_INFO(get_logger(),
        "kalman_filter_node iniciando — esperando primer ArUco para establecer pose...\n"
        "  scan_match gating: alpha=%.1f low_cov=%.3f high_cov=%.3f mahal_gate=%.1f",
        sm_gate_alpha_, sm_low_cov_thresh_, sm_high_cov_thresh_, sm_mahal_gate_);
    } else {
      RCLCPP_INFO(get_logger(),
        "kalman_filter_node started — pose inicial: x=%.3f y=%.3f theta=%.3f rad",
        x_[0], x_[1], x_[2]);
    }
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    auto t = rclcpp::Time(msg->header.stamp);
    if (!initialized_) {
      last_time_ = t;
      return;
    }

    double dt = (t - last_time_).seconds();
    last_time_ = t;
    if (dt <= 0.0 || dt > 1.0) { publish(); return; }

    double v   = msg->twist.twist.linear.x;
    double w   = msg->twist.twist.angular.z;
    double th  = x_[2];

    double delta_d  = v * dt;
    double delta_th = w * dt;

    x_[0] += delta_d * std::cos(th);
    x_[1] += delta_d * std::sin(th);
    x_[2]  = norm_angle(x_[2] + delta_th);

    Mat3 F = {1,0,-delta_d*std::sin(th),
              0,1, delta_d*std::cos(th),
              0,0,1};

    double speed    = std::abs(v) + std::abs(w);
    double q_scale  = std::min(speed / std::max(zupt_threshold_, 1e-6), 1.0);

    Mat3 Q_dt{};
    for (int i = 0; i < 9; ++i) Q_dt[i] = Q_[i] * dt * q_scale;
    P_ = mat_add(mat_mul(mat_mul(F, P_), mat_transpose(F)), Q_dt);

    P_[0] = std::min(P_[0], p_max_xy_);
    P_[4] = std::min(P_[4], p_max_xy_);
    P_[8] = std::min(P_[8], p_max_theta_);

    publish();
  }

  void aruco_cb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    const auto & p = msg->pose.pose;
    tf2::Quaternion q(p.orientation.x, p.orientation.y,
                      p.orientation.z, p.orientation.w);
    double z[3] = {p.position.x, p.position.y, tf2::getYaw(q)};

    const auto & cov = msg->pose.covariance;
    if (cov[0] <= 1e-9 || cov[7] <= 1e-9) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "ArUco: covarianza degenerada — ignorando");
      return;
    }

    if (!initialized_) {
      x_[0] = z[0];
      x_[1] = z[1];
      x_[2] = z[2];
      double p_theta0 = std::max(cov[35] > 1e-9 ? cov[35] : R_[8], R_[8]);
      P_ = {std::max(cov[0], R_[0]),  0.0,      0.0,
            0.0,     std::max(cov[7], R_[4]),    0.0,
            0.0,     0.0,                         p_theta0};
      initialized_ = true;
      last_time_   = now();
      RCLCPP_INFO(get_logger(),
        "✅ Pose inicial desde ArUco: x=%.3f y=%.3f theta=%.3f rad  "
        "std=(%.3f, %.3f) m",
        x_[0], x_[1], x_[2], std::sqrt(cov[0]), std::sqrt(cov[7]));
      publish();
      return;
    }

    double r_x     = std::max(cov[0],                          R_[0]);
    double r_y     = std::max(cov[7],                          R_[4]);
    double r_theta = std::max(cov[35] > 1e-9 ? cov[35] : R_[8], R_[8]);
    Mat3 R = {r_x, 0.0, 0.0,  0.0, r_y, 0.0,  0.0, 0.0, r_theta};

    Mat3 S = mat_add(P_, R);
    Mat3 K = mat_mul(P_, mat_inv3(S));

    double inn[3] = {z[0]-x_[0], z[1]-x_[1], norm_angle(z[2]-x_[2])};
    x_[0] += K[0]*inn[0] + K[1]*inn[1] + K[2]*inn[2];
    x_[1] += K[3]*inn[0] + K[4]*inn[1] + K[5]*inn[2];
    x_[2]  = norm_angle(x_[2] + K[6]*inn[0] + K[7]*inn[1] + K[8]*inn[2]);

    Mat3 IK = {1-K[0],-K[1],-K[2], -K[3],1-K[4],-K[5], -K[6],-K[7],1-K[8]};
    P_ = mat_mul(IK, P_);

    publish();
  }

  void scan_match_cb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    if (!initialized_) return;

    const auto & p = msg->pose.pose;
    tf2::Quaternion q(p.orientation.x, p.orientation.y,
                      p.orientation.z, p.orientation.w);
    double z[3] = {p.position.x, p.position.y, tf2::getYaw(q)};

    const auto & cov = msg->pose.covariance;

    // ── Fase 2: R adaptativo según trace(P_xy) ───────────────────────────────
    // Cuando P es pequeña (ArUco activo), se infla R_scan para reducir la
    // autoridad del scan matching. La inflación desaparece suavemente cuando
    // P crece (ArUco perdido, drift acumulado).
    //
    // trace_P = P[0] + P[4]  (incertidumbre en x + incertidumbre en y)
    // factor  = 1 + alpha * max(0, low_thresh - trace_P) / (trace_P + eps)
    //
    // Con trace_P = 0.01 (ArUco activo):  factor ≈ 1 + 5*(0.05/0.01) = 26× → gating fuerte
    // Con trace_P = 0.05 (low_thresh):    factor = 1 + 5*0 = 1.0              → sin gating
    // Con trace_P = 0.30 (high_thresh):   factor = 1.0                         → sin gating
    double trace_P = P_[0] + P_[4];
    double cov_gate_factor = 1.0;
    if (trace_P < sm_low_cov_thresh_) {
      double deficit = sm_low_cov_thresh_ - trace_P;
      cov_gate_factor = 1.0 + sm_gate_alpha_ * deficit / (trace_P + 1e-9);
    }
    // Factor clampeado para no inflar R en exceso (máximo 50×)
    cov_gate_factor = std::min(cov_gate_factor, 50.0);

    double r_x     = std::max(cov[0]  > 1e-9 ? cov[0]  : R_scan_[0], R_scan_[0]) * cov_gate_factor;
    double r_y     = std::max(cov[7]  > 1e-9 ? cov[7]  : R_scan_[4], R_scan_[4]) * cov_gate_factor;
    double r_theta = std::max(cov[35] > 1e-9 ? cov[35] : R_scan_[8], R_scan_[8]) * cov_gate_factor;
    Mat3 R = {r_x, 0.0, 0.0,  0.0, r_y, 0.0,  0.0, 0.0, r_theta};

    // ── Fase 4: Mahalanobis gate ─────────────────────────────────────────────
    // Rechaza correcciones cuya innovación sea estadísticamente incompatible
    // con el estado actual. S = P + R, d² = sum(inn[i]² / S[i,i]).
    // Si d > mahal_gate → outlier, descartado.
    Mat3 S = mat_add(P_, R);
    double inn[3] = {z[0]-x_[0], z[1]-x_[1], norm_angle(z[2]-x_[2])};

    double mahal2 = 0.0;
    if (S[0] > 1e-12) mahal2 += inn[0]*inn[0] / S[0];
    if (S[4] > 1e-12) mahal2 += inn[1]*inn[1] / S[4];
    if (S[8] > 1e-12) mahal2 += inn[2]*inn[2] / S[8];
    double mahal = std::sqrt(mahal2);

    if (mahal > sm_mahal_gate_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Scan match RECHAZADO — Mahalanobis=%.2f > %.1f  "
        "(inn=[%.3f, %.3f, %.3f]°  trace_P=%.4f  cov_factor=%.1f)",
        mahal, sm_mahal_gate_,
        inn[0], inn[1], inn[2]*180.0/M_PI,
        trace_P, cov_gate_factor);
      return;
    }

    // ── Fusión EKF ───────────────────────────────────────────────────────────
    Mat3 K = mat_mul(P_, mat_inv3(S));

    x_[0] += K[0]*inn[0] + K[1]*inn[1] + K[2]*inn[2];
    x_[1] += K[3]*inn[0] + K[4]*inn[1] + K[5]*inn[2];
    x_[2]  = norm_angle(x_[2] + K[6]*inn[0] + K[7]*inn[1] + K[8]*inn[2]);

    Mat3 IK = {1-K[0],-K[1],-K[2], -K[3],1-K[4],-K[5], -K[6],-K[7],1-K[8]};
    P_ = mat_mul(IK, P_);

    RCLCPP_DEBUG(get_logger(),
      "Scan match aceptado — Mahalanobis=%.2f  trace_P=%.4f  cov_factor=%.1f",
      mahal, trace_P, cov_gate_factor);

    publish();
  }

  void publish_status()
  {
    double trace_p = P_[0] + P_[4];

    std_msgs::msg::String status_msg;
    if (!initialized_) {
      status_msg.data = "INITIALIZING";
    } else if (trace_p > loc_lost_thresh_) {
      status_msg.data = "LOST";
    } else {
      status_msg.data = "OK";
    }
    pub_loc_status_->publish(status_msg);

    std_msgs::msg::Float32 cov_msg;
    cov_msg.data = static_cast<float>(trace_p);
    pub_loc_cov_->publish(cov_msg);
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
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_aruco_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_scan_match_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_loc_status_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_loc_cov_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::array<double, 3> x_;
  Mat3 P_, Q_, R_, R_scan_;
  rclcpp::Time last_time_;
  bool init_from_aruco_;
  bool initialized_;
  double zupt_threshold_;
  double p_max_xy_;
  double p_max_theta_;
  // Fase 2 + 4
  double sm_gate_alpha_;
  double sm_low_cov_thresh_;
  double sm_high_cov_thresh_;
  double sm_mahal_gate_;
  double loc_lost_thresh_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}
