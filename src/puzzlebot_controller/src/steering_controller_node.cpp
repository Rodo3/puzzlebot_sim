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
 * steering_controller_node — Controlador de seguimiento de trayectoria (Pure Pursuit).
 *
 * POSICIÓN EN EL PIPELINE:
 *   path_planner_node → /planned_path ─┐
 *   kalman_filter_node → /odom ────────┤→ [ESTE NODO] → /cmd_vel_in → obstacle_avoidance_node
 *
 * FUNCIÓN:
 *   Recorre la ruta en /planned_path usando el algoritmo Pure Pursuit:
 *   1. Encuentra el punto de la ruta que está a "lookahead_distance" metros adelante.
 *   2. Calcula la curvatura necesaria para alcanzarlo (κ = 2y / L²).
 *   3. Publica velocidad lineal constante y velocidad angular proporcional a la curvatura.
 *   4. Se detiene cuando la distancia al último punto < goal_tolerance.
 *
 *   Pure Pursuit da trayectorias suaves incluso con rutas en escalera (A* en grid).
 *   Si el ángulo de heading error es > 45°, reduce la velocidad al 50% para girar antes.
 *
 * TOPICS SUSCRITOS:
 *   /odom          (nav_msgs/Odometry) — pose filtrada del robot (desde kalman_filter_node)
 *   /planned_path  (nav_msgs/Path)     — ruta calculada por path_planner_node
 *
 * TOPICS PUBLICADOS:
 *   /cmd_vel_in    (geometry_msgs/Twist) — velocidad hacia obstacle_avoidance_node
 *
 * PARÁMETROS:
 *   lookahead_distance  [0.30 m]   — distancia del punto de mira
 *   max_linear_vel      [0.30 m/s] — velocidad máxima hacia adelante
 *   max_angular_vel     [1.50 rad/s] — velocidad angular máxima
 *   goal_tolerance      [0.10 m]   — radio de aceptación del goal
 *   control_frequency   [20.0 Hz]  — frecuencia del loop de control
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
    path_         = msg->poses;
    goal_reached_ = false;
    // Path con waypoints → habrá control activo: permite frenar de nuevo cuando
    // este path se agote. Path vacío → no rearmar (deja el tópico cedido).
    if (!path_.empty()) {
      idle_braked_ = false;
    }

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

  void control_loop()
  {
    geometry_msgs::msg::Twist cmd;

    if (!have_pose_ || path_.empty() || goal_reached_) {
      // Sin path activo: publica UN cero para frenar y luego cede el tópico.
      if (!idle_braked_) {
        pub_cmd_->publish(cmd);  // zero velocity (frenado único)
        idle_braked_ = true;
      }
      return;
    }

    // Check if final goal reached
    const auto & goal = path_.back().pose.position;
    double dist_to_goal = std::hypot(goal.x - robot_x_, goal.y - robot_y_);
    if (dist_to_goal < goal_tol_) {
      goal_reached_ = true;
      idle_braked_  = true;  // ya frenamos aquí; no spamear ceros en idle
      RCLCPP_INFO(get_logger(), "Goal reached (dist=%.3f m)", dist_to_goal);
      pub_cmd_->publish(cmd);
      return;
    }

    // Hay control activo → rearmar el frenado de idle para el próximo periodo sin path
    idle_braked_ = false;

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
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr  pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<geometry_msgs::msg::PoseStamped> path_;
  int    path_idx_{0};
  bool   goal_reached_{false};
  bool   have_pose_{false};
  // Cuando no hay path activo publicamos UN solo cero (para frenar) y luego
  // callamos, cediendo /cmd_vel_in a otros publicadores (p.ej. el barrido del
  // mission_manager_node). Si siguiéramos publicando ceros en cada tick,
  // pisaríamos esos comandos. Se rearma al recibir un path nuevo.
  bool   idle_braked_{false};

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
