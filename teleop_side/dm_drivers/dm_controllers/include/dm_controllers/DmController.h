#ifndef MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H
#define MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <dm_common/HybridJointInterface.h>
#include <mutex>

namespace damiao
{


class DmController : public controller_interface::Controller<HybridJointInterface>
{
public:
  DmController() = default;
  ~DmController() = default;

  bool init(HybridJointInterface* robot_hw, ros::NodeHandle& nh) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& time) override;

private:
  std::vector<HybridJointHandle> hybridJointHandles_;
  ros::Subscriber repulsive_force_sub_;
  ros::NodeHandle nh_;
  
  // Thread-safe storage for repulsive force values
  std::mutex repulsive_force_mutex_;
  double repulsive_force_x_;
  double repulsive_force_y_;
  
  // Control parameters (loaded from ROS parameters)
  double base_kp_;                    // Base kp value for motor1 and motor2
  double repulsive_force_threshold_;   // Threshold for repulsive force
  double pos_des_positive_;           // Positive position command when threshold exceeded
  double pos_des_negative_;            // Negative position command when threshold exceeded
  double kd_default_;                 // Default kd value
  double kd_high_;                    // High kd value when threshold exceeded
  double kp_motor0_;                  // kp value for motor0
  double kd_motor0_;                  // kd value for motor0
  
  void repulsiveForceCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg);

};

}  // namespace my_controller_plugin

#endif  // MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H