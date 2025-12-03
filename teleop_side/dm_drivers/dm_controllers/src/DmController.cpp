#include "dm_controllers/DmController.h"

namespace damiao
{

	bool DmController::init(HybridJointInterface *robot_hw, ros::NodeHandle &nh)
	{
		std::cerr << "Successfully got HybridEffort joint interface" << std::endl;

		std::vector<std::string> joint_names{"joint0_motor", "joint1_motor", "joint2_motor"};
		// std::vector<std::string> joint_names{"joint0_motor", "joint1_motor"};
		for (const auto &joint_name : joint_names)
		{
			hybridJointHandles_.push_back(robot_hw->getHandle(joint_name));
		}

		// Initialize repulsive force values
		repulsive_force_x_ = 0.0;
		repulsive_force_y_ = 0.0;

		// Subscribe to repulsive force vector topic
		nh_ = nh;
		repulsive_force_sub_ = nh_.subscribe("/repulsive_force_vector", 1, &DmController::repulsiveForceCallback, this);

		// Load control parameters from ROS parameter server
		nh_.param("base_kp", base_kp_, 5.0);
		nh_.param("repulsive_force_threshold", repulsive_force_threshold_, 5.0);
		nh_.param("pos_des_positive", pos_des_positive_, 0.1);
		nh_.param("pos_des_negative", pos_des_negative_, -0.1);
		nh_.param("kd_default", kd_default_, 0.1);
		nh_.param("kd_high", kd_high_, 2.0);
		nh_.param("kp_motor0", kp_motor0_, 20.0);
		nh_.param("kd_motor0", kd_motor0_, 0.1);

		ROS_INFO("DmController parameters loaded:");
		ROS_INFO("  base_kp: %.2f", base_kp_);
		ROS_INFO("  repulsive_force_threshold: %.2f", repulsive_force_threshold_);
		ROS_INFO("  pos_des_positive: %.2f", pos_des_positive_);
		ROS_INFO("  pos_des_negative: %.2f", pos_des_negative_);
		ROS_INFO("  kd_default: %.2f", kd_default_);
		ROS_INFO("  kd_high: %.2f", kd_high_);
		ROS_INFO("  kp_motor0: %.2f", kp_motor0_);
		ROS_INFO("  kd_motor0: %.2f", kd_motor0_);

		return true;
	}

	void DmController::starting(const ros::Time &time)
	{
		// ROS_INFO("DmController started.");
		std::cerr << "DmController started." << std::endl;
	}

	void DmController::update(const ros::Time &time, const ros::Duration &period)
	{
		// 设置关节命令
		// std::cerr<<"dmcontroller update"<<std::endl;
		// std::cerr<<"size: "<<hybridJointHandles_.size()<<std::endl;
		// float q = sin(std::chrono::system_clock::now().time_since_epoch().count() / 1e9);
		// hybridJointHandles_[0].setCommand(0.0, q*-3.0,0.0,0.3,0.0);
		// hybridJointHandles_[1].setCommand(0.0, q*2.0,0.0,0.3,0.0);
		// hybridJointHandles_[2].setCommand(0.0, q*-5.0,0.0,0.3,0.0);

		// Get repulsive force values (thread-safe)
		double repulsive_x, repulsive_y;
		{
			std::lock_guard<std::mutex> lock(repulsive_force_mutex_);
			repulsive_x = repulsive_force_x_;
			repulsive_y = repulsive_force_y_;
		}

		// Calculate kp for motor1 (index 1) and motor2 (index 2) with absolute values added
		double kp_motor1 = base_kp_ + std::abs(repulsive_x);
		double pos_des_x = 0.0;
		double kd_motor1 = kd_default_;
		if (repulsive_x < -repulsive_force_threshold_)
		{
			pos_des_x = pos_des_negative_;
			kd_motor1 = kd_high_;
		}
		else if (repulsive_x > repulsive_force_threshold_)
		{
			pos_des_x = pos_des_positive_;
			kd_motor1 = kd_high_;
		}

		double kp_motor2 = base_kp_ + std::abs(repulsive_y);
		double pos_des_y = 0.0;
		double kd_motor2 = kd_default_;
		if (repulsive_y < -repulsive_force_threshold_)
		{
			pos_des_y = pos_des_negative_;
			kd_motor2 = kd_high_;
		}
		else if (repulsive_y > repulsive_force_threshold_)
		{
			pos_des_y = pos_des_positive_;
			kd_motor2 = kd_high_;
		}

		hybridJointHandles_[0].setCommand(0.0, 0.0, kp_motor0_, kd_motor0_, 0.0);
		hybridJointHandles_[1].setCommand(pos_des_x, 0.0, kp_motor1, kd_motor1, 0.0);
		hybridJointHandles_[2].setCommand(pos_des_y, 0.0, kp_motor2, kd_motor2, 0.0);
	}

	void DmController::stopping(const ros::Time &time)
	{
		// ROS_INFO("DmController stopped.");
		std::cerr << "DmController stop." << std::endl;
	}

	void DmController::repulsiveForceCallback(const geometry_msgs::Vector3Stamped::ConstPtr &msg)
	{
		std::lock_guard<std::mutex> lock(repulsive_force_mutex_);
		repulsive_force_x_ = msg->vector.x;
		repulsive_force_y_ = msg->vector.y;
	}

} // namespace damiao

// 注册插件
PLUGINLIB_EXPORT_CLASS(damiao::DmController, controller_interface::ControllerBase);