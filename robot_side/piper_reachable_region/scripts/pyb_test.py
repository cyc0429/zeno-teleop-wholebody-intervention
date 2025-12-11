import os
import time

import pybullet_planning as pp
import pybullet as p
import rospkg
from pipper import Piper
from torch_kin_sampler import TorchKinMapper

rospack = rospkg.RosPack()
piper_description_path = rospack.get_path("piper_description")
piper_moveit_config_path = rospack.get_path("piper_moveit_config")
robot_model_path = os.path.join(piper_description_path, "urdf", "piper_description.urdf")
semantics_path = os.path.join(piper_moveit_config_path, "config", "piper.srdf")

# p.configureDebugVisualizer(p.COV_ENABLE_GUI, True, physicsClientId=sever_id)

sampler = TorchKinMapper(N_fk=200000, num_loops=20, use_pybullet_limits=True, post_process=False)
# sampler.generate_maps()
# sampler.visualize_h5_pybullet(os.path.join(sampler.reach_map_dir, f"3D_{sampler.reach_map_file_name}.h5"))
sever_id = pp.connect(use_gui=True)
piper = Piper(robot_model_path, semantics_path=semantics_path)

sampler.visualize_h5_pybullet(os.path.join(sampler.reach_map_dir, f"3D_reach_map_gripper_base_0.05_2025-12-11-16-15-10.h5"))

# while True:
#     joint_pos = piper.sample_joint_positions()
#     piper.set_joint_positions(joint_pos)
#     collision = piper.check_collision(joint_pos)
#     if collision:
#         print("Collision detected")
#     pp.wait_for_user()