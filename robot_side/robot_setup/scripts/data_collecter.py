#!/usr/bin/env python3
"""LeRobot Dataset Collector for Whole-body Intervention System.

This script records ROS topics at a fixed frequency (e.g., 30Hz) using an
asynchronous cache to avoid frame drops. It saves the data directly into
Hugging Face `datasets` format, which is fully compatible with LeRobot.

Controls:
  [S] : Start recording a new episode
  [E] : End the current episode
  [Q] : Quit and save the dataset to disk
"""

import sys
import select
import termios
import tty
import threading
import time
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import JointState, Image as RosImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# Hugging Face Datasets
from datasets import Dataset, Features, Sequence, Value, Image as HfImage

# ==========================================
# 核心配置区
# ==========================================
FPS = 30  # 录制频率 (匹配相机帧率)
OUTPUT_DIR = "data/lerobot_intervention_dataset"

# 你的 22 个 Topic 定义
TOPICS = {
    "robot_left_pose": "/robot/arm_left/end_pose",
    "robot_right_pose": "/robot/arm_right/end_pose",
    "robot_left_joint": "/robot/arm_left/joint_states_single",
    "robot_right_joint": "/robot/arm_right/joint_states_single",
    "robot_left_cmd": "/robot/arm_left/pos_cmd",
    "robot_right_cmd": "/robot/arm_right/pos_cmd",
    
    "teleop_left_pose": "/teleop/arm_left/end_pose",
    "teleop_right_pose": "/teleop/arm_right/end_pose",
    "teleop_left_joint": "/teleop/arm_left/joint_states_single",
    "teleop_right_joint": "/teleop/arm_right/joint_states_single",
    
    "cam_left_color": "/realsense_left/color/image_raw",
    "cam_left_depth": "/realsense_left/aligned_depth_to_color/image_raw",
    "cam_right_color": "/realsense_right/color/image_raw",
    "cam_right_depth": "/realsense_right/aligned_depth_to_color/image_raw",
    "cam_top_color": "/realsense_top/color/image_raw",
    "cam_top_depth": "/realsense_top/aligned_depth_to_color/image_raw",
    
    "intervention_flags": "/intervention/flags"
}

class RawTerminal:
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

class LeRobotDataCollector:
    def __init__(self):
        rospy.init_node("lerobot_data_collector", anonymous=True)
        self.bridge = CvBridge()
        
        # 异步缓存池
        self.cache_lock = threading.Lock()
        self.state_cache = {key: None for key in TOPICS.keys()}
        
        # 数据集容器
        self.dataset_dict = {
            "observation.images.left": [],
            "observation.images.right": [],
            "observation.images.top": [],
            "observation.state": [],      # 14D (Left 7 + Right 7)
            "action": [],                 # 14D Cmd
            "intervention_arm": [],       # Evo-RL Flag
            "intervention_base": [],      # Evo-RL Flag
            "episode_index": [],
            "frame_index": [],
            "timestamp": []
        }
        
        self.is_recording = False
        self.current_episode = 0
        self.current_frame = 0
        
        self._setup_subscribers()
        
    def _setup_subscribers(self):
        """批量注册回调函数"""
        def make_callback(key):
            return lambda msg: self._generic_callback(key, msg)
            
        for key, topic in TOPICS.items():
            if "joint" in key or "cmd" in key:
                rospy.Subscriber(topic, JointState, make_callback(key), queue_size=1, tcp_nodelay=True)
            elif "pose" in key:
                rospy.Subscriber(topic, PoseStamped, make_callback(key), queue_size=1, tcp_nodelay=True)
            elif "color" in key or "depth" in key:
                rospy.Subscriber(topic, RosImage, make_callback(key), queue_size=1, tcp_nodelay=True)
            elif "flags" in key:
                rospy.Subscriber(topic, Float32MultiArray, make_callback(key), queue_size=1, tcp_nodelay=True)

    def _generic_callback(self, key, msg):
        """将最新消息放入缓存池"""
        with self.cache_lock:
            self.state_cache[key] = msg

    def _extract_joint_state(self, msg: JointState):
        """提取 7 自由度 (6关节+1夹爪)"""
        if msg is None or len(msg.position) < 7:
            return [0.0] * 7
        return list(msg.position[:7])

    def _ros_img_to_pil(self, msg: RosImage, is_depth=False):
        """将 ROS 图像转换为 PIL Image 供 HuggingFace 存储"""
        if msg is None:
            # 返回空黑图以防崩溃 (尺寸假设为 640x480)
            if is_depth:
                return Image.fromarray(np.zeros((480, 640), dtype=np.uint16))
            else:
                return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        
        try:
            if is_depth:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return Image.fromarray(cv_img)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                return Image.fromarray(cv_img)
        except Exception as e:
            rospy.logwarn(f"Image Conversion Error: {e}")
            return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))

    def _record_step(self, event):
        """以固定频率 (30Hz) 触发的数据快照保存"""
        if not self.is_recording:
            return

        with self.cache_lock:
            # 为了保证时序一致性，瞬间对整个缓存池做一个浅拷贝
            snapshot = self.state_cache.copy()

        # 1. 提取相机图像 (此处仅保存 Color 以适配基础 LeRobot，Depth 可根据需要存入额外列)
        img_left = self._ros_img_to_pil(snapshot["cam_left_color"])
        img_right = self._ros_img_to_pil(snapshot["cam_right_color"])
        img_top = self._ros_img_to_pil(snapshot["cam_top_color"])

        # 2. 提取 14D State (左臂 7D + 右臂 7D)
        state_left = self._extract_joint_state(snapshot["robot_left_joint"])
        state_right = self._extract_joint_state(snapshot["robot_right_joint"])
        full_state = state_left + state_right

        # 3. 提取 14D Action (发给控制器的 Cmd)
        # 注意：在真实的 BC 训练中，Action 通常是下一时刻的期望位置，
        # 在这里我们记录人类的遥控指令或策略输出的 Cmd
        cmd_left = self._extract_joint_state(snapshot["robot_left_cmd"])
        cmd_right = self._extract_joint_state(snapshot["robot_right_cmd"])
        full_action = cmd_left + cmd_right

        # 4. 提取极其珍贵的 Intervention Flags
        flags_msg = snapshot["intervention_flags"]
        if flags_msg and len(flags_msg.data) >= 2:
            arm_flag = int(flags_msg.data[0])
            base_flag = int(flags_msg.data[1])
        else:
            arm_flag = 0
            base_flag = 0

        # 5. 压入数据集
        self.dataset_dict["observation.images.left"].append(img_left)
        self.dataset_dict["observation.images.right"].append(img_right)
        self.dataset_dict["observation.images.top"].append(img_top)
        self.dataset_dict["observation.state"].append(full_state)
        self.dataset_dict["action"].append(full_action)
        self.dataset_dict["intervention_arm"].append(arm_flag)
        self.dataset_dict["intervention_base"].append(base_flag)
        self.dataset_dict["episode_index"].append(self.current_episode)
        self.dataset_dict["frame_index"].append(self.current_frame)
        self.dataset_dict["timestamp"].append(time.time())

        self.current_frame += 1

    def save_dataset(self):
        """将内存中的数据打包并保存为 LeRobot 兼容的 HuggingFace Dataset"""
        if len(self.dataset_dict["episode_index"]) == 0:
            print("\n[Save] No data collected. Exiting.")
            return
            
        print(f"\n[Save] Processing {len(self.dataset_dict['episode_index'])} frames...")
        
        # 定义 HuggingFace 的特征格式 (这正是 LeRobot 读取数据时需要的格式)
        features = Features({
            "observation.images.left": HfImage(),
            "observation.images.right": HfImage(),
            "observation.images.top": HfImage(),
            "observation.state": Sequence(Value("float32"), length=14),
            "action": Sequence(Value("float32"), length=14),
            "intervention_arm": Value("int64"),
            "intervention_base": Value("int64"),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "timestamp": Value("float32"),
        })
        
        hf_dataset = Dataset.from_dict(self.dataset_dict, features=features)
        
        print(f"[Save] Saving dataset to {OUTPUT_DIR} ...")
        hf_dataset.save_to_disk(OUTPUT_DIR)
        print("[Save] Done! The dataset is ready for LeRobot.")

    def run(self):
        print("\n=============================================")
        print("  LeRobot 数据采集器 (30Hz 异步快照机制)  ")
        print("=============================================")
        print(" [S] : 开始录制新 Episode")
        print(" [E] : 结束当前 Episode")
        print(" [Q] : 退出并保存 Dataset")
        print("=============================================\n")

        # 启动定时快照定时器
        rospy.Timer(rospy.Duration(1.0 / FPS), self._record_step)

        with RawTerminal():
            while not rospy.is_shutdown():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                
                if self.is_recording:
                    sys.stdout.write(f"\r[Recording] Episode: {self.current_episode} | Frames: {self.current_frame}  ")
                    sys.stdout.flush()

                if not readable:
                    continue
                
                ch = sys.stdin.read(1).lower()
                
                if ch == 's':
                    if not self.is_recording:
                        self.is_recording = True
                        self.current_frame = 0
                        print(f"\n[Action] Episode {self.current_episode} Started!")
                elif ch == 'e':
                    if self.is_recording:
                        self.is_recording = False
                        print(f"\n[Action] Episode {self.current_episode} Ended. Total frames: {self.current_frame}")
                        self.current_episode += 1
                elif ch == 'q':
                    if self.is_recording:
                        print(f"\n[Action] Episode {self.current_episode} Ended.")
                    self.is_recording = False
                    self.save_dataset()
                    return

if __name__ == "__main__":
    collector = LeRobotDataCollector()
    collector.run()