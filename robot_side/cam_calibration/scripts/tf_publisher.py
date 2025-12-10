#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布 gripper -> camera 的 TF。

逻辑：
1. 自动查找 samples 目录下最新的标定文件（文件名含时间戳 handeye_data_YYYYMMDD_HHMMSS.json）。
2. 读取其中的 R_gripper2cam / t_gripper2cam（若没有则回退到 calib_result 的位姿；兼容旧键）。
3. 计算 gripper->cam 变换并持续发布 TF。
"""

import os
import json
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import rospy
import rospkg
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import (
    quaternion_from_matrix,
    quaternion_matrix,
)


def _extract_timestamp(filename: str) -> datetime:
    """从文件名 handeye_data_YYYYMMDD_HHMMSS.json 提取时间戳，解析失败返回最小时间。"""
    try:
        if filename.startswith("handeye_data_") and filename.endswith(".json"):
            ts_str = filename[len("handeye_data_") : -len(".json")]
            if "_" in ts_str:
                date_part, time_part = ts_str.split("_", 1)
                if len(date_part) == 8 and len(time_part) == 6:
                    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        pass
    return datetime.min


class GripperCamTFPublisher:
    def __init__(self) -> None:
        # 参数
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("cam_calibration")
        self.samples_dir = rospy.get_param("~samples_dir", os.path.join(pkg_path, "data", "samples"))
        self.parent_frame = rospy.get_param("~parent_frame", "end_effector")
        self.child_frame = rospy.get_param("~child_frame", "handeye_camera")
        self.publish_rate = rospy.get_param("~publish_rate", 30.0)
        self.specific_file = rospy.get_param("~trajectory_file", "")

        os.makedirs(self.samples_dir, exist_ok=True)

        # 载入变换
        self.translation, self.quaternion = self._load_transform()
        self.br = tf2_ros.TransformBroadcaster()

    def _find_latest_file(self) -> Optional[str]:
        """根据文件名中的时间戳选择最新的 handeye_data_*.json。"""
        try:
            candidates = [
                f
                for f in os.listdir(self.samples_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(self.samples_dir, f))
            ]
            if not candidates:
                return None
            candidates.sort(key=_extract_timestamp, reverse=True)
            latest = candidates[0]
            return os.path.join(self.samples_dir, latest)
        except Exception as e:
            rospy.logerr(f"Failed to list samples in {self.samples_dir}: {e}")
            return None

    def _load_from_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """从 JSON 文件读取 gripper->cam 或 calib_result，返回 (R_gripper2cam, t_gripper2cam)。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        R_gripper2cam = None
        t_gripper2cam = None

        if "R_gripper2cam" in data and "t_gripper2cam" in data:
            R_gripper2cam = np.array(data["R_gripper2cam"], dtype=float)
            t_gripper2cam = np.array(data["t_gripper2cam"], dtype=float)
        elif "R_cam2gripper" in data and "t_cam2gripper" in data:
            # 兼容旧字段名，语义仍表示 gripper->camera
            rospy.logwarn_once("Using legacy keys R_cam2gripper/t_cam2gripper; please update saved files.")
            R_gripper2cam = np.array(data["R_cam2gripper"], dtype=float)
            t_gripper2cam = np.array(data["t_cam2gripper"], dtype=float)

        # 如果缺少矩阵，则回退到 calib_result
        if (R_gripper2cam is None or t_gripper2cam is None) and "calib_result" in data:
            pose = data["calib_result"]
            try:
                q = [
                    pose["orientation"]["x"],
                    pose["orientation"]["y"],
                    pose["orientation"]["z"],
                    pose["orientation"]["w"],
                ]
                T = quaternion_matrix(q)
                T[0, 3] = pose["position"]["x"]
                T[1, 3] = pose["position"]["y"]
                T[2, 3] = pose["position"]["z"]
                R_gripper2cam = T[:3, :3]
                t_gripper2cam = T[:3, 3]
            except Exception as e:
                rospy.logwarn(f"Failed to parse calib_result from {filepath}: {e}")

        if R_gripper2cam is None or t_gripper2cam is None:
            raise ValueError("No R_gripper2cam / t_gripper2cam / calib_result found in file")

        return R_gripper2cam, t_gripper2cam

    def _load_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载 gripper->cam 的平移和四元数。"""
        # 优先使用参数指定的文件
        filepath = self.specific_file
        if filepath and not os.path.isabs(filepath):
            filepath = os.path.join(self.samples_dir, filepath)

        if filepath and not os.path.exists(filepath):
            rospy.logwarn(f"Specified trajectory_file not found: {filepath}, fallback to latest.")
            filepath = ""

        if not filepath:
            filepath = self._find_latest_file()

        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("No valid calibration file found in samples directory.")

        rospy.loginfo(f"Using calibration file: {filepath}")
        R_gripper2cam, t_gripper2cam = self._load_from_file(filepath)

        # 构造 4x4 矩阵以生成四元数
        T = np.eye(4)
        T[:3, :3] = R_gripper2cam
        T[:3, 3] = t_gripper2cam
        quat = quaternion_from_matrix(T)

        return t_gripper2cam, quat

    def publish(self) -> None:
        rate = rospy.Rate(self.publish_rate)

        if self.translation is None or self.quaternion is None:
            rospy.logerr("No transform loaded, aborting TF publishing.")
            return

        while not rospy.is_shutdown():
            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.parent_frame
            msg.child_frame_id = self.child_frame
            msg.transform.translation.x = float(self.translation[0])
            msg.transform.translation.y = float(self.translation[1])
            msg.transform.translation.z = float(self.translation[2])
            msg.transform.rotation.x = float(self.quaternion[0])
            msg.transform.rotation.y = float(self.quaternion[1])
            msg.transform.rotation.z = float(self.quaternion[2])
            msg.transform.rotation.w = float(self.quaternion[3])
            self.br.sendTransform(msg)
            rate.sleep()


def main() -> int:
    rospy.init_node("handeye_tf_publisher")
    try:
        node = GripperCamTFPublisher()
        node.publish()
    except Exception as e:
        rospy.logerr(f"TF publisher failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
