#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标板位姿检测节点（camera→target）

订阅：
    /realsense_left/color/image_raw
    /realsense_left/color/camera_info

发布：
    /handeye/target_pose (geometry_msgs/PoseStamped)
    frame_id = 相机光学坐标系 (例如 realsense_left_color_optical_frame)

内部使用 OpenCV ArUco 检测，计算标定板相对相机的姿态 T_cam_target
"""

import sys
from typing import Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation


class TargetDetector:
    """ArUco 目标板检测器，发布目标板相对相机的位姿"""

    # 支持的 ArUco 字典类型
    ARUCO_DICT_MAP = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self) -> None:
        # 从 ROS 参数服务器获取配置
        self.image_topic = rospy.get_param("~image_topic", "/realsense_left/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense_left/color/camera_info")
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/handeye/target_pose")

        # ArUco 字典类型
        aruco_dict_name = rospy.get_param("~aruco_dict", "DICT_5X5_250")
        if aruco_dict_name not in self.ARUCO_DICT_MAP:
            rospy.logwarn("Unknown ArUco dictionary '%s', using DICT_5X5_250", aruco_dict_name)
            aruco_dict_name = "DICT_5X5_250"
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT_MAP[aruco_dict_name])

        # ArUco GridBoard 参数
        self.markers_x = rospy.get_param("~markers_x", 3)  # 横向 marker 数量
        self.markers_y = rospy.get_param("~markers_y", 4)  # 纵向 marker 数量
        self.marker_size = rospy.get_param("~marker_size", 0.0564)  # ArUco marker 边长 (米)
        self.marker_separation = rospy.get_param("~marker_separation", 0.0057)  # marker 间距 (米)

        # ArUco 检测器参数
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        # 创建 GridBoard，原点在左下角 marker 的左下角
        self.grid_board = cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_size,
            self.marker_separation,
            self.aruco_dict
        )

        # 是否显示可视化窗口
        self.show_visualization = rospy.get_param("~show_visualization", True)
        self.window_name = "Target Detection"

        # 相机内参
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_frame_id: str = ""

        self.bridge = CvBridge()
        self.last_image: Optional[np.ndarray] = None
        self.last_pose: Optional[PoseStamped] = None

        # 初始化可视化窗口
        if self.show_visualization:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 发布者
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)

        # 订阅者
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)

        rospy.loginfo("TargetDetector initialized")
        rospy.loginfo("  Image topic: %s", self.image_topic)
        rospy.loginfo("  Camera info topic: %s", self.camera_info_topic)
        rospy.loginfo("  Target pose topic: %s", self.target_pose_topic)
        rospy.loginfo("  ArUco dictionary: %s", aruco_dict_name)
        rospy.loginfo("  GridBoard: %dx%d markers", self.markers_x, self.markers_y)
        rospy.loginfo("  Marker size: %.4fm, separation: %.4fm", self.marker_size, self.marker_separation)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """接收相机内参"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.camera_frame_id = msg.header.frame_id
            rospy.loginfo("Received camera intrinsics from frame: %s", self.camera_frame_id)
            rospy.loginfo("Camera matrix:\n%s", self.camera_matrix)
            rospy.loginfo("Distortion coefficients: %s", self.dist_coeffs)

    def _image_callback(self, msg: Image) -> None:
        """接收图像并检测目标板"""
        if self.camera_matrix is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics...")
            return

        try:
            # 转换为 OpenCV 图像
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "CvBridge error: %s", str(e))
            return

        if image is None:
            rospy.logwarn_throttle(5.0, "Received empty image")
            return

        self.last_image = image.copy()

        # 检测目标板位姿
        pose_result = self._detect_aruco(image, msg.header.stamp)

        if pose_result is not None:
            self.last_pose = pose_result
            self.pose_pub.publish(pose_result)

    def _detect_aruco(self, image: np.ndarray, stamp: rospy.Time) -> Optional[PoseStamped]:
        """使用 ArUco GridBoard 检测位姿，原点在 board 左下角"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            rospy.logdebug_throttle(2.0, "No ArUco markers detected")
            self._draw_visualization(image, None, None)
            return None

        # 使用 GridBoard 估计整个板的位姿
        # 获取 board 上所有 marker 的 3D 点和对应的 2D 图像点
        obj_points, img_points = self.grid_board.matchImagePoints(corners, ids)

        if obj_points is None or len(obj_points) == 0:
            rospy.logdebug_throttle(
                2.0,
                "No board markers matched from detected markers: %s",
                ids.flatten().tolist(),
            )
            self._draw_visualization(image, corners, ids)
            return None

        # 使用 solvePnP 估计 board 位姿
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs
        )

        if not success:
            rospy.logwarn_throttle(2.0, "solvePnP failed for GridBoard")
            self._draw_visualization(image, corners, ids)
            return None

        # 转换为 PoseStamped
        pose_stamped = self._create_pose_stamped(rvec, tvec, stamp)

        # 可视化
        self._draw_visualization(image, corners, ids, rvec, tvec)

        tvec_flat = tvec.flatten()
        rospy.logdebug(
            "Detected GridBoard (%d markers), position: [%.3f, %.3f, %.3f]",
            len(ids),
            tvec_flat[0],
            tvec_flat[1],
            tvec_flat[2],
        )

        return pose_stamped

    def _create_pose_stamped(self, rvec: np.ndarray, tvec: np.ndarray, stamp: rospy.Time) -> PoseStamped:
        """将旋转向量和平移向量转换为 PoseStamped"""
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.camera_frame_id

        # 平移
        tvec_flat = tvec.flatten()
        pose.pose.position.x = tvec_flat[0]
        pose.pose.position.y = tvec_flat[1]
        pose.pose.position.z = tvec_flat[2]

        # 旋转向量转四元数
        rvec_flat = rvec.flatten()
        rotation = Rotation.from_rotvec(rvec_flat)
        quat = rotation.as_quat()  # [x, y, z, w]

        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose

    def _draw_visualization(
        self,
        image: np.ndarray,
        corners,
        ids,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
    ) -> None:
        """绘制 ArUco GridBoard 检测可视化"""
        if not self.show_visualization:
            return

        vis_image = image.copy()

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)

            if rvec is not None and tvec is not None:
                # 绘制坐标轴，原点在 board 左下角
                axis_length = self.marker_size * 1.5
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length)

        # 显示状态信息
        status = "Detected" if rvec is not None else "Searching..."
        color = (0, 255, 0) if rvec is not None else (0, 0, 255)
        cv2.putText(vis_image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if self.last_pose is not None:
            pos = self.last_pose.pose.position
            text = f"Pos: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]"
            cv2.putText(vis_image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.last_vis_image = vis_image

    def spin(self) -> None:
        """主循环"""
        rospy.loginfo("Target detector running. Press 'q' to exit visualization.")
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.show_visualization and hasattr(self, "last_vis_image"):
                cv2.imshow(self.window_name, self.last_vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    rospy.loginfo("Quit visualization requested")
                    break
            rate.sleep()

        if self.show_visualization:
            cv2.destroyAllWindows()


def main() -> int:
    rospy.init_node("detect_target", anonymous=False)

    try:
        detector = TargetDetector()
        detector.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error in target detector: %s", str(e))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
