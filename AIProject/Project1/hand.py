import cv2
import numpy as np
from collections import deque
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTrackerTasksAPI:
    def __init__(self, max_trace_length=100, model_path="hand_landmarker.task"):
        """
        初始化手部追踪器（使用Tasks API）
        
        Args:
            max_trace_length: 最大轨迹点数量
            model_path: 手部检测模型路径（.task文件）
        """
        # 轨迹点列表
        self.trace_points = deque(maxlen=max_trace_length)
        self.max_trace_length = max_trace_length
        
        # 追踪点类型：'palm'(手腕) 或 'index_finger'(食指)
        self.track_point = 'index_finger'  # 默认追踪食指，效果更好
        
        # 平滑窗口大小
        self.smoothing_window = 5
        
        # 历史中心位置（用于速度计算）
        self.prev_center = None
        
        # 初始化Tasks API的手部检测器
        self.detector = self._init_detector(model_path)
        
        # 手部关键点索引定义
        self.LANDMARKS = {
            'wrist': 0,           # 手腕
            'thumb_tip': 4,       # 拇指尖
            'index_finger_tip': 8,   # 食指尖
            'middle_finger_tip': 12, # 中指尖
            'ring_finger_tip': 16,   # 无名指尖
            'pinky_tip': 20,      # 小指尖
        }
        
        # FPS计算相关
        self.frame_count = 0
        self.start_time = cv2.getTickCount()
        self.fps = 0
    def _init_detector(self, model_path):
        """初始化MediaPipe HandLandmarker"""
        # 配置检测器选项
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # 视频模式（需要时间戳）
            num_hands=1,  # 只追踪一只手
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return vision.HandLandmarker.create_from_options(options)
    
    def detect_hand(self, frame, timestamp_ms):
        """
        检测手部并返回追踪点坐标
        
        Args:
            frame: OpenCV图像帧(BGR格式)
            timestamp_ms: 时间戳（毫秒）
            
        Returns:
            hand_center: 追踪点坐标(x, y)或None
            hand_landmarks: 所有手部关键点坐标列表
            handedness: 左右手标签
        """
        # 将BGR格式转换为RGB（MediaPipe需要RGB格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建MediaPipe Image对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 执行检测（VIDEO模式需要时间戳）
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        hand_center = None
        hand_landmarks = None
        handedness = None
        
        # 检查是否检测到手部
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            # 获取第一只手的关节点
            landmarks = detection_result.hand_landmarks[0]
            hand_landmarks = landmarks
            
            # 获取左右手信息
            if detection_result.handedness and len(detection_result.handedness) > 0:
                handedness = detection_result.handedness[0][0].display_name
            
            h, w, _ = frame.shape
            
            # 根据追踪点类型选择坐标
            if self.track_point == 'palm':
                # 使用手腕作为追踪点
                wrist = landmarks[self.LANDMARKS['wrist']]
                hand_center = (int(wrist.x * w), int(wrist.y * h))
            else:  # index_finger
                # 使用食指指尖
                index_tip = landmarks[self.LANDMARKS['index_finger_tip']]
                hand_center = (int(index_tip.x * w), int(index_tip.y * h))
        
        return hand_center, hand_landmarks, handedness
    
    def draw_hand_skeleton(self, frame, hand_landmarks):
        """在帧上绘制手部骨架（连线）"""
        if hand_landmarks is None:
            return
        
        h, w, _ = frame.shape
        
        # 手部关键点连接关系（MediaPipe标准连接）
        connections = [
            # 拇指
            (1, 2), (2, 3), (3, 4),
            # 食指
            (5, 6), (6, 7), (7, 8),
            # 中指
            (9, 10), (10, 11), (11, 12),
            # 无名指
            (13, 14), (14, 15), (15, 16),
            # 小指
            (17, 18), (18, 19), (19, 20),
            # 手掌
            (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        
        # 获取所有关键点的像素坐标
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        
        # 绘制连接线
        for connection in connections:
            if connection[0] < len(points) and connection[1] < len(points):
                cv2.line(frame, points[connection[0]], points[connection[1]], (0, 255, 0), 2)
        
        # 绘制关键点
        for i, point in enumerate(points):
            # 特殊标记追踪点（食指尖）
            if i == self.LANDMARKS['index_finger_tip']:
                cv2.circle(frame, point, 8, (0, 0, 255), -1)
            else:
                cv2.circle(frame, point, 4, (0, 255, 255), -1)
    
    def smooth_trace(self, points):
        """对轨迹点进行平滑处理"""
        if len(points) < self.smoothing_window:
            return list(points)
        
        smoothed = []
        for i in range(len(points)):
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(points), i + self.smoothing_window // 2 + 1)
            window = list(points)[start:end]
            avg_x = int(np.mean([p[0] for p in window]))
            avg_y = int(np.mean([p[1] for p in window]))
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def draw_trace(self, frame, points):
        """在帧上绘制轨迹"""
        if len(points) < 2:
            return
        
        # 平滑轨迹点
        smoothed_points = self.smooth_trace(points)
        
        # 绘制轨迹线（颜色渐变）
        for i in range(1, len(smoothed_points)):
            if smoothed_points[i-1] and smoothed_points[i]:
                # 根据点在轨迹中的位置设置颜色（蓝色→绿色→红色）
                t = i / len(smoothed_points)
                color = (
                    int(255 * t),      # B: 逐渐增加
                    int(255 * (1 - abs(0.5 - t) * 2)),  # G: 中间亮
                    int(255 * (1 - t)) # R: 逐渐减少
                )
                thickness = 4
                cv2.line(frame, smoothed_points[i-1], smoothed_points[i], color, thickness)
        
        # 绘制轨迹点
        for point in smoothed_points:
            cv2.circle(frame, point, 5, (255, 255, 0), -1)
    
    def draw_info(self, frame, hand_center, handedness, fps, hand_detected):
        """在帧上显示信息"""
        if hand_detected and hand_center:
            # 在追踪点绘制大红圆点
            cv2.circle(frame, hand_center, 12, (0, 0, 255), -1)
            cv2.circle(frame, hand_center, 16, (0, 0, 255), 2)
            
            # 显示追踪信息
            tracking_text = f"Tracking: {self.track_point.upper()}"
            cv2.putText(frame, tracking_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if handedness:
                cv2.putText(frame, f"Hand: {handedness}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_text = f"Position: ({hand_center[0]}, {hand_center[1]})"
            cv2.putText(frame, info_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示轨迹点数量
        cv2.putText(frame, f"Trace Length: {len(self.trace_points)}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 操作提示
        cv2.putText(frame, "Press 'c' to clear trace | 't' to toggle tracking point", 
                   (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit , 's' to save ", 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_velocity(self, current_center):
        """计算手部移动速度（像素/帧）"""
        if self.prev_center is None:
            self.prev_center = current_center
            return 0
        
        distance = math.sqrt((current_center[0] - self.prev_center[0])**2 + 
                           (current_center[1] - self.prev_center[1])**2)
        self.prev_center = current_center
        return distance
    
    def draw_velocity(self, frame, velocity):
        """在帧上显示速度信息"""
        if velocity > 0:
            cv2.putText(frame, f"Speed: {velocity:.1f} px/frame", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = cv2.getTickCount()
            self.fps = 30 / ((end_time - self.start_time) / cv2.getTickFrequency())
            self.frame_count = 0
            self.start_time = end_time
        return self.fps


def download_model():
    """下载手部检测模型文件"""
    import urllib.request
    import os
    
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_path = "hand_landmarker.task"
    
    if not os.path.exists(model_path):
        print("正在下载手部检测模型（约10MB）...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("模型下载完成！")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型文件：")
            print(model_url)
            return None
    return model_path


def main():
    """主函数"""
    # 下载模型（如果不存在）
    model_path = download_model()
    if model_path is None:
        print("无法获取模型文件，程序退出")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 创建手部追踪器
    tracker = HandTrackerTasksAPI(max_trace_length=200, model_path=model_path)
    
    # 获取摄像头FPS（用于时间戳计算）
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps <= 0:
        camera_fps = 30.0
    
    print("操作说明：")
    print("  'c' - 清除轨迹")
    print("  't' - 切换追踪点（食指/手腕）")
    print("  'q' 或 ESC - 退出程序")
    print("   按s键保存当前画面    ")
    print("=" * 60)
    
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法获取摄像头画面")
            break
        
        # 镜像翻转，使移动更自然
        frame = cv2.flip(frame, 1)
        
        # 计算时间戳（毫秒）
        timestamp_ms = int(frame_index * (1000 / camera_fps))
        frame_index += 1
        
        # 检测手部
        hand_center, hand_landmarks, handedness = tracker.detect_hand(frame, timestamp_ms)
        
        # 绘制手部骨架
        tracker.draw_hand_skeleton(frame, hand_landmarks)
        
        # 更新轨迹
        hand_detected = hand_center is not None
        if hand_detected:
            tracker.trace_points.append(hand_center)
            
            # 计算速度
            velocity = tracker.calculate_velocity(hand_center)
            tracker.draw_velocity(frame, velocity)
        else:
            tracker.prev_center = None
        
        # 绘制轨迹
        if len(tracker.trace_points) > 0:
            tracker.draw_trace(frame, list(tracker.trace_points))
        
        # 计算并更新FPS
        fps = tracker.update_fps()
        
        # 绘制检测信息
        tracker.draw_info(frame, hand_center, handedness, fps, hand_detected)
        
        # 显示画面
        cv2.imshow('Hand Tracking - MediaPipe Tasks API', frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            break
        elif key == ord('c'):  # 清除轨迹
            tracker.trace_points.clear()
            print("✓ 轨迹已清除")
        elif key == ord('t'):  # 切换追踪点
            tracker.track_point = 'palm' if tracker.track_point == 'index_finger' else 'index_finger'
            print(f"✓ 追踪点切换为: {tracker.track_point}")
        elif key==ord('s') or key==29:
            cv2.imwrite("handdetect_out.png",frame)
            print("保存成功")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Windows上需要处理销毁事件，否则窗口会卡死
    tracker.detector.close()
    print("\n程序已退出")


if __name__ == "__main__":
    main()