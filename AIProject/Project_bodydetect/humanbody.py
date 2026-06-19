import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 人体骨架连接关系（MediaPipe Pose 33关键点）
SKELETON_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # 躯干
    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # 右臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
# 关键骨架连接，只绘制主要骨骼线条
MAIN_SKELETON = [
    (11, 12),  # 肩膀
    (11, 23), (12, 24), (23, 24),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 上臂
    (15, 17), (16, 18),  # 前臂 (手腕需单独连)
    (23, 25), (25, 27), (24, 26), (26, 28),  # 腿
    (27, 29), (28, 30),  # 小腿
    (29, 31), (30, 32),  # 脚
    (15, 19), (16, 20), (19, 21), (20, 22),  # 手指
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

def draw_pose_landmarks(image, detection_result):
    """在图像上绘制人体姿态关键点和骨架连线"""
    h, w = image.shape[:2]

    if not detection_result.pose_landmarks:
        return image

    for landmarks in detection_result.pose_landmarks:
        points = {}
        for idx, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            visibility = getattr(lm, 'visibility', 1.0)
            if visibility < 0.5:
                continue
            points[idx] = (x, y)
            # 绘制关键点
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # 绘制骨架连线
        for start_idx, end_idx in MAIN_SKELETON:
            if start_idx in points and end_idx in points:
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)

        # 标注关键点编号
        for idx, (x, y) in points.items():
            cv2.putText(image, str(idx), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return image


def main():
    image_path = os.path.join(os.path.dirname(__file__), "test.jpg")
    output_path = os.path.join(os.path.dirname(__file__), "output.jpg")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 创建 PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # 转换为 RGB（MediaPipe 需要 RGB 输入）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # 执行人体姿态检测
    detection_result = detector.detect(mp_image)

    # 绘制骨架
    output_image = draw_pose_landmarks(image, detection_result)

    # 保存结果
    cv2.imwrite(output_path, output_image)
    print(f"检测完成，结果已保存至 {output_path}")

    # 显示结果
    cv2.imshow("Human Body Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.close()

if __name__ == "__main__":
    main()
