import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 平滑参数
SMOOTH_WINDOW = 5
history = {
    "x": deque(maxlen=SMOOTH_WINDOW),
    "y": deque(maxlen=SMOOTH_WINDOW),
    "z": deque(maxlen=SMOOTH_WINDOW),
    "yaw": deque(maxlen=SMOOTH_WINDOW),
    "pitch": deque(maxlen=SMOOTH_WINDOW),
    "roll": deque(maxlen=SMOOTH_WINDOW),
}


def smooth(values):
    if not values:
        return 0
    return sum(values) / len(values)


def draw_axes(img, center, rotation_vec, translation_vec, camera_matrix, dist_coeffs, length=80):
    """在图像上绘制 3D 坐标轴"""
    axis = np.float32([
        [length, 0, 0],
        [0, -length, 0],
        [0, 0, -length],
    ]).reshape(-1, 3)

    img_points, _ = cv2.projectPoints(axis, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2).astype(int)
    center = tuple(center.astype(int))

    cv2.line(img, center, tuple(img_points[0]), (0, 0, 255), 2)   # X红
    cv2.line(img, center, tuple(img_points[1]), (0, 255, 0), 2)   # Y绿
    cv2.line(img, center, tuple(img_points[2]), (255, 0, 0), 2)   # Z蓝


def main():
    # 摄像头内参（近似值，根据实际摄像头可标定）
    frame_w, frame_h = 640, 480
    focal = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D 面部关键点模型（MediaPipe Face Landmarker 的部分关键点索引）
    model_points = np.array([
        (0.0, 0.0, 0.0),          # 鼻尖 (1)
        (0.0, -63.6, -31.3),      # 下巴 (199)
        (-43.3, 34.0, -31.1),     # 左眼外角 (33)
        (43.3, 34.0, -31.1),      # 右眼外角 (263)
        (-28.5, -30.4, -13.0),    # 左嘴角 (61)
        (28.5, -30.4, -13.0),     # 右嘴角 (291)
    ], dtype=np.float64)

    landmark_indices = [1, 199, 33, 263, 61, 291]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    # 创建 FaceLandmarker
    base_options = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    frame_timestamp_ms = 0
    print("摄像头头部追踪已启动 | 按 Q 退出,按 S 保存 ")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33  # ~30fps

        h, w = frame.shape[:2]
        display = np.zeros((h + 120, w, 3), dtype=np.uint8)
        display[:h, :w] = frame

        if result.face_landmarks:
            for face_idx, face_landmarks in enumerate(result.face_landmarks):
                # 计算人脸边界框
                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]
                x_min = int(min(xs) * w)
                y_min = int(min(ys) * h)
                x_max = int(max(xs) * w)
                y_max = int(max(ys) * h)

                # 画方框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # 画标签
                label = f"Face {face_idx + 1}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x_min, y_min - label_size[1] - 10),
                              (x_min + label_size[0] + 10, y_min), (0, 255, 0), -1)
                cv2.putText(frame, label, (x_min + 5, y_min - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # 提取 2D 关键点
                img_points = []
                for idx in landmark_indices:
                    lm = face_landmarks[idx]
                    img_points.append([lm.x * w, lm.y * h])
                img_points = np.array(img_points, dtype=np.float64)

                # 用 solvePnP 计算头部姿态
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    model_points, img_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                if not success:
                    continue

                # 将旋转向量转为欧拉角
                rot_mat, _ = cv2.Rodrigues(rotation_vec)
                proj_mat = np.hstack((rot_mat, translation_vec))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

                pitch, yaw, roll = euler.flatten()

                # 获取平移量
                tx, ty, tz = translation_vec.flatten()

                # 平滑
                smooth_x = smooth(history["x"]); history["x"].append(tx)
                smooth_y = smooth(history["y"]); history["y"].append(ty)
                smooth_z = smooth(history["z"]); history["z"].append(tz)
                smooth_yaw = smooth(history["yaw"]); history["yaw"].append(yaw)
                smooth_pitch = smooth(history["pitch"]); history["pitch"].append(pitch)
                smooth_roll = smooth(history["roll"]); history["roll"].append(roll)

                # 鼻子位置
                nose = face_landmarks[1]
                nose_pt = np.array([nose.x * w, nose.y * h])

                # 画坐标轴
                draw_axes(frame, nose_pt, rotation_vec, translation_vec,
                          camera_matrix, dist_coeffs, length=60)

                # 画关键点
                for pt in img_points:
                    cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 255), -1)

                # 画鼻子高亮
                cv2.circle(frame, tuple(nose_pt.astype(int)), 5, (0, 255, 0), -1)

                # ----- 信息面板 -----
                display[:h, :w] = frame
                panel = display[h:, :]
                panel[:] = (30, 30, 30)

                info_lines = [
                    f"Position | X: {smooth_x:+7.1f}mm  Y: {smooth_y:+7.1f}mm  Z: {smooth_z:+7.1f}mm",
                    f"Rotation  | Yaw: {smooth_yaw:+7.1f}   Pitch: {smooth_pitch:+7.1f}   Roll: {smooth_roll:+7.1f}",
                ]
                for i, text in enumerate(info_lines):
                    cv2.putText(display, text, (15, h + 30 + i * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # 方向指示
                dir_text = ""
                if smooth_yaw > 8:
                    dir_text += "Facing left |"
                elif smooth_yaw < -8:
                    dir_text += "Facing Right |"
                else:
                    dir_text += "Descend |"

                if smooth_pitch > 8:
                    dir_text += "Up"
                elif smooth_pitch < -8:
                    dir_text += "Down"
                else:
                    dir_text += "Flat"

                cv2.putText(display, f">> {dir_text}", (15, h + 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            display[:h, :w] = frame
            cv2.putText(display, "No Face Detected", (15, h + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Head Tracking", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前画面
            cv2.imwrite('saved_frame.jpg', frame)
            print("画面已保存为 saved_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Windows上需要处理销毁事件，否则窗口会卡死
    detector.close()
    print("\n程序已退出")

if __name__ == "__main__":
    main()
