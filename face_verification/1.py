import cv2
import os
import numpy as np
from PIL import Image

# ========== 1. 人脸录入（注册） ==========
def register_face(name):
    """录入人脸，保存到 dataset 文件夹"""
    # 加载 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 绘制矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 保存人脸图像
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            
            # 创建用户文件夹
            user_dir = f'dataset/{name}'
            os.makedirs(user_dir, exist_ok=True)
            cv2.imwrite(f'{user_dir}/{count}.jpg', face_img)
            
            # 显示提示
            cv2.putText(frame, f'Captured: {count}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Register Face - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'✅ 人脸录入完成！共 {count} 张照片，保存在 dataset/{name}/')


# ========== 2. 训练人脸识别器 ==========
def train_recognizer():
    """训练 LBPH 人脸识别器"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    # 遍历 dataset 文件夹
    for user_name in os.listdir('dataset'):
        user_path = f'dataset/{user_name}'
        if not os.path.isdir(user_path):
            continue
        
        label_map[current_label] = user_name
        
        for img_file in os.listdir(user_path):
            img_path = f'{user_path}/{img_file}'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            faces.append(img)
            labels.append(current_label)
        
        current_label += 1
    
    # 训练
    recognizer.train(faces, np.array(labels))
    recognizer.save('trainer.yml')
    
    print(f'✅ 训练完成！共 {len(label_map)} 个用户')
    return recognizer, label_map


# ========== 3. 人脸识别 ==========
def recognize_face():
    """实时人脸识别"""
    # 加载训练好的识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # 加载标签映射
    label_map = {}
    for idx, name in enumerate(os.listdir('dataset')):
        if os.path.isdir(f'dataset/{name}'):
            label_map[idx] = name
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 提取人脸 ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            # 预测
            label, confidence = recognizer.predict(roi_gray)
            
            # 置信度越低越准确（0为最准确）
            if confidence < 80:
                name = label_map.get(label, 'Unknown')
                confidence_text = f'{100 - confidence:.1f}%'
                color = (0, 255, 0)  # 绿色 - 识别成功
            else:
                name = 'Unknown'
                confidence_text = f'{confidence:.1f}'
                color = (0, 0, 255)  # 红色 - 未知
            
            # 绘制结果
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{name} ({confidence_text})', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Face Recognition - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ========== 4. 主程序 ==========
if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    
    while True:
        print('\n' + '='*40)
        print('1. 录入人脸')
        print('2. 训练识别器')
        print('3. 开始人脸识别')
        print('4. 退出')
        print('='*40)
        
        choice = input('请选择操作: ')
        
        if choice == '1':
            name = input('请输入姓名: ')
            register_face(name)
        elif choice == '2':
            train_recognizer()
        elif choice == '3':
            recognize_face()
        elif choice == '4':
            break
        else:
            print('❌ 无效输入，请重新选择')