import cv2 as cv
import numpy as np
import os

# ========== 第一步：加载或训练模型 ==========
def load_or_train_model():
    save_file = 'knn_data.npz'
    
    if os.path.exists(save_file):
        print("加载已保存的训练数据...")
        with np.load(save_file) as data:
            train = data['train']
            train_labels = data['train_labels']
    else:
        print("未找到训练数据，正在生成...")
        img = cv.imread('digits.png')
        if img is None:
            print("错误：无法读取 digits.png")
            return None, None
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
        x = np.array(cells)
        
        train = x[:,:50].reshape(-1, 400).astype(np.float32)
        k = np.arange(10)
        train_labels = np.repeat(k, 250)[:, np.newaxis]
        
        np.savez(save_file, train=train, train_labels=train_labels)
    
    # 训练模型
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    return knn

# ========== 第二步：预处理用户输入的图片 ==========
def preprocess_image(image_path):
    """将输入图片处理成模型可识别的格式（20x20灰度图）"""
    
    # 读取图片
    img = cv.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 方法1：如果图片已经是20x20，直接使用
    if gray.shape == (20, 20):
        processed = gray.reshape(1, 400).astype(np.float32)
        return processed
    
    # 方法2：如果不是20x20，需要缩放
    # 先找到数字的轮廓并裁剪
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓（应该是数字）
        contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(contour)
        
        # 裁剪出数字区域
        digit = gray[y:y+h, x:x+w]
        
        # 添加边距使数字居中
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # 缩放到20x20
        resized = cv.resize(padded, (20, 20))
    else:
        # 如果没有找到轮廓，直接缩放整个图片
        resized = cv.resize(gray, (20, 20))
    
    # 反色处理（如果数字是黑底白字）
    if np.mean(resized) > 127:
        resized = 255 - resized
    
    # 转换为模型输入格式
    processed = resized.reshape(1, 400).astype(np.float32)
    return processed

# ========== 第三步：识别单个数字 ==========
def recognize_digit(knn, image_path):
    """识别单张图片中的数字"""
    
    # 预处理图片
    digit_data = preprocess_image(image_path)
    if digit_data is None:
        return None
    
    # 预测
    ret, results, neighbours, dist = knn.findNearest(digit_data, k=5)
    
    # 获取识别结果
    predicted_digit = int(results[0][0])
    confidence = 1.0 - (dist[0][0] / 10000)  # 简单的置信度计算
    
    return predicted_digit, confidence, neighbours[0]

# ========== 第四步：实时摄像头识别 ==========
def real_time_recognition(knn):
    """使用摄像头实时识别手写数字"""
    
    # 打开摄像头
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("按 'q' 退出，按 's' 保存当前画面")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 在画面中心画一个矩形框作为识别区域
        h, w = frame.shape[:2]
        roi_size = 200
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # 绘制识别区域
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 提取ROI区域
        roi = frame[y1:y2, x1:x2]
        
        # 转换为灰度图并预处理
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        
        # 寻找轮廓
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(contour)
            
            # 提取数字区域
            digit = thresh[y:y+h, x:x+w]
            
            # 缩放到20x20
            if digit.size > 0:
                digit_resized = cv.resize(digit, (20, 20))
                digit_data = digit_resized.reshape(1, 400).astype(np.float32)
                
                # 识别
                ret, results, neighbours, dist = knn.findNearest(digit_data, k=5)
                predicted = int(results[0][0])
                
                # 在画面上显示识别结果
                cv.putText(frame, f"Digit: {predicted}", (10, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # 显示提示文字
        cv.putText(frame, "Press 'q' to quit, 's' to save", (10, h - 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示画面
        cv.imshow('Handwritten Digit Recognition', frame)
        
        # 按键处理
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前画面
            cv.imwrite('saved_frame.jpg', frame)
            print("画面已保存为 saved_frame.jpg")
    
    cap.release()
    cv.destroyAllWindows()

# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载/训练模型
    print("正在初始化模型...")
    knn = load_or_train_model()
    
    if knn is None:
        print("模型初始化失败")
        exit()
    
    print("模型已就绪！")
    
    # 选择识别模式
    print("\n请选择识别模式：")
    print("1. 识别单张图片文件")
    print("2. 实时摄像头识别")
    print("3. 测试内置测试集")
    
    choice = input("请输入选项 (1/2/3): ")
    
    if choice == '1':
        # 模式1：识别单张图片
        image_path = input("请输入图片路径: ")
        result = recognize_digit(knn, image_path)
        
        if result:
            digit, confidence, neighbours = result
            print(f"\n识别结果：{digit}")
            print(f"置信度：{confidence:.2f}")
            print(f"近邻数字：{neighbours}")
            
            # 显示图片
            img = cv.imread(image_path)
            if img is not None:
                cv.putText(img, f"Result: {digit}", (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow('Recognition Result', img)
                cv.waitKey(0)
                cv.destroyAllWindows()
    
    elif choice == '2':
        # 模式2：实时摄像头识别
        real_time_recognition(knn)
    
    elif choice == '3':
        # 模式3：测试内置测试集（验证模型准确性）
        print("\n正在测试模型准确率...")
        
        # 重新加载测试数据
        img = cv.imread('digits.png')
        if img is not None:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
            x = np.array(cells)
            
            test = x[:,50:100].reshape(-1, 400).astype(np.float32)
            test_labels = np.repeat(np.arange(10), 250)[:, np.newaxis]
            
            ret, results, neighbours, dist = knn.findNearest(test, k=5)
            matches = results == test_labels
            correct = np.count_nonzero(matches)
            accuracy = correct * 100.0 / results.size
            print(f"测试集准确率：{accuracy:.2f}%")
    
    else:
        print("无效选项")