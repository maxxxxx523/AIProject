import argparse
import cv2
import os
from ultralytics import YOLO


def detect_image(model, image_path, conf=0.25, show=True, save=True):
    """对单张图片进行猫狗检测"""
    if not os.path.exists(image_path):
        print(f"错误: 图片路径不存在: {image_path}")
        return

    results = model.predict(image_path, conf=conf, verbose=False)
    result = results[0]

    # COCO 类别: cat=15, dog=16
    cat_dog_boxes = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [15, 16]:
            cat_dog_boxes.append(box)

    if not cat_dog_boxes:
        print(f"未检测到猫或狗: {image_path}")
    else:
        print(f"检测结果 - {image_path}:")
        for i, box in enumerate(cat_dog_boxes):
            cls_id = int(box.cls[0])
            label = "猫" if cls_id == 15 else "狗"
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  [{i+1}] {label}  置信度: {conf_val:.2%}  位置: ({x1},{y1})-({x2},{y2})")

    if show or save:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return

        for box in cat_dog_boxes:
            cls_id = int(box.cls[0])
            label = "Cat" if cls_id == 15 else "Dog"
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 0, 255) if cls_id == 15 else (255, 0, 0)  # 猫绿色 狗蓝色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_y = max(y1 - 10, 20)
            cv2.putText(img, f"{label} confidence: {conf_val:.2f}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if save:
            save_dir = "detect_results"
            os.makedirs(save_dir, exist_ok=True)
            name = os.path.basename(image_path)
            save_path = os.path.join(save_dir, f"result_{name}")
            cv2.imwrite(save_path, img)
            print(f"结果已保存: {save_path}")

        if show:
            cv2.imshow("Cat-Dog Detection", img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def detect_folder(model, folder_path, conf=0.25, save=True):
    """批量检测文件夹中的图片"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in exts]

    if not files:
        print(f"文件夹中无图片文件: {folder_path}")
        return

    cat_count = 0
    dog_count = 0
    total = len(files)

    if save:
        save_dir = "detect_results"
        os.makedirs(save_dir, exist_ok=True)

    for f in files:
        fp = os.path.join(folder_path, f)
        results = model.predict(fp, conf=conf, verbose=False)
        result = results[0]

        cat_dog_boxes = []
        has_cat = has_dog = False
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 15:
                has_cat = True
                cat_dog_boxes.append(box)
            elif cls_id == 16:
                has_dog = True
                cat_dog_boxes.append(box)

        if has_cat:
            cat_count += 1
        if has_dog:
            dog_count += 1

        status = ""
        if has_cat and has_dog:
            status = "[猫+狗]"
        elif has_cat:
            status = "[猫]"
        elif has_dog:
            status = "[狗]"
        else:
            status = "[无]"
        print(f"{status} {f}")

        if save and cat_dog_boxes:
            img = cv2.imread(fp)
            if img is None:
                continue

            for box in cat_dog_boxes:
                cls_id = int(box.cls[0])
                label = "Cat" if cls_id == 15 else "Dog"
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 0, 255) if cls_id == 15 else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text_y = max(y1 - 10, 20)
                cv2.putText(img, f"{label} confidence: {conf_val:.2f}", (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            save_path = os.path.join(save_dir, f"result_{f}")
            cv2.imwrite(save_path, img)

    print(f"\n总计 {total} 张图片: 检测到猫 {cat_count} 张, 狗 {dog_count} 张")


def main():
    parser = argparse.ArgumentParser(description="猫狗识别程序 (YOLOv8)")
    parser.add_argument("source", nargs="?", help="图片路径或文件夹路径")
    parser.add_argument("--model", default="runs/detect/train-118/weights/best.pt",
                        help="模型权重路径 (默认: 最新训练结果)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示图片窗口")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存结果图片")
    args = parser.parse_args()

    # 加载模型
    model_path = args.model
    if not os.path.exists(model_path):
        # 尝试找最新的训练权重
        weights = []
        for root, _, files in os.walk("runs"):
            for f in files:
                if f == "best.pt":
                    weights.append(os.path.join(root, f))
        if weights:
            model_path = sorted(weights)[-1]
            print(f"使用模型: {model_path}")
        else:
            print("未找到训练权重，使用 yolo11n.pt 预训练模型")
            model_path = "yolov8n.pt"

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    if args.source:
        src = args.source
        if os.path.isfile(src):
            detect_image(model, src, conf=args.conf,
                        show=not args.no_show, save=not args.no_save)
        elif os.path.isdir(src):
            detect_folder(model, src, conf=args.conf, save=not args.no_save)
        else:
            print(f"路径不存在: {src}")
    else:
        # 交互模式
        print("\n===== 猫狗识别程序 =====")
        print("输入图片路径进行识别，输入 'q' 退出")
        print("输入文件夹路径进行批量检测\n")

        while True:
            src = input("图片路径: ").strip().strip('"')
            if src.lower() == 'q':
                break
            if not src:
                continue
            if os.path.isfile(src):
                detect_image(model, src, conf=args.conf,
                            show=not args.no_show, save=not args.no_save)
            elif os.path.isdir(src):
                detect_folder(model, src, conf=args.conf, save=not args.no_save)
            else:
                print(f"路径无效: {src}")


if __name__ == "__main__":
    main()
