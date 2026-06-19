import torch
import numpy as np
import tkinter as tk
from PIL import Image
from img_to_mnist import image_to_mnist_array, load_idx_images


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="best_model.pth", device=None):
    if device is None:
        device = get_device()
    import importlib
    m = importlib.import_module("t1")
    SimpleCNN = m.SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, image_path, device=None, binarize=None):
    """Predict a single image file (png, jpg, etc.)."""
    if device is None:
        device = get_device()
    arr = image_to_mnist_array(image_path, invert=True, binarize_method=binarize)
    return _predict_array(model, arr, device)


def save_processed_image(image_path, output_path, invert=True, binarize=None):
    """Save the 28x28 preprocessed image to see what the model sees."""
    arr = image_to_mnist_array(image_path, invert=invert, binarize_method=binarize)
    img = Image.fromarray(arr, mode="L")
    img = img.resize((280, 280), Image.NEAREST)  # scale up for viewing
    img.save(output_path)
    print(f"Processed image saved to {output_path} ({arr.shape})")
    print(f"Value range: [{arr.min()}, {arr.max()}], mean: {arr.mean():.1f}")


def predict_idx(model, idx_path, device=None):
    """Predict all images in an IDX file."""
    if device is None:
        device = get_device()
    images = load_idx_images(idx_path)  # shape: (N, 28, 28)
    results = []
    for i in range(images.shape[0]):
        arr = images[i]
        pred = _predict_array(model, arr, device)
        results.append(pred)
    return results


def _predict_array(model, arr, device):
    """
    arr: (28, 28) numpy uint8 array, white digit on black background.
    Returns predicted digit (int).
    """
    img = arr.astype(np.float32) / 255.0
    img = (img - 0.1307) / 0.3081
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence, probs[0].cpu().numpy()


def show_popup_single(filepath, digit, confidence, probs):
    root = tk.Tk()
    root.title("预测结果")
    root.resizable(False, False)

    # header
    tk.Label(root, text="MNIST 数字识别", font=("Microsoft YaHei", 16, "bold")).pack(pady=(15, 0))
    tk.Label(root, text=f"文件: {filepath}", font=("Microsoft YaHei", 9), fg="gray").pack()

    # predicted digit
    tk.Label(root, text=str(digit), font=("Consolas", 72, "bold"), fg="#1a73e8").pack(pady=(10, 0))
    tk.Label(
        root,
        text=f"预测结果: {digit}　　置信度: {confidence:.2%}",
        font=("Microsoft YaHei", 14),
        fg="#333",
    ).pack(pady=(0, 10))

    # probability bars
    frame = tk.Frame(root)
    frame.pack(padx=30, pady=(0, 15), fill=tk.BOTH)

    max_p = max(probs)
    for d, p in enumerate(probs):
        row = tk.Frame(frame)
        row.pack(fill=tk.X, pady=1)

        tk.Label(row, text=str(d), font=("Consolas", 11, "bold"), width=2, anchor="e").pack(side=tk.LEFT)
        bar_len = int(p / max_p * 200) if max_p > 0 else 0
        bar_color = "#1a73e8" if p == max_p else "#bbb"
        bar_frame = tk.Frame(row, bg=bar_color, width=bar_len, height=18)
        bar_frame.pack(side=tk.LEFT, padx=(4, 0))
        bar_frame.pack_propagate(False)

        pct_label = tk.Label(row, text=f"{p:.2%}", font=("Consolas", 9), width=7, anchor="w", fg="#555")
        pct_label.pack(side=tk.LEFT, padx=(4, 0))

    tk.Button(root, text="确定", command=root.destroy, width=10, font=("Microsoft YaHei", 11)).pack(pady=(0, 15))

    root.eval(f'tk::PlaceWindow {root} center')
    root.mainloop()


def show_popup_idx(filepath, results):
    root = tk.Tk()
    root.title(f"预测结果 - {len(results)} 张图片")
    root.resizable(True, True)

    tk.Label(
        root,
        text=f"IDX 文件: {filepath}\n共 {len(results)} 张图片",
        font=("Microsoft YaHei", 12, "bold"),
        fg="#333",
    ).pack(pady=(15, 10), padx=20)

    text = tk.Text(root, font=("Consolas", 10), wrap=tk.NONE, padx=15, pady=10,
                   bg="#fafafa", relief=tk.FLAT, width=60, height=min(len(results), 25))
    text.pack(padx=15, pady=(0, 5), fill=tk.BOTH, expand=True)

    for i, (digit, conf, probs) in enumerate(results):
        text.insert(tk.END, f"[{i:3d}] 数字: {digit}    置信度: {conf:.2%}\n")
        top3 = np.argsort(probs)[::-1][:3]
        for d in top3:
            bar = "█" * int(probs[d] * 25)
            text.insert(tk.END, f"       {d}: {bar} {probs[d]:.4f}\n")
        text.insert(tk.END, "\n")

    text.config(state=tk.DISABLED)

    scrollbar = tk.Scrollbar(root, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.config(yscrollcommand=scrollbar.set)

    tk.Button(root, text="确定", command=root.destroy, width=10,
              font=("Microsoft YaHei", 11)).pack(pady=(5, 15))

    root.eval(f'tk::PlaceWindow {root} center')
    root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict digits using trained MNIST model")
    parser.add_argument("input", help="Input image file (.png, .jpg, etc.) or .idx3-ubyte file")
    parser.add_argument("--model", default="best_model.pth", help="Model weights file")
    parser.add_argument("--show-processed", metavar="OUTPUT", help="Save the 28x28 preprocessed image for inspection")
    parser.add_argument("--binarize", choices=["otsu", "adaptive"], default=None,
                        help="Binarization method for sharper features")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable contrast enhancement")
    args = parser.parse_args()

    if args.show_processed:
        save_processed_image(args.input, args.show_processed, binarize=args.binarize)
        return

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from {args.model}...")
    model, device = load_model(args.model, device)
    print("Model loaded.\n")

    if args.input.endswith(".idx3-ubyte"):
        results = predict_idx(model, args.input, device)
        show_popup_idx(args.input, results)
    else:
        digit, conf, probs = predict_image(model, args.input, device, binarize=args.binarize)
        show_popup_single(args.input, digit, conf, probs)


if __name__ == "__main__":
    main()
