import struct
import numpy as np
from PIL import Image, ImageFilter
import os
import argparse


def binarize(arr, method="otsu", block_size=15, c=10):
    """
    Binarize a grayscale image to make digit features sharp.
    arr: uint8 numpy array.
    method: "otsu" (global) or "adaptive" (local threshold, better for uneven lighting).
    Returns binary uint8 array (0 or 255).
    """
    if method == "otsu":
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(arr)
        return ((arr > thresh) * 255).astype(np.uint8)
    elif method == "adaptive":
        from skimage.filters import threshold_local
        thresh = threshold_local(arr, block_size, method='gaussian', offset=c / 255.0)
        return ((arr > thresh) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown binarize method: {method}")


def enhance_contrast(arr, percentile=5):
    """
    Stretch the histogram so the digit stands out from the background.
    Pixels below `percentile` are mapped to 0, above (100-percentile) to 255.
    """
    lo = np.percentile(arr, percentile)
    hi = np.percentile(arr, 100 - percentile)
    if hi <= lo:
        return arr
    stretched = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)


def center_and_scale(arr, canvas_size=28, margin=4):
    """
    Center the digit in a canvas_size x canvas_size frame.
    arr: uint8 numpy array, white digit (non-zero) on black background (0).
    Works at any input resolution — the digit is cropped first, then scaled.
    Returns centered uint8 array of shape (canvas_size, canvas_size).
    """
    # Find bounding box of the digit (non-zero pixels)
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add a small padding around the bbox before cropping (in original pixels)
    h, w = arr.shape
    pad = max(2, int(min(h, w) * 0.02))
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    # Crop to bounding box
    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    ch, cw = cropped.shape

    # Scale to fit inside (canvas_size - 2*margin) preserving aspect ratio
    target_size = canvas_size - 2 * margin
    scale = target_size / max(ch, cw)
    new_h, new_w = int(ch * scale), int(cw * scale)
    # Ensure at least 1 pixel
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    scaled = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS))

    # Place centered in canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    offset_h = (canvas_size - new_h) // 2
    offset_w = (canvas_size - new_w) // 2
    canvas[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = scaled

    return canvas


def image_to_mnist_array(image_path, invert=True, binarize_method=None,
                         enhance=True, contrast_percentile=5):
    """
    Convert an image to a 28x28 grayscale numpy array matching MNIST format.

    Pipeline (correct order):
    1. Convert to grayscale at original resolution
    2. Enhance contrast (stretch histogram)
    3. Invert so digit is white on black background (MNIST convention)
    4. Binarize to make strokes sharp (optional)
    5. Find digit bounding box, crop, scale, and center in 28x28

    Parameters:
        image_path: path to input image
        invert: if True, invert colors for white-digit-on-black (MNIST style)
        binarize_method: None, "otsu", or "adaptive" — thresholding for sharp features
        enhance: if True, apply contrast stretching
        contrast_percentile: percentile for contrast stretching
    """
    img = Image.open(image_path).convert("L")  # grayscale, original resolution

    arr = np.array(img, dtype=np.uint8)

    # Enhance contrast so the digit pops out from the background
    if enhance:
        arr = enhance_contrast(arr, percentile=contrast_percentile)

    # Invert: MNIST uses white digit (255) on black background (0)
    if invert:
        arr = 255 - arr

    # Binarize to make digit edges crisp (removes anti-aliasing fuzz)
    if binarize_method is not None:
        arr = binarize(arr, method=binarize_method)

    # Crop to digit bounding box, scale proportionally, center in 28x28
    arr = center_and_scale(arr)

    return arr


def save_idx_images(filepath, images):
    """
    Save images in MNIST IDX format.
    images: numpy array of shape (N, 28, 28) or (28, 28), dtype uint8.
    """
    if images.ndim == 2:
        images = images[np.newaxis, :, :]

    magic = 2051  # image file magic number
    n = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]

    header = struct.pack(">IIII", magic, n, rows, cols)
    data = images.tobytes()

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(data)

    print(f"Saved {n} image(s) ({rows}x{cols}) to {filepath}")


def save_idx_labels(filepath, labels):
    """
    Save labels in MNIST IDX label format.
    labels: list/array of integers.
    """
    labels = np.array(labels, dtype=np.uint8)
    magic = 2049  # label file magic number
    n = len(labels)

    header = struct.pack(">II", magic, n)
    data = labels.tobytes()

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(data)

    print(f"Saved {n} label(s) to {filepath}")


def load_idx_images(filepath):
    """Load MNIST IDX image file, return numpy array (N, 28, 28)."""
    with open(filepath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def load_idx_labels(filepath):
    """Load MNIST IDX label file, return numpy array (N,)."""
    with open(filepath, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049")
        return np.frombuffer(f.read(), dtype=np.uint8)


def batch_convert(input_dir, output_dir, label=None,
                  binarize_method=None, enhance=True):
    """Convert all images in a directory to a single MNIST IDX file."""
    images = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

    files = sorted(os.listdir(input_dir))
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            fpath = os.path.join(input_dir, fname)
            arr = image_to_mnist_array(fpath, binarize_method=binarize_method,
                                       enhance=enhance)
            images.append(arr)

    if not images:
        print(f"No images found in {input_dir}")
        return

    images = np.stack(images)
    os.makedirs(output_dir, exist_ok=True)
    save_idx_images(os.path.join(output_dir, "images.idx3-ubyte"), images)

    if label is not None:
        labels = [label] * len(images)
        save_idx_labels(os.path.join(output_dir, "labels.idx1-ubyte"), labels)


def append_to_existing(images_path, new_images):
    """Append images to an existing IDX file."""
    existing = load_idx_images(images_path)
    combined = np.concatenate([existing, new_images])
    save_idx_images(images_path, combined)


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to MNIST IDX format"
    )
    sub = parser.add_subparsers(dest="cmd")

    # Single image
    p = sub.add_parser("single", help="Convert a single image")
    p.add_argument("input", help="Input image path")
    p.add_argument("output", help="Output .idx3-ubyte path")
    p.add_argument("--no-invert", action="store_true",
                   help="Skip color inversion")
    p.add_argument("--binarize", choices=["otsu", "adaptive"], default=None,
                   help="Binarization method for sharper features")
    p.add_argument("--no-enhance", action="store_true",
                   help="Skip contrast enhancement")

    # Batch with labels
    p = sub.add_parser("batch", help="Batch convert a directory of images")
    p.add_argument("input_dir", help="Directory containing images")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("--label", type=int,
                   help="Label to assign to all images")
    p.add_argument("--binarize", choices=["otsu", "adaptive"], default=None,
                   help="Binarization method")

    # Append
    p = sub.add_parser("append", help="Append image to existing IDX file")
    p.add_argument("input", help="Input image path")
    p.add_argument("idx_file", help="Existing .idx3-ubyte file")

    # Info
    p = sub.add_parser("info", help="Print info about an IDX file")
    p.add_argument("file", help="IDX file path")

    args = parser.parse_args()

    if args.cmd == "single":
        arr = image_to_mnist_array(args.input, invert=not args.no_invert,
                                   binarize_method=args.binarize,
                                   enhance=not getattr(args, "no_enhance", False))
        save_idx_images(args.output, arr)

    elif args.cmd == "batch":
        batch_convert(args.input_dir, args.output_dir, args.label,
                      binarize_method=getattr(args, "binarize", None))

    elif args.cmd == "append":
        arr = image_to_mnist_array(args.input)
        arr = arr[np.newaxis, :, :]
        append_to_existing(args.idx_file, arr)

    elif args.cmd == "info":
        try:
            imgs = load_idx_images(args.file)
            print(f"Images: {imgs.shape[0]}, size: {imgs.shape[1]}x{imgs.shape[2]}")
            print(f"Value range: [{imgs.min()}, {imgs.max()}]")
        except Exception:
            try:
                labels = load_idx_labels(args.file)
                print(f"Labels: {labels.shape[0]}")
                print(f"Classes: {np.unique(labels).tolist()}")
            except Exception as e:
                print(f"Not a valid IDX file: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
