# 1. 单张图片检测（显示窗口 + 保存结果）
run
```
python vividation.py path/to/image.jpg
```
# 2. 批量检测整个文件夹
run
```
python vividation.py data/images/val
```

# 3. 交互模式（不传参数，逐步输入路径）
run
```
python vividation.py
```

# 4. 其他参数
```
python vividation.py cat.jpg --model runs/detect/train-118/weights/best.pt --conf 0.5 --no-show
```