from ultralytics import YOLO as yolo

def main():
    model=yolo('yolov8n.pt')
    model.train(data="data.yml",epochs=10,imgsz=640,batch=4,device='cuda:0')

if __name__=="__main__":
    main()