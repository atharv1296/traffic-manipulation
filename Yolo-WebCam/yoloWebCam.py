from ultralytics import YOLO
import cv2
import cvzone
import math


print("HI")
import torch
print(torch.cuda.is_available())  # Should output "True" for CUDA support
#cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)
cap = cv2.VideoCapture("../Yolo/videos/tra2.mp4")


model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag" "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Normal Box around object
            #x1,y1,x2,y2= box.xyxy[0]
            #x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)


            #Fancy or better box around object
            #Bounding Box below
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            #Confidence
            conf = math.ceil(box.conf[0]*100) / 100
            print(conf)

            #Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)