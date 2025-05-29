from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolov8x.pt")

#personのみ抽出
results = model.predict("ex2.jpg", classes = [0], conf=0.1) 

img = results[0].orig_img
boxes = results[0].boxes

max_mass = 0
xy1 = None
xy2 = None
for box in boxes:
    x1 = box.data[0][0]
    x2 = box.data[0][2]
    y1 = box.data[0][1]
    y2 = box.data[0][3]
    mass = abs(x2-x1)*abs(y2-y1)

    if max_mass < mass:
        max_mass = mass
        xy1 = box.data[0][0:2]
        xy2 = box.data[0][2:4]
    

if xy1 != None and xy2 != None:
    cv2.rectangle(img, xy1.to(torch.int).tolist(), xy2.to(torch.int).tolist(), (0, 0, 255), thickness=3,)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()