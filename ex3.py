from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolov8x.pt")

#personのみ抽出
results = model.predict("ex2.jpg", classes = [0], conf=0.1) 

img = results[0].orig_img
boxes = results[0].boxes


for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    cv2.rectangle(img, xy1.to(torch.int).tolist(), xy2.to(torch.int).tolist(), (0, 0, 255), thickness=3,)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()