import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex2.jpg", classes = [0], conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes


for box in boxes:
    # 物体領域の始点xy座標を得る．
    xy1 = box.data[0][0:2]
    # 物体領域の終点xy座標を得る．
    xy2 = box.data[0][2:4]
    cv2.rectangle(
        img,
        xy1.to(torch.int).tolist(),
        xy2.to(torch.int).tolist(),
        (0, 0, 255),
        thickness=3,
    )

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()