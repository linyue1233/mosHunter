import predict
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import time

# 已部署到本地，延时1s-1.3s(备用方法)

vc = cv2.VideoCapture('mosMove.mp4')
rval, firstFrame = vc.read()

firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_CUBIC)
# 获取帧率
fps = vc.get(propId=5)
gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)  # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)  # 高斯去噪
prveFrame = firstFrame.copy()

frameCount = -1

rectangles = []
# 遍历视频的每一帧
while True:
    (ret, frame) = vc.read()
    # 如果没有，则结束
    if not ret:
        break

    frameCount = frameCount + 1
    if frameCount % 3 == 0:
        rectangles = []
        cv2.imshow('frame_with_result', frame)
        image = Image.fromarray(frame)
        # 调用本地接口约耗时1s，所以会存在1s的延时
        data = predict.checkThing(image)
        x1, y1 = np.size(image)

        for sign_obj in data:
            if sign_obj['probability'] < 0.5:
                continue
            x, y, w, h = sign_obj['boundingBox']['left'], sign_obj['boundingBox']['top'], sign_obj['boundingBox'][
                'width'], sign_obj['boundingBox']['height']
            x, y, w, h = int(x * x1), int(y * y1), int(w * x1), int(h * y1)
            # print(x, y, w, h)
            rectangles.append((x, y, w, h))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in rectangles:
        # print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame_with_result', frame)

    # 处理按键效果
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):  # 按下q时，退出当前帧
        break
    elif key == ord(" "):
        cv2.waitKey(0)
    else:
        continue

vc.release()
cv2.destroyAllWindows()
