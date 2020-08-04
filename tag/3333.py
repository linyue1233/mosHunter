import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from tags import mosHunter  as tc

# 专门用来写测试案例

vc = cv2.VideoCapture('mosMove.mp4')
rval, firstFrame = vc.read()

firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_CUBIC)

gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)  # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)  # 高斯去噪


prveFrame = firstFrame.copy()
frameCount = 0

# 遍历视频的每一帧
while True:
    (ret, frame) = vc.read()

    frameCount = frameCount + 1
    # 如果没有，则结束
    if not ret:
        break
    if frameCount % 3 != 0:
        continue

    # 对拿到的帧进行预处理
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(prveFrame, gray_frame)
    prveFrame = gray_frame.copy()

    # 忽略较小的差别
    retVal, thresh = cv2.threshold(frameDiff, 25, 150, cv2.THRESH_BINARY)

    # 对阈值图像进行填充补洞
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




    # 遍历轮廓
    for contour in contours:
        if cv2.contourArea(contour) < 400 or cv2.contourArea(contour) > 1000:  # 面积阈值
            continue
            # 计算最小外接矩形
        (x, y, w, h) = cv2.boundingRect(contour)
        cut_img = frame[y: y + h, x: x + w]
        cv2.imwrite(r"/Users/hayashietsu/Desktop/game/pic/" + str(frameCount) + ".jpg", cut_img)
        image_path = "/Users/hayashietsu/Desktop/game/pic/" + str(frameCount) + ".jpg"
        image_data = open(image_path, "rb").read()
        image = Image.open(BytesIO(image_data))
        if tc.showTag(image) == 'mosquito':
            cv2.imwrite(r"/Users/hayashietsu/Desktop/game/test/" + str(frameCount) + ".jpg", cut_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame_with_result', frame)

        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):  # 按下q时，退出当前帧
            break
        elif key == ord(" "):
            cv2.waitKey(0)
        else:
            continue

vc.release()
cv2.destroyAllWindows()
