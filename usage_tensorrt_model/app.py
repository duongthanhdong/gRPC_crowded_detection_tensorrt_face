from yolo_with_tensorrt import TrtYOLO
import cv2
import time

print('load')
test = TrtYOLO(model='yolov4-tiny-3l-416', input_shape=(416,416), category_num=1, cuda_ctx=None)
print("finish")
image = cv2.imread("matnghieng.jpg")

start = time.time()

boxes, scores, classes = test.detect(image)

end = time.time()
fps = 1/(end-start)
print(fps)

for bb in boxes:
    x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 2)

# print("bboxes",boxes)
# print('scroces',scores)
# print('class',classes)

cv2.imshow("image.jpg",image)
cv2.waitKey(0)