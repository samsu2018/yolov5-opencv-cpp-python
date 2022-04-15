# -*- coding: utf-8 -*-
###################################################################
# Object detection - YOLOv5 - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : https://github.com/hpc203/yolov5-dnn-cpp-python
# Modify : Sam Su (October 14, 2021)
# This version only for 640x640
##################################################################
import cv2
import argparse
import numpy as np
# import time


def get_obj(img, confThreshold, nmsThreshold, objThreshold, model, inpWidth=640, inpHeight=640):
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default=img, help="image path")
    parser.add_argument('--net_type', default=model, choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument('--confThreshold', default=confThreshold, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=nmsThreshold, type=float, help='nms iou thresh') # low are best
    parser.add_argument('--objThreshold', default=objThreshold, type=float, help='object confidence')
    args = parser.parse_args()
    yolonet = yolov5(args.net_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)

    srcimg = cv2.imread(args.imgpath)
    dets = yolonet.detect(srcimg, inpWidth, inpHeight)
    nms_dets, frame = yolonet.postprocess(srcimg, dets, inpWidth, inpHeight)
    return nms_dets, frame

def resize(img, resize_width=416):
    (h, w) = img.shape[:2] #tuple
    m = max(w, h)
    ratio = resize_width / m
    new_w, new_h = int(ratio * w), int(ratio *h)
    if new_w > 0 and new_h > 0:
        img_resized = cv2.resize(img, (new_w, new_h))
    else:
        img_resized = img
    return img_resized


class yolov5():
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5, inpWidth=640, inpHeight=640):
        with open('model/coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)

        self.net = cv2.dnn.readNet(yolo_type + '.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def postprocess(self, frame, outs, inpWidth, inpHeight):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / inpHeight, frameWidth / inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    center_x = int(detection[0] * ratiow)
                    center_y = int(detection[1] * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width / 2) # x
                    top = int(center_y - height / 2) # y
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        nms_classIds = []
        nms_confidences = []
        nms_boxes = []
        nms_dets = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            nms_left = box[0] # x
            nms_top = box[1]  # y
            nms_width = box[2]
            nms_height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], nms_left, nms_top, nms_left + nms_width, nms_top + nms_height)
            # print('After NMS:',classIds[i], confidences[i], nms_left, nms_top, nms_left + nms_width, nms_top + nms_height)
            nms_classIds.append(classIds[i])
            label = str(self.classes[classIds[i]])
            nms_confidences.append(confidences[i])
            nms_boxes.append([left, top, width, height])
            nms_dets.append([label, confidences[i], nms_left, nms_top, nms_left + nms_width, nms_top + nms_height, classIds[i]])
        return nms_dets, frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def detect(self, srcimg, inpWidth, inpHeight):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (inpWidth, inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # outs[i] = outs[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outs[i] = outs[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, self.no))
        z = np.concatenate(z, axis=1)
        return z

# =============================================================================
# The following main functions are used for standalong testing
# =============================================================================
# if __name__ == "__main__":
#     imgpath = 'bus.jpg'
#     tStart = time.time()
#     dets, frame = get_obj(imgpath)
#     print(dets)
#     cv2.imwrite(output.jpg, frame)
#     print('Spend time:{}'.format(time.time()-tStart))