from libFuncs import YOLOv5
import cv2
import json
import sys
import onnxruntime
import numpy as np

w = r'/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/trained_weights/onnx/yolov5s_fp32.onnx'
#w = r'/DS/Datasets/CH_custom/VOC/Human/Himant_body_parts/trained_weights/onnx/best_fp32.onnx'
img_path = r"/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/dataset/images/00002442.jpg"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.2
colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(255,255,255),(0,255,255),(255,255,255),(255,255,255)]
cnames = ['0','eye','nose','mouth','face','mface','hface','head','body']
#colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
#cnames = ['head','upper','lower','body']
#cnames = ['head', 'body']

w = w.replace('\\', '/')
img_path = img_path.replace('\\', '/')

if __name__ == '__main__':
    yolov5_detector = YOLOv5(w, cnames, conf_thres=0.2, iou_thres=0.45)
    srcimg = cv2.imread(img_path)
    
    # Detect Objects
    boxes, scores, class_ids = yolov5_detector.detect(srcimg)
    #print(boxes)
    # Draw detections
    dstimg = yolov5_detector.draw_detections(srcimg, boxes, scores, class_ids)
    cv2.imwrite('/DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/ONNX/output.jpg', dstimg)
    winName = 'Deep learning object detection in ONNXRuntime'
    #cv2.namedWindow(winName, 0)
    #cv2.imshow(winName, dstimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
