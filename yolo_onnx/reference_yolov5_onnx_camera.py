from libFuncs import YOLOv5
import cv2
import json
import sys
import onnxruntime
import numpy as np

w = r'models/onnx/yolov5s_face_eyeball.onnx'
#w = r'/DS/Datasets/CH_custom/VOC/Human/Himant_body_parts/trained_weights/onnx/best_fp32.onnx'
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

cam_id = 0
write_output = False
output_video_path = "output.avi"
video_size = (640, 480)  #x,y
video_rate = 24.0

winName = 'Deep learning object detection in ONNXRuntime'
w = w.replace('\\', '/')

camera = cv2.VideoCapture(cam_id)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("This webcam's resolution is: %d x %d" % (width, height))
if(write_output is True):
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))

yolov5_detector = YOLOv5(w, cnames, conf_thres=0.2, iou_thres=0.45)

if __name__ == '__main__':
    cv2.namedWindow(winName, 0)
    (grabbed, frame) = camera.read()

    while grabbed:
        boxes, scores, class_ids = yolov5_detector.detect(frame)
        dstimg = yolov5_detector.draw_detections(frame, boxes, scores, class_ids)

        if(write_output is True):
            out.write(dstimg)

        cv2.imshow(winName, dstimg)
        cv2.waitKey(1)
        (grabbed, frame) = camera.read()

    cv2.destroyAllWindows()
