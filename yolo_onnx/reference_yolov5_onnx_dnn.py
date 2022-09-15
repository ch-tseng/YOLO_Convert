import cv2
import json
import sys
import numpy as np

w = r'/DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/ONNX/yolov7_fp32.onnx'
#w = r'/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/trained_weights/onnx/yolov7e6_fp32.onnx'
img = r"/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/dataset/images/00001070.jpg"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
#colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(255,255,255),(0,255,255),(255,255,255),(255,255,255)]
#cnames = ['0','eye','nose','mouth','face','mface','hface','head','body']
colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
cnames = ['head','upper','lower','body']

w = w.replace('\\', '/')
img = img.replace('\\', '/')

def build_model(is_cuda, onnx_path):
    net = cv2.dnn.readNet(onnx_path)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


print(sys.argv)
is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
net = build_model(True, w)

inputImage = format_yolov5(cv2.imread(img))
outs = detect(inputImage, net)

class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
print(class_ids, confidences, boxes)

for id, box in enumerate(boxes):
    (x1,y1,x2,y2) = (box[0], box[1], box[0]+box[2], box[1]+box[3])
    cv2.rectangle(inputImage, (x1,y1), (x2, y2), colors[class_ids[id]], 2)
    cv2.putText(inputImage, cnames[class_ids[id]], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  colors[class_ids[id]], 1, cv2.LINE_AA)
    
cv2.imshow('test', inputImage)
cv2.waitKey(0)
