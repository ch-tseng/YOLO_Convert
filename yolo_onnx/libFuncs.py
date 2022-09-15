import cv2
import json
import sys
import onnxruntime
import numpy as np

class YOLOv5:
    def __init__(self, path, cnames, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.class_names = cnames
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        boxes, scores, class_ids = self.process_output(outputs)

        return boxes, scores, class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def get_class_predicted(self, datas):
        max_id = 0
        max_score = 0
        for i, d in enumerate(datas):
            if d > max_score:
                max_id = i
                max_score = d

        return max_id, max_score

    def process_output(self, output):
        boxes, scores, class_ids = [], [], []
        predictions = output[0]
        print(predictions.shape)
        for p in predictions:
            if p[4]>self.conf_threshold:
                print(p)
                id, s =  self.get_class_predicted(p[5:])
                x = p[0] - (p[2]/2)
                y = p[1] - (p[3]/2)
                boxes.append( (x, y, p[2], p[3] ) )
                scores.append(s)
                class_ids.append(id)

        boxes, scores, class_ids = self.nms_boxes(class_ids, boxes, scores)
        boxes = self.rescale_boxes(boxes)
        return boxes, scores, class_ids

    def nms_boxes(self, class_ids, boxes, confidences):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_boxes, result_confidences, result_class_ids

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        #input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        #boxes = np.divide(boxes, input_shape, dtype=np.float32)

        rw, rh = self.img_width/self.input_width, self.img_height/self.input_height
        #print((rw, rh), (self.img_width,self.img_height), (self.input_width,self.input_height))
        bbox = []
        for box in boxes:
            nx,ny,nw,nh = int(box[0]*rw), int(box[1]*rh), int(box[2]*rw), int(box[3]*rh)
            bbox.append([nx,ny,nw,nh])
        #boxes *= np.array([rw, rh, rw, rh])
        return bbox

    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])

            # Draw rectangle
            #print((self.img_width, self.img_height), (x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return image


class YOLOv7:
    def __init__(self, path, cnames, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = cnames
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        boxes, scores, class_ids = self.process_output(outputs[0])

        return boxes, scores, class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def process_output(self, output):
        print('output', output.shape)
        #print('output', output)

        boxes, scores, class_ids = [], [], []
        predictions = output
        for p in predictions:
            scores.append(p[4])
            class_ids.append(int(p[5]))
            boxes.append( (p[1], p[2], p[3], p[4] ) )

        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        #input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        #boxes = np.divide(boxes, input_shape, dtype=np.float32)

        rw, rh = self.img_width/self.input_width, self.img_height/self.input_height
        print((rw, rh), (self.img_width,self.img_height), (self.input_width,self.input_height))
        bbox = []
        for box in boxes:
            nx,ny,nw,nh = int(box[0]*rw), int(box[1]*rh), int(box[2]*rw), int(box[3]*rh)
            bbox.append([nx,ny,nw,nh])
        #boxes *= np.array([rw, rh, rw, rh])
        return bbox

    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw rectangle
            print((self.img_width, self.img_height), (x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return image
