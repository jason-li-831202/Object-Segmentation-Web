import math
import cv2
import random
import numpy as np
import onnxruntime as ort
import os

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ObjectOnnxDetector(object):
    _defaults = {
        "model_path": None,
        "classes_path" : None,
        "box_aspect_ratio" : None,
        "box_stretch" : None,
        "box_score" : None,
        "box_nms_iou" : None,
    }

    @classmethod
    def set_defaults(cls, config) :
        print('*'*28)
        print('* Check model settings *')
        print('*'*28)
        for key in config:
            print ('  ', key,'=', config[key])
        cls._defaults = config

    @classmethod
    def check_defaults(cls):
        return cls._defaults
        
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.box_points = []
        self.mask_maps = []
        self.style = None
        self.keep_ratio = False

        classes_path = os.path.expanduser(self.classes_path)
        if (os.path.isfile(classes_path) is False):
            raise Exception("%s is not exist." % classes_path)

        model_path = os.path.expanduser(self.model_path)
        if (os.path.isfile(model_path) is False):
            raise Exception("%s is not exist." % model_path)
        assert model_path.endswith('.onnx'), 'Onnx Parameters must be a .onnx file.'

        self._get_class(classes_path)
        self._load_model_onnxruntime_version(model_path)
        self._get_input_details()
        self._get_output_details()

    def _get_class(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]
        self.priority_target = self.class_names
        self.priority_target.append("unknown")

        get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
        self.colors_dict = dict(zip(list(self.class_names), get_colors))
        self.colors_dict["unknown"] = '#ffffff'

    def _load_model_onnxruntime_version(self, model_path) :
        if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
            self.providers = 'CUDAExecutionProvider'
        else :
            self.providers = 'CPUExecutionProvider'
        self.session = ort.InferenceSession(model_path, providers= [self.providers] )
        print("AutoLabel Inference Version : ", self.providers)

    def _get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.model_shapes = model_inputs[0].shape
        self.model_height = self.model_shapes[2]
        self.model_width = self.model_shapes[3]

    def _get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_layers_count = len(self.output_names)

    def _process_box_output(self, box_output, conf_thres, num_masks=32):
        class_ids = []
        confidences = []
        box_predictions = []
        mask_predictions = []

        if (num_masks != None) :
            num_classes = box_output.shape[1] - num_masks - 4

        box_output = np.squeeze(box_output).T
        for predictions in box_output:
            if (num_masks != None) :
                scores = predictions[4:4+num_classes]
            else :
                scores = predictions[4:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_thres :
                if (num_masks != None) :
                    mask_predictions.append(predictions[num_classes+4:])
                x, y, w, h = predictions[0].item(), predictions[1].item(), predictions[2].item(), predictions[3].item() 
                class_ids.append(classId)
                confidences.append(float(confidence))
                box_predictions.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1)) #x1, y1, x2, y2
        return box_predictions, confidences, class_ids, np.array(mask_predictions)

    def _process_mask_output(self, mask_output, input_boxes, mask_predictions, indices):
        if mask_predictions.shape[0] == 0:
            return []
        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        mask_boxes = self.rescale_boxes(input_boxes,
                                   (self.input_height, self.input_width),
                                   (mask_height, mask_width))
        

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(indices), self.input_height, self.input_width))
        blur_size = (int(self.input_width / mask_width), int(self.input_height / mask_height))
        if len(indices) > 0:
            for i, indice in enumerate(indices):
                scale_x1 = int(math.floor(mask_boxes[indice][0]) )
                scale_y1 = int(math.floor(mask_boxes[indice][1]) )
                scale_x2 = int(math.ceil(mask_boxes[indice][2]) )
                scale_y2 = int(math.ceil(mask_boxes[indice][3]) )
                
                x1 = int(math.floor(input_boxes[indice][0]))
                y1 = int(math.floor(input_boxes[indice][1]))
                x2 = int(math.ceil(input_boxes[indice][2]))
                y2 = int(math.ceil(input_boxes[indice][3]))

                scale_crop_mask = masks[indice][scale_y1:scale_y2, scale_x1:scale_x2]
                crop_mask = cv2.resize(scale_crop_mask,
                                (x2 - x1, y2 - y1),
                                interpolation=cv2.INTER_CUBIC)

                crop_mask = cv2.blur(crop_mask, blur_size)

                crop_mask = (crop_mask > 0.5).astype(np.uint8)
                mask_maps[i, y1:y2, x1:x2] = crop_mask
        return mask_maps
    
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
    
    def resize_image_format(self, srcimg, frame_resize):
        padh, padw, newh, neww = 0, 0, frame_resize, frame_resize
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = frame_resize, int(frame_resize / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
                padw = int((frame_resize - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, frame_resize - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(frame_resize * hw_scale) + 1, frame_resize
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
                padh = int((frame_resize - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, frame_resize - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (frame_resize, frame_resize), interpolation=cv2.INTER_CUBIC)
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        return img, newh, neww, ratioh, ratiow, padh, padw

    def adjust_boxes_ratio(self, bounding_box, ratio, stretch_type) :
        """ Adjust the aspect ratio of the box according to the orientation """
        xmin, ymin, width, height = bounding_box 
        width = int(width)
        height = int(height)
        xmax = xmin + width
        ymax = ymin + height
        if (ratio != None) :
            ratio = float(ratio)
        else :
            return (xmin, ymin, xmax, ymax)
        
        center = ( (xmin + xmax) / 2, (ymin + ymax) / 2 )
        if (stretch_type == "居中水平") :
            # print("test : 居中水平")
            changewidth = int(height * (1/ratio))
            xmin = center[0] - changewidth/2
            xmax = xmin + changewidth
        elif (stretch_type == "居中垂直") :
            # print("test : 居中垂直")
            changeheight =  int(width * ratio)
            ymin = center[1] - (changeheight/2)
            ymax = ymin + changeheight
        elif (stretch_type == "向下") : 
            # print("test : 向下")
            changeheight =  int(width * ratio)
            ymax = ymin + changeheight
        elif (stretch_type == "向上") :
            # print("test : 向上")
            changeheight = int( width * ratio)
            ymin =ymax - changeheight
        elif (stretch_type == "向左") :
            # print("test : 向左")
            changewidth = int(height * (1/ratio))
            xmin =xmax - changewidth
        elif (stretch_type == "向右") :
            # print("test : 向右")
            changewidth = int(height * (1/ratio))
            xmax = xmin + changewidth
        else :
            print("stretch_type not defined.")
        return (xmin, ymin, xmax, ymax)
    
    def get_boxes_coordinate(self, bounding_boxes, ratiow, ratioh, padh, padw ) :
        scaled_xyxy_boxes, scaled_xywh_boxes = np.copy(bounding_boxes), np.copy(bounding_boxes)
        unscaled_xyxy_boxes, unscaled_xywh_boxes = np.copy(bounding_boxes), np.copy(bounding_boxes)

        if (bounding_boxes != []) :
            bounding_boxes = np.vstack(bounding_boxes)

            # xyxy format
            scaled_xyxy_boxes[:, 0] =  np.clip(bounding_boxes[:, 0], 0, self.model_width)
            scaled_xyxy_boxes[:, 1] =  np.clip(bounding_boxes[:, 1], 0, self.model_height)
            scaled_xyxy_boxes[:, 2] =  np.clip(bounding_boxes[:, 2], 0, self.model_width)
            scaled_xyxy_boxes[:, 3] =  np.clip(bounding_boxes[:, 3], 0, self.model_height)

            unscaled_xyxy_boxes[:, 0] = np.clip((scaled_xyxy_boxes[:, 0] - padw) * ratiow, 0, self.input_width)
            unscaled_xyxy_boxes[:, 1] = np.clip((scaled_xyxy_boxes[:, 1] - padh) * ratioh, 0, self.input_height)
            unscaled_xyxy_boxes[:, 2] = np.clip(scaled_xyxy_boxes[:, 2] * ratiow, 0, self.input_width)
            unscaled_xyxy_boxes[:, 3] = np.clip(scaled_xyxy_boxes[:, 3] * ratioh, 0, self.input_height)

            # xywh format
            bounding_boxes[:, 2:4] = bounding_boxes[:, 2:4] - bounding_boxes[:, 0:2]
            scaled_xywh_boxes[:, 0] = np.clip(bounding_boxes[:, 0], 0, self.model_width)
            scaled_xywh_boxes[:, 1] = np.clip(bounding_boxes[:, 1], 0, self.model_height)
            scaled_xywh_boxes[:, 2] = np.clip(bounding_boxes[:, 2], 0, self.model_width)
            scaled_xywh_boxes[:, 3] = np.clip(bounding_boxes[:, 3], 0, self.model_height)

            unscaled_xywh_boxes[:, 0] = scaled_xywh_boxes[:, 0] * ratiow
            unscaled_xywh_boxes[:, 1] = scaled_xywh_boxes[:, 1] * ratioh
            unscaled_xywh_boxes[:, 2] = scaled_xywh_boxes[:, 2] * ratiow
            unscaled_xywh_boxes[:, 3] = scaled_xywh_boxes[:, 3] * ratioh

            # bounding_boxes[:, 2:4] = bounding_boxes[:, 2:4] - bounding_boxes[:, 0:2]
            # bounding_boxes[:, 0] = (bounding_boxes[:, 0] - padw) * ratiow
            # bounding_boxes[:, 1] = (bounding_boxes[:, 1] - padh) * ratioh
            # bounding_boxes[:, 2] = bounding_boxes[:, 2] * ratiow
            # bounding_boxes[:, 3] = bounding_boxes[:, 3] * ratioh

            # return bounding_boxes
        return scaled_xyxy_boxes, scaled_xywh_boxes, unscaled_xyxy_boxes, unscaled_xywh_boxes

    def get_nms_results(self, bounding_boxes, confidences, class_ids, score, iou):
        results = []
        indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, score, iou) 
        if len(indices) > 0:
            for i in indices:
                try :
                    predicted_class = self.class_names[class_ids[i]]
                except :
                    predicted_class = "unknown"

                bounding_box = self.adjust_boxes_ratio(bounding_boxes[i], self.box_aspect_ratio, self.box_stretch)

                xmin, ymin, xmax, ymax = list(map(int, bounding_box))
                results.append([ymin, xmin, ymax, xmax, predicted_class])
        return results, indices

    def SetDisplayStyle(self, engine) :
        self.style = engine

    def SetDisplayTarget(self, targets) :
        self.priority_target = targets

    def DetectFrame(self, srcimg, frame_resize=None) :
        self.input_height, self.input_width = srcimg.shape[0],  srcimg.shape[1]
        score_thres = float(self.box_score)
        iou_thres = float(self.box_nms_iou)

        if (frame_resize == None) :
            model_size = self.model_width
        else :
            model_size = frame_resize

        image, newh, neww, ratioh, ratiow, padh, padw = self.resize_image_format(srcimg, model_size)
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (newh, neww), swapRB=True, crop=False)
        output_from_network = self.session.run(self.output_names, {self.input_names[0]:  blob})

        bounding_boxes, confidences, class_ids, mask_pred = self._process_box_output(output_from_network[0], score_thres, (32 if self.output_layers_count==2 else None) )

        scaled_xyxy_boxes, scaled_xywh_boxes, unscaled_xyxy_boxes, unscaled_xywh_boxes = self.get_boxes_coordinate( bounding_boxes, ratiow, ratioh, padh, padw)
        
        self.box_points, indices = self.get_nms_results( unscaled_xywh_boxes, confidences, class_ids, score_thres, iou_thres)
        if self.output_layers_count==2 :
            self.mask_maps = self._process_mask_output(output_from_network[1], unscaled_xyxy_boxes, mask_pred, indices)

    def DrawIdentifyOnFrame(self, frame_show, mask_alpha=0.3, detect=True, seg=False) :
        if (self.style != None) :
            frame_show = self.style(frame_show)
        mask_img = frame_show.copy()

        if ( len(self.box_points) != 0 )  :
            for index, box in enumerate(self.box_points):
                ymin, xmin, ymax, xmax, label = box

                if (label in self.priority_target ) :
                    if seg and self.mask_maps!= []:
                        # Draw fill mask image
                        crop_mask = self.mask_maps[index][ymin:ymax, xmin:xmax, np.newaxis]
                        crop_mask_img = mask_img[ymin:ymax, xmin:xmax]
                        crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * hex_to_rgb(self.colors_dict[label])
                        mask_img[ymin:ymax, xmin:xmax] = crop_mask_img

                    if detect :
                        cv2.putText(frame_show, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_rgb(self.colors_dict[label]), 2)
                        cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax),  hex_to_rgb(self.colors_dict[label]), 2)
        return cv2.addWeighted(mask_img, mask_alpha, frame_show, 1 - mask_alpha, 0) 
      
    def DrawIdentifyOverlayOnFrame(self, frame_overlap, frame_show, detect=True, seg=False) :
        mask_img = frame_show.copy()
        mask_binaray = np.zeros(( int(frame_show.shape[0]), int(frame_show.shape[1]),3 ), np.uint8)
        if ( len(self.box_points) != 0 )  :
            for index, box in enumerate(self.box_points):
                snap = np.zeros(( int(frame_show.shape[0]), int(frame_show.shape[1]), 3 ), np.uint8)
                ymin, xmin, ymax, xmax, label = box
                if (label in self.priority_target ) :

                    mask_box_img = frame_overlap[ymin:ymax, xmin:xmax]
                    if (self.style != None) :
                        mask_box_img = self.style(mask_box_img)

                    if (detect) :    
                        frame_show[ymin:ymax, xmin:xmax] = mask_box_img
                    else :
                        frame_overlap[ymin:ymax, xmin:xmax] = mask_box_img

                    # Draw fill mask image
                    if (seg and self.mask_maps!= []):
                        crop_mask = self.mask_maps[index][ymin:ymax, xmin:xmax, np.newaxis]
                        crop_mask_img = mask_binaray[ymin:ymax, xmin:xmax]

                        snap[ymin:ymax, xmin:xmax] = crop_mask_img * (1 - crop_mask) + crop_mask
                        mask_binaray += snap
            # cv2.imshow("test :", frame_show)
            # cv2.waitKey(33)
            if (seg and self.mask_maps!= []):
                mask_img[mask_binaray[:,:,0] >= 1] = [0, 0, 0]
                frame_overlap[mask_binaray[:,:,0] < 1] = [0, 0, 0]
                frame_show = mask_img + frame_overlap
        return frame_show
