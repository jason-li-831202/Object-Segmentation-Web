from __future__ import annotations

import math
import os, cv2
import random
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

from .AnimeGAN import AnimeGAN
from .utils import OnnxBaseEngine, convert_3channel_add_alpha, hex_to_rgba

def sigmoid(x):
	return np.exp(x) / (1 +  np.exp(x)) # equal to 1 / (1 + np.exp(-x))

@dataclass
class ObjectInfo:
	x: float
	y: float
	width: float
	height: float
	label: str
	conf: float
	mask_map: Optional[np.ndarray]

	def tolist(self, format_type: str = "xyxy"):
		if (format_type == "xyxy"):
			temp = [self.x, self.y, self.x + self.width, self.y + self.height]
		else :
			temp = [self.x, self.y, self.width, self.height]
		return temp

	@property
	def crop_mask(self) -> Optional[np.ndarray]:
		if not isinstance(self.mask_map, np.ndarray): return None
		return self.mask_map[self.y:self.y + self.height, self.x:self.x + self.width, np.newaxis]

	@property
	def polygon_mask(self) -> Optional[np.ndarray]:
		if not isinstance(self.mask_map, np.ndarray): return None
		_, binary_mask = cv2.threshold(self.mask_map, 1, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		return contours

	def pad(self, padding: int) -> ObjectInfo:
		return ObjectInfo(
			x=self.x - padding,
			y=self.y - padding,
			width=self.width + 2 * padding,
			height=self.height + 2 * padding,
			conf=self.conf,
			label=self.label,
			mask_map=self.mask_map)
	
	
class ObjectOnnxDetector(OnnxBaseEngine):
	_defaults = {
		"model_path": None,
		"classes_path" : None,
		"box_aspect_ratio" : None,
		"box_stretch" : None,
		"box_score" : None,
		"box_nms_iou" : None,
	}

	def __init__(self, **kwargs):
		OnnxBaseEngine.__init__(self)
		self.__dict__.update(kwargs) # and update with user overrides
		print("Detector Inference Version : ", self.providers)

		self._object_info = []

		self.style = None
		self.keep_ratio = True
		self._get_class(self.classes_path)
		self._get_model_details()

	def _get_class(self, classes_path : str) -> None:
		classes_path = os.path.expanduser(classes_path)
		if (os.path.isfile(classes_path) is False):
			raise Exception("%s is not exist." % classes_path)
		
		with open(classes_path) as f:
			class_names = f.readlines()
		self.class_names = [c.strip() for c in class_names]
		self.display_target = self.class_names
		self.display_target.append("unknown")

		get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))
		self.colors_dict["unknown"] = '#ffffff'

	def _get_model_details(self):
		self.model_in_shapes, self.model_in_names = self.get_engine_input_shape()
		self.model_out_shapes, self.model_out_names = self.get_engine_output_shape()

		if len(self.model_out_shapes) ==2 :
			# (N, 80[obj_conf] + 4[bbox] + 32[mask], -1), (N, 32[mask_num], mask_h, mask_w)
			self.box_out_shape, self.mask_out_shape = self.model_out_shapes 
		else :
			# (N, 80[obj_conf] + 4[bbox], -1), ()
			self.box_out_shape, self.mask_out_shape = self.model_out_shapes, []
		self.model_height, self.model_width = self.model_in_shapes[0][2], self.model_in_shapes[0][3]

	def _prepare_input(self, srcimg : cv2) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[int, int]] :
		self.input_height, self.input_width = srcimg.shape[0],  srcimg.shape[1]

		image, real_shape, src_shape, pad_shape = self.scaler_image(srcimg, (self.model_height, self.model_width), self.keep_ratio)
		blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.model_width, self.model_height), 
										swapRB=True, crop=False).astype(self.engine_dtype)

		return blob, real_shape, src_shape, pad_shape

	def _process_output(self, outputs: np.ndarray) -> Tuple[List[np.ndarray], List[str], List[Optional[np.ndarray]]]:
		score_thres = float(self.box_score)
		iou_thres = float(self.box_nms_iou)
		if len(self.model_out_shapes) ==2 :
			crood_outputs, mask_outputs = outputs
		else :
			crood_outputs, mask_outputs = outputs, []
		
		# box output (x, y, x, y)
		_raw_boxes, _raw_class_confs, _raw_class_ids, _raw_mask_confs = self.__process_box_output(crood_outputs, score_thres, 
																			(self.mask_out_shape[1] if self.mask_out_shape else None) )
		_raw_masks = self.__process_mask_output(mask_outputs, _raw_mask_confs)

		return self.__process_nms(_raw_boxes, _raw_class_confs, _raw_class_ids, 
									_raw_masks, score_thres, iou_thres)
	
	def __process_box_output(self, box_output : np.ndarray, conf_thres : int, num_masks : int =32) -> Tuple[list, list, list, list]:
		_raw_boxes = []
		_raw_class_ids = []
		_raw_class_confs = []
		_raw_mask_confs = []

		if (num_masks != None) :
			num_classes = self.box_out_shape[1] - num_masks - 4

		box_output = np.squeeze(box_output).T
		for predictions in box_output:
			if (num_masks != None) :
				obj_cls_probs = predictions[4:4+num_classes]
			else :
				obj_cls_probs = predictions[4:]

			classId = np.argmax(obj_cls_probs)
			classConf = float(obj_cls_probs[classId])
			if classConf > conf_thres :
				if (num_masks != None) :
					_raw_mask_confs.append(predictions[num_classes+4:])
				x, y, w, h = predictions[0:4]
				_raw_class_ids.append(classId)
				_raw_class_confs.append(classConf)
				_raw_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1)) #x1, y1, x2, y2

		return _raw_boxes, _raw_class_confs, _raw_class_ids, _raw_mask_confs

	def __process_mask_output(self, mask_output : np.ndarray, mask_preds: list) -> Union[np.ndarray, list]:
		if mask_preds == []:
			return []
		mask_output = np.squeeze(mask_output)

		# Calculate the mask maps for each box
		_, num_mask, mask_height, mask_width = self.mask_out_shape
		_raw_mask = sigmoid(mask_preds @ mask_output.reshape((num_mask, -1)))

		_raw_mask = _raw_mask.reshape((-1, mask_height, mask_width))
		return _raw_mask

	def __process_nms(self, raw_boxes, raw_confs, raw_ids, raw_masks, score: float, iou: float) -> Tuple[List[np.ndarray], List[str], List[Optional[np.ndarray]]]:
		if (raw_masks != []):
			# Downscale the boxes to match the mask size
			_, num_mask, mask_height, mask_width = self.mask_out_shape
			pre_boxes, _, post_boxes,_ = self.scaler_coord(raw_boxes, 
															(self.model_height, self.model_width), 
															(mask_height, mask_width), 
															(0,0))
		else :
			pre_boxes, _, _, _ = self.scaler_coord(raw_boxes, 
													(self.model_height, self.model_width))

		indices = cv2.dnn.NMSBoxes(pre_boxes, raw_confs, score, iou) 
		model_coords = []
		model_classes = []
		model_mask_maps = [] if (raw_masks == []) else np.zeros((len(indices), self.model_height, self.model_width))
		if len(indices) > 0:
			for idx, indice in enumerate(indices):
				# Get class names 
				try :
					pred_class = self.class_names[raw_ids[indice]]
				except :
					pred_class = "unknown"
				pred_conf = raw_confs[indice]
				model_classes.append((pred_class, pred_conf))

				# Get coords
				pred_box = self.__adjust_boxes_ratio(pre_boxes[indice], self.box_aspect_ratio, self.box_stretch)
				xmin, ymin, xmax, ymax = pred_box
				model_coords.append(np.stack([xmin, ymin, xmax, ymax], axis=-1))
				
				# Get masks (For every box/mask pair, get the mask map)
				if (raw_masks != []):
					mask_xmin, mask_ymin, mask_xmax, mask_ymax = post_boxes[indice]
					pred_crop_mask = raw_masks[indice][mask_ymin:mask_ymax, mask_xmin:mask_xmax].astype(np.float32)
					model_crop_mask = cv2.resize(pred_crop_mask,
												(xmax - xmin, ymax - ymin),
												interpolation=cv2.INTER_CUBIC)
					model_crop_mask = (model_crop_mask > 0.5).astype(np.uint8)
					model_mask_maps[idx, ymin:ymax, xmin:xmax] = model_crop_mask
				else :
					model_mask_maps.append(None)
		return model_coords, model_classes, model_mask_maps
	
	@staticmethod
	def __adjust_boxes_ratio(bounding_box, ratio, stretch_type) :
		""" Adjust the aspect ratio of the box according to the orientation """
		xmin, ymin, xmax, ymax = bounding_box 
		width = int(xmax - xmin)
		height = int(ymax - ymin)

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
	
	@staticmethod
	def scaler_image(srcimg: np.ndarray, output_size: Tuple, keep_ratio: bool = False) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
		oldh, oldw = srcimg.shape[0], srcimg.shape[1]
		padh, padw, newh, neww = 0, 0, *output_size # (h, w)
		if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
			hw_scale = srcimg.shape[0] / srcimg.shape[1]
			if hw_scale > 1:
				newh, neww = output_size[0], int(output_size[0] / hw_scale)
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padw = int((output_size[1] - neww) * 0.5)
				img = cv2.copyMakeBorder(img, 0, 0, padw, output_size[1] - neww - padw, cv2.BORDER_CONSTANT,
										 value=0)  # add border
			else:
				newh, neww = int(output_size[1] * hw_scale), output_size[1]
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padh = int((output_size[0] - newh) * 0.5)
				img = cv2.copyMakeBorder(img, padh, output_size[0]  - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
		else:
			img = cv2.resize(srcimg, (output_size[0] , output_size[0] ), interpolation=cv2.INTER_CUBIC)
		return img, (newh, neww), (oldh, oldw), (padh, padw)

	@staticmethod
	def scaler_coord(bboxes, pre_shape: tuple = (0, 0), post_shape: tuple = (0, 0),  pad_shape:tuple = (0, 0) ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
		pre_xyxy_boxes, pre_xywh_boxes, post_xyxy_boxes, post_xywh_boxes = [np.copy([])]*4
		if (bboxes != []) :
			newh, neww = post_shape
			oldh, oldw = pre_shape
			padh, padw = pad_shape
			ratioh, ratiow = newh / oldh,  neww / oldw
			bboxes = np.vstack(bboxes)

			# xyxy format
			pre_xyxy_boxes = np.copy(bboxes)
			pre_xyxy_boxes[:, 0] =  np.clip(bboxes[:, 0], 0, oldw + 2*padw)
			pre_xyxy_boxes[:, 1] =  np.clip(bboxes[:, 1], 0, oldh + 2*padh)
			pre_xyxy_boxes[:, 2] =  np.clip(bboxes[:, 2], 0, oldw + 2*padw)
			pre_xyxy_boxes[:, 3] =  np.clip(bboxes[:, 3], 0, oldh + 2*padh)

			post_xyxy_boxes = np.copy(bboxes)
			post_xyxy_boxes[:, 0] = np.clip((bboxes[:, 0] - padw) * ratiow, 0, neww)
			post_xyxy_boxes[:, 1] = np.clip((bboxes[:, 1] - padh) * ratioh, 0, newh)
			post_xyxy_boxes[:, 2] = np.clip((bboxes[:, 2] - padw) * ratiow, 0, neww)
			post_xyxy_boxes[:, 3] = np.clip((bboxes[:, 3] - padh) * ratioh, 0, newh)

			# xywh format
			pre_xywh_boxes = np.copy(pre_xyxy_boxes)
			pre_xywh_boxes[:, 2:4] = pre_xywh_boxes[:, 2:4] - pre_xywh_boxes[:, 0:2]

			post_xywh_boxes = np.copy(post_xyxy_boxes)
			post_xywh_boxes[:, 2:4] = post_xywh_boxes[:, 2:4] - post_xywh_boxes[:, 0:2]
		
		return pre_xyxy_boxes.astype(int), pre_xywh_boxes.astype(int), post_xyxy_boxes.astype(int), post_xywh_boxes.astype(int)

	@staticmethod
	def scaler_mask(mask_maps, post_shape: tuple, pad_shape:tuple = (0, 0) ):
		post_mask_maps = mask_maps.copy()
		if ( isinstance(mask_maps, np.ndarray) ) :
			newh, neww = post_shape
			padh, padw = pad_shape

			post_mask_maps = np.zeros((mask_maps.shape[0], newh, neww))
			for idx, mask_map in enumerate(mask_maps):
				post_mask_maps[idx] = cv2.resize(mask_map[padh:-padh-1, padw:-padw-1],
												(neww, newh), 
												interpolation=cv2.INTER_CUBIC)
		return post_mask_maps
	
	def SetDisplayStyle(self, engine: AnimeGAN) -> None:
		self.style = engine

	def SetDisplayTarget(self, targets : list) -> None:
		self.display_target = targets

	def DetectFrame(self, srcimg : cv2) -> None:
		input_tensor, real_shape, src_shape, pad_shape = self._prepare_input(srcimg)

		output_from_network = self.engine_inference(input_tensor)

		box_info, label_info, mask_maps_info = self._process_output(output_from_network)

		box_info = self.scaler_coord(box_info, real_shape, src_shape, pad_shape)[3] 
		mask_maps_info = self.scaler_mask(mask_maps_info, src_shape, pad_shape)

		self._object_info = []
		for bbox, label, mask in zip(box_info, label_info, mask_maps_info):
			self._object_info.append(ObjectInfo(*bbox, label=label[0],
									  				  conf=label[1],
													  mask_map=mask))

	def DrawIdentifyOnFrame(self, frame_show : cv2, mask_alpha : float = 0.3, detect : bool = True, seg : bool = False) -> cv2:
		if (self.style != None) :
			frame_show = self.style(frame_show)
		mask_img = frame_show.copy()

		if ( len(self._object_info) != 0 )  :
			for index, _info in enumerate(self._object_info):
				xmin, ymin, xmax, ymax = _info.tolist()
				label = _info.label
				mask_map = _info.mask_map

				if (label in self.display_target ) :

					if detect :
						cv2.putText(frame_show, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_rgba(self.colors_dict[label]), 2)
						cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax),  hex_to_rgba(self.colors_dict[label]), 2)

					if seg and isinstance(mask_map, np.ndarray):
						# Draw fill mask image
						crop_mask = _info.crop_mask 
						crop_mask_img = mask_img[ymin:ymax, xmin:xmax]
						crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * hex_to_rgba(self.colors_dict[label])
						mask_img[ymin:ymax, xmin:xmax] = crop_mask_img
						cv2.drawContours(frame_show, _info.polygon_mask, -1, hex_to_rgba(self.colors_dict[label]), 1, lineType=cv2.LINE_AA) # cv2.FILLED

		return cv2.addWeighted(mask_img, mask_alpha, frame_show, 1 - mask_alpha, 0) 
	  
	def DrawIdentifyOverlayOnFrame(self, frame_overlap : cv2, frame_show : cv2, detect : bool = True, seg : bool = False) -> cv2:
		assert detect!=seg, "both mode can't together use it."
		if (self.style != None) :
			frame_overlap = self.style(frame_overlap)
		frame_show = cv2.resize(frame_show, (self.input_width, self.input_height))
		mask_img = frame_show.copy()
		mask_status = False
		mask_binaray = np.zeros(( int(frame_show.shape[0]), int(frame_show.shape[1]), 3 ), np.uint8)
		mask_binaray = convert_3channel_add_alpha(mask_binaray, alpha=255)

		if ( len(self._object_info) != 0 )  :
			for index, _info in enumerate(self._object_info):
				xmin, ymin, xmax, ymax = _info.tolist()
				label = _info.label
				mask_map = _info.mask_map

				snap = np.zeros(( int(frame_show.shape[0]), int(frame_show.shape[1]), 4 ), np.uint8)
				if (label in self.display_target ) :
					
					if (detect) :    
						frame_show[ymin:ymax, xmin:xmax] = frame_overlap[ymin:ymax, xmin:xmax]

					# Draw fill mask image
					if (seg and isinstance(mask_map, np.ndarray)):
						mask_status = True
						crop_mask = _info.crop_mask 
						crop_mask_img = mask_binaray[ymin:ymax, xmin:xmax]

						snap[ymin:ymax, xmin:xmax] = crop_mask_img * (1 - crop_mask) + crop_mask
						mask_binaray += snap

			if (seg and mask_status):
				mask_img[mask_binaray[:,:,0] >= 1] = [0, 0, 0, 0]
				frame_overlap[mask_binaray[:,:,0] < 1] = [0, 0, 0, 0]
				frame_show = mask_img + frame_overlap
		return frame_show
	