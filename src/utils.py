import abc, os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional

def convert_3channel_add_alpha(image, alpha=255):
	b_channel, g_channel, r_channel = cv2.split(image)
	alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha  # alpha通道每个像素点区间为[0,255], 0为完全透明，255是完全不透明
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def hex_to_rgba(value, alpha=255):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + (alpha,)

class OnnxBaseEngine(abc.ABC):
	_defaults = {
		"model_path": None,
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
		
	def __init__(self, model_path: Optional[str] = None) -> None:
		self.__dict__.update(self._defaults) # set up default values
		if (model_path != None):
			self.model_path = model_path
		if not os.path.isfile(self.model_path):
			raise Exception("The model path [%s] can't not found!" % self.model_path)
		assert self.model_path.endswith('.onnx'), 'Onnx Parameters must be a .onnx file.'
		self.init_engine()
		self.__load_engine_interface()

	def __load_engine_interface(self):
		self.__input_shape = [input.shape for input in self.session.get_inputs()]
		self.__input_names = [input.name for input in self.session.get_inputs()]
		self.__output_shape = [output.shape for output in self.session.get_outputs()]
		self.__output_names = [output.name for output in self.session.get_outputs()]
		print(f"-> Engine Type  : {self.engine_dtype}")
		print(f"-> Input Shape  : {self.__input_shape}")
		print(f"-> Output Shape : {self.__output_shape}")

	def init_engine(self):
		if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
			self.providers = 'CUDAExecutionProvider'
		else :
			self.providers = 'CPUExecutionProvider'
		self.session = ort.InferenceSession(self.model_path, providers = [self.providers] )
		self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32

	def clear_engine(self):
		del self.session
		self.session = None

	def get_engine_input_shape(self):
		return self.__input_shape, self.__input_names

	def get_engine_output_shape(self):
		return self.__output_shape, self.__output_names
	
	def engine_inference(self, input_tensor):
		output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
		return output

	@abc.abstractmethod
	def _prepare_input(self):
		return NotImplemented
	
	@abc.abstractmethod
	def _process_output(self):
		return NotImplemented