import os
import cv2
import typing
import numpy as np
import onnxruntime as ort
import time

class AnimeGAN:
    """ Object to image animation using AnimeGAN models
    https://github.com/TachibanaYoshino/AnimeGANv2

    onnx models:
    'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe' AnimeGAN_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1' AnimeGANv2_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK' AnimeGANv2_Shinkai.onnx
    'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI' AnimeGANv2_Paprika.onnx

    """
    def __init__( self, model_path: str = '', downsize_ratio: float = 1.0, ) -> None:
        """
        Args:
            model_path: (str) - path to onnx model file
            downsize_ratio: (float) - ratio to downsize input frame for faster inference
        """
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exists in {model_path}")
        self.model_path = model_path
        self.downsize_ratio = downsize_ratio
        self.providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        
        self.ort_sess = None
        # self.ort_sess = ort.InferenceSession(self.model_path, providers=self.providers)

    def __to_32s(self, x):
        return 256 if x < 256 else x - x%32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        """ Function to process frame to fit model input as 32 multiplier and resize to fit model input

        Args:
            frame: (np.ndarray) - frame to process
            x32: (bool) - if True, resize frame to 32 multiplier

        Returns:
            frame: (np.ndarray) - processed frame
        """
        h, w = frame.shape[:2]
        if x32: # resize image to multiple of 32s
            frame = cv2.resize(frame, (self.__to_32s(int(w*self.downsize_ratio)), self.__to_32s(int(h*self.downsize_ratio))))
        frame = frame.astype(self.input_types) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        """ Convert model float output to uint8 image resized to original frame size

        Args:
            frame: (np.ndarray) - AnimeGaAN output frame
            wh: (typing.Tuple[int, int]) - original frame size

        Returns:
            frame: (np.ndarray) - original size animated image
        """
        if ( "Quant_output" in self.ort_sess._outputs_meta[0].name) :
            frame = np.tanh(frame).transpose((0,2,3,1)) 
        frame = frame.astype(np.float32)
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to process selfie semgentation on each call

        Args:
            frame: (np.ndarray) - frame to excecute face detection on

        Returns:
            frame: (np.ndarray) - processed frame with face detection
        """
        if (self.ort_sess == None) : 
            self.ort_sess = ort.InferenceSession(self.model_path, providers=self.providers)
            self.input_types = np.float16 if 'float16' in self.ort_sess.get_inputs()[0].type else np.float32

        b_channel, g_channel, r_channel, alpha_channel  = cv2.split(frame)
        frame_bgr = cv2.merge((b_channel, g_channel, r_channel))

        image = self.process_frame(frame_bgr)
        outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})
        frame_bgr = self.post_process(outputs[0], frame_bgr.shape[:2][::-1])

        return cv2.merge((frame_bgr, alpha_channel ))
    
    def unload(self):
        del self.ort_sess
        self.ort_sess = None