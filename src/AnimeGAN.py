import os
import cv2
import typing
import numpy as np
from .utils import OnnxBaseEngine

class AnimeGAN(OnnxBaseEngine):
    """ Object to image animation using AnimeGAN models
    https://github.com/TachibanaYoshino/AnimeGANv2

    onnx models:
    'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe' AnimeGAN_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1' AnimeGANv2_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK' AnimeGANv2_Shinkai.onnx
    'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI' AnimeGANv2_Paprika.onnx

    """
    _defaults = {
        "model_path": None,
        "downsize_ratio" : None,
    }

    def __init__( self, model_path: str = '', downsize_ratio: float = 1.0, ) -> None:
        """
        Args:
            model_path: (str) - path to onnx model file
            downsize_ratio: (float) - ratio to downsize input frame for faster inference
        """
        OnnxBaseEngine.__init__(self, model_path)
        print("AnimeGAN Inference Version : ", self.providers)

        self.input_shape, self.input_names = self.get_engine_input_shape()
        self.output_shape, self.output_names = self.get_engine_output_shape()
        self.downsize_ratio = downsize_ratio

    def __to_32s(self, x):
        return 256 if x < 256 else x - x%32

    def _prepare_input(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        """ Function to process frame to fit model input as 32 multiplier and resize to fit model input

        Args:
            frame: (np.ndarray) - frame to process
            x32: (bool) - if True, resize frame to 32 multiplier

        Returns:
            frame: (np.ndarray) - processed frame
        """
        b_channel, g_channel, r_channel, alpha_channel  = cv2.split(frame)
        frame_bgr = cv2.merge((b_channel, g_channel, r_channel))

        h, w = frame_bgr.shape[:2]
        if x32: # resize image to multiple of 32s
            frame_bgr = cv2.resize(frame_bgr, (self.__to_32s(int(w*self.downsize_ratio)), self.__to_32s(int(h*self.downsize_ratio))))
        bgr_channels = np.expand_dims(frame_bgr.astype(self.engine_dtype) / 127.5 - 1.0, axis=0)

        return bgr_channels, alpha_channel

    def _process_output(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        """ Convert model float output to uint8 image resized to original frame size

        Args:
            frame: (np.ndarray) - AnimeGaAN output frame
            wh: (typing.Tuple[int, int]) - original frame size

        Returns:
            frame: (np.ndarray) - original size animated image
        """
        if ( "Quant_output" in self.session._outputs_meta[0].name) :
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
        if (self.session == None) : 
            self.init_engine()

        bgr_channel, alpha_channel = self._prepare_input(frame)

        outputs = self.engine_inference(bgr_channel)

        bgr_channel = self._process_output(outputs[0], frame.shape[:2][::-1])

        return cv2.merge((bgr_channel, alpha_channel))
    
