import time
import cv2, os
import numpy as np
import pafy
from PIL import Image, ImageSequence
from enum import Enum

from src.ObjectDetector import ObjectOnnxDetector
from src.AnimeGAN import AnimeGAN

class DisplayType(Enum):
	NONE = "None"
	BASIC_MODE = "Basic Mode"
	DETECT_MODE = "Detect Mode"
	SEMANTIC_MODE = "Semantic Mode"

class imageCapture() :
    def __init__(self, filename) -> None:
        self.img  = cv2.imread(filename)
        self.type = "image"
        if self.img is None :
            gif = Image.open(filename)
            self.img = []                                      
            self.gif_index = 0
            for frame in ImageSequence.Iterator(gif):
                red, green, blue, alpha = np.array(frame.convert('RGBA')  ) .T
                data = np.array([blue, green, red])        
                self.img.append(data.transpose())         
            self.type = "gif"

    def read(self):
        if (self.type == "gif") :
            current_index = self.gif_index
            self.gif_index += 1
            if (self.gif_index == len(self.img)) :
                self.gif_index = 0
            return True, np.array(self.img[current_index])
        else :
            if self.img is None :
                return False, self.img
            else :
                return True, self.img
                 
class VideoStreaming(object):
    def __init__(self, cam_config=None, model_config=None):
        super(VideoStreaming, self).__init__()
        self.cam_config = cam_config
        self.cam_config['blur'] = 0
        if (cam_config == None) :
            raise Exception("cam_config setting is %s." % cam_config)

        if (model_config == None) :
            raise Exception("model_config setting is %s." % model_config)
        
        self.CAM = cv2.VideoCapture(self.cam_config['cam_id'])
        if (not self.CAM.isOpened()) :
            raise Exception("video root [%s] is error. please check it." % self.cam_config['cam_id'])
        self.BG = None
        self._exposure = None
        self._contrast = None
        self._blur = None
        self.style_dict = {"None" : None}
        self.InitCamSettings()

        ObjectOnnxDetector.set_defaults(model_config)
        self.MODEL = ObjectOnnxDetector()

        style_path = [_ for _ in os.listdir("./models/Style") if _.endswith(".onnx")]
        for type in style_path :
            self.style_dict.update({os.path.splitext(type)[0]: AnimeGAN('./models/Style/'+  type)})
        self._preview = True
        self._flipH = False
        self._detect = DisplayType.SEMANTIC_MODE
        self._background = False

    def InitCamSettings(self) :
        print('*'*28)
        print('* Init model settings *')
        print('*'*28)
        self.CAM.set(cv2.CAP_PROP_EXPOSURE, self.cam_config['exposure'])
        self._exposure = self.cam_config['exposure']
        self.CAM.set(cv2.CAP_PROP_CONTRAST, self.cam_config['contrast'])
        self._contrast = self.cam_config['contrast']

        blur_range = range(1, 203, 2)
        self.blur_dict = {i:blur_range[i]  for i in range(len(blur_range))}
        self._blur = self.blur_dict[self.cam_config['blur']]
        self.H = self.CAM.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.W = self.CAM.get(cv2.CAP_PROP_FRAME_WIDTH)
        for key in self.cam_config:
            print ('  ', key,'=', self.cam_config[key])
        return True
    
    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = eval(value)

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        self._background = eval(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = eval(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        for type in list(DisplayType) :
            if (type.value == value) :
                self._detect = type

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.CAM.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.CAM.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    @property
    def blur(self):
        return self._blur

    @blur.setter
    def blur(self, value):
        self._blur = self.blur_dict[value]

    def setBackGround(self, url):
        start = time.time()

        root, extension = os.path.splitext(url)
        if extension in {".mp4", ".m4v", ".mkv", ".mov", ".avi", ".wmv", ".flv" }:
            if os.path.isfile(url):
                self.BG = cv2.VideoCapture(url)
            else :
                print("File don't exist.")
        elif extension in {".png", ".PNG", ".jpg", ".jpeg", ".gif", ".bmp", ".tif" }:
            if os.path.isfile(url):
                self.BG = imageCapture(url)
            else :
                print("File don't exist.")
        else :
            try :
                video = pafy.new(url)
                best = video.getbest(preftype="mp4")
                self.BG = cv2.VideoCapture(best.url)
            except OSError :
                print("You need to revise [your path]\site-packages\youtube_dl\extractor\youtube.py : ['uploader_id': self._search_regex(r'/(?:channel|user)/([^/?&#]+)', owner_profile_url, 'uploader id') if owner_profile_url else None,] to \
                    ['uploader_id': self._search_regex(r'/(?:channel/|user/|@)([^/?&#]+)', owner_profile_url, 'uploader id', default=None),] ")
            except KeyError:
                print("You need to comment out [your path]\site-packages\pafy\backend_youtube_dl.py : [self._likes = self._ydl_info['like_count'] and [self._dislikes = self._ydl_info['dislike_count']")
            except :
                print("Invalid URL escape while parsing URL")

        end = time.time()
        print("loading background time (sec) : ", round(end-start, 2) )

    def setViewTarget(self, targets):
        self.MODEL.SetDisplayTarget(targets)

    def setViewStyle(self, style_name) :
        self.MODEL.SetDisplayStyle( self.style_dict[style_name])

    def show(self):
        while(self.CAM.isOpened()):
            ret, snap = self.CAM.read()
            if (self._background and self.BG != None):
                gt_ret, snap_bg = self.BG.read()
                if gt_ret == True:
                    snap_bg = cv2.resize(snap_bg, (int(self.W), int(self.H)))
                    snap_bg = cv2.GaussianBlur(snap_bg,(self._blur, self._blur), cv2.BORDER_DEFAULT)
                else :
                    snap_bg = np.zeros((
                        int(self.CAM.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.CAM.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                        3
                    ), np.uint8)
            else :
                snap_bg = np.zeros((
                    int(self.CAM.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self.CAM.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    3
                ), np.uint8)

            if self.flipH:
                snap = cv2.flip(snap, 1)

            if ret == True:
                if self._preview:
                    if self.detect != DisplayType.NONE:
                        self.MODEL.DetectFrame(snap)

                        if self.detect == DisplayType.BASIC_MODE :
                            snap = self.MODEL.DrawIdentifyOnFrame(snap, detect=True, seg=True)
                        elif self.detect == DisplayType.DETECT_MODE :
                            snap = self.MODEL.DrawIdentifyOverlayOnFrame(snap, snap_bg, detect=True)
                        elif self.detect == DisplayType.SEMANTIC_MODE :
                            snap = self.MODEL.DrawIdentifyOverlayOnFrame(snap, snap_bg, detect=False, seg=True)
                else:
                    snap = np.zeros((
                        int(self.CAM.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.CAM.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = 'camera disabled'
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_COMPLEX
                    color = (255,255,255)
                    cv2.putText(snap, label, (W//2, H//2), font, 2, color, 2)

                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print('Out Display Loop!')
