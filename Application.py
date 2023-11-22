import os
import socket   
from flask import Flask, render_template, request, Response, jsonify
from flask_bootstrap import Bootstrap
from gevent import monkey
from gevent.pywsgi import WSGIServer
monkey.patch_all()

from src.VideoStream import *

model_config = {
    "model_path": 'models/yolov8n-seg-coco.onnx', # model path
    "classes_path" : 'models/coco_label.txt', # classes path
    "box_score" : 0.4,
    "box_nms_iou" : 0.45,
    "box_aspect_ratio" : None,
    "box_stretch" : None,
}

cam_config = {
    "cam_id" : 0,
    'exposure': -2, # init cam exposure
    'contrast': 50 # init cam contrast
}
  
VIDEO = VideoStreaming(cam_config=cam_config, model_config=model_config)
application = Flask(__name__, 
                    static_folder='./src/templates/static', 
                    template_folder='./src/templates')
Bootstrap(application)

@application.route('/')
def home():
    TITLE = 'Object Segmentation App'
    CAM_CONFIG = cam_config.copy()
    CAM_CONFIG["sensor_h"] = int(VIDEO.sensorH)
    CAM_CONFIG["sensor_w"] = int(VIDEO.sensorW)
    CAM_CONFIG["display_h"] = int(VIDEO.displayH)
    CAM_CONFIG["display_w"] = int(VIDEO.displayW)

    MODLE_CONFIG = model_config.copy()
    for key, value in model_config.items():
        if type(value) == str:
            MODLE_CONFIG[key] = os.path.basename(value)

    CLASSES_CONFIG = VIDEO.MODEL.colors_dict.copy()
    STYLE_CONFIG = VIDEO.style_dict.copy()
    return render_template('index.html', TITLE=TITLE, 
                                        CAM_CONFIG = CAM_CONFIG, 
                                        MODEL_CONFIG = MODLE_CONFIG, 
                                        TARGETLIST = CLASSES_CONFIG,
                                        STYLELIST = STYLE_CONFIG)

@application.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO.show(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@application.route('/request_target_display')
def request_target_display():
    targets_list  = request.args.get('targetList')
    print('*'*10)
    print("display targets :", targets_list)
    print('*'*10)
    VIDEO.setViewTarget(targets_list) 

    return "nothing"

# Button requests called from ajax
@application.route('/request_preview_switch')
def request_preview_switch():
    active  = request.args.get('active')
    VIDEO.preview = active
    print('*'*10)
    print("display preview :", VIDEO.preview)
    print('*'*10)
    return "nothing"

@application.route('/request_background_video')
def request_background_video():
    # url = "https://youtu.be/LtrtLL_8mLM" # testing url
    url  = request.args.get('url')
    print('*'*10)
    print("video url or path :", url)
    print('*'*10)
    VIDEO.setBackGround(url)
    return "nothing"

@application.route('/request_background_switch')
def request_background_switch():
    active  = request.args.get('active')
    VIDEO.background = active
    print('*'*10, VIDEO.background)
    return "nothing"

@application.route('/request_flipH_switch')
def request_flipH_switch():
    active  = request.args.get('active')
    VIDEO.flipH = active
    print('*'*10)
    print("display flip :",  VIDEO.flipH)
    print('*'*10)
    return "nothing"

@application.route('/request_model_switch')
def request_model_switch():
    type  = request.args.get('type')
    VIDEO.detect = type
    print('*'*10)
    print("display type :",  type)
    print('*'*10)
    return "nothing"

@application.route('/request_style_switch')
def request_style_switch():
    type  = request.args.get('type')
    VIDEO.setViewStyle(type)
    print('*'*10)
    print("display style :",  type)
    print('*'*10)
    return "nothing"

@application.route('/request_exposure')
def request_exposure():
    value  = request.args.get('value')
    VIDEO.exposure = value
    print('*'*10)
    print("display exposure :", VIDEO.exposure)
    print('*'*10)
    return "nothing"


@application.route('/request_contrast')
def request_contrast():
    value  = request.args.get('value')
    VIDEO.contrast = value
    print('*'*10)
    print("display contrast :",VIDEO.contrast)
    print('*'*10)
    return "nothing"

@application.route('/request_blur')
def request_blur():
    value  = request.args.get('value')
    VIDEO.blur = value
    print('*'*10)
    print("display blur (kernel):",VIDEO.blur)
    print('*'*10)
    return "nothing"

@application.route('/reset_camera')
def reset_camera():
    STATUS =VIDEO.initCamSettings()
    active  = request.args.get('active')
    VIDEO.flipH = active
    type  = request.args.get('type')
    VIDEO.detect = type
    print('*'*10)
    print("reset :",STATUS)
    print('*'*10)
    return "nothing"


if __name__ == "__main__":
    hostname=socket.gethostname()   
    IPAddr=socket.gethostbyname(hostname)   
    print("Your Computer Name is:"+hostname)   
    print("Your Computer IP Address is:"+IPAddr)  

    http_server = WSGIServer(('0.0.0.0', 8080), application)
    print("==============================================================")
    print("The server will be accessible at [ http://"+ IPAddr +":8080 ].")
    print("Use Docker will be accessible at [ http://localhost:8080 ]")
    print("==============================================================")
    http_server.serve_forever()
