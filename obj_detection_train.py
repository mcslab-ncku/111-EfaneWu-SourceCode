import time
import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#Timmy
import os
from os.path import join
import pickle

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', True, 'disable cv2 window during the process') # this is good for the .ipynb
#Timmy
flags.DEFINE_string('vid_path', "./Data/0318_92589_train_image/", 'path to input video')
flags.DEFINE_string('file_path', "./Data/0318_92589_image_result/train/", 'path to output video')
flags.DEFINE_string('res_path', "./Data/0318_92589_train_image/detect_res.pickle", 'path to detect result file')
flags.DEFINE_boolean('flip', False, 'flip frame 180 degree if true')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.vid_path
    
    res = []
    
    for position in os.listdir(video_path):
        print("position = ",position)
        if not os.path.isdir(join(video_path, position)): 
            continue
        for video in os.listdir(join(video_path, position)):
            if video[len(video)-3:] != "mp4": 
                continue

            print("Video from: ", join(video_path, position, video) )
            vid = cv2.VideoCapture(join(video_path, position, video))
            
            vid_res = []
            
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(join(FLAGS.file_path, "result_" + position + ".mp4"), codec, fps, (width, height))
        
            frame_id = 0
            while True:
                return_value, frame = vid.read()
                if return_value:

                    if FLAGS.flip:
                        frame = np.rot90(frame)
                        frame = np.rot90(frame)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                        print("Video processing complete")
                        break
                    raise ValueError("No image! Try with another video format")
                
                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                prev_time = time.time()
        
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]
        
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=FLAGS.iou,
                    score_threshold=FLAGS.score
                )
                pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                #Timmy
                frame_res = [int(position[0:2]), int(position[2:4]), \
                             frame_id, \
                             boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0],\
                             vid.get(0),\
                             0]                                       
                
                vid_res.append(frame_res)
                
                image = utils.draw_bbox(frame, pred_bbox)
                curr_time = time.time()
                exec_time = curr_time - prev_time
                result = np.asarray(image)
                info = "time: %.2f ms" %(1000*exec_time)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                out.write(result)
        
                frame_id += 1
                
            res.append(vid_res)    
        
                
    with open(join(FLAGS.res_path), 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
