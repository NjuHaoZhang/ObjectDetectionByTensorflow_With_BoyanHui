import numpy as np  
import os  
import six.moves.urllib as urllib  
import sys  
import tarfile  
import tensorflow as tf  
import zipfile  
  
from collections import defaultdict  
from io import StringIO  
from matplotlib import pyplot as plt  
from PIL import Image  
  
# This is needed since the notebook is stored in the object_detection folder.  
sys.path.append("..")  
from object_detection.utils import ops as utils_ops  
  
if tf.__version__ < '1.4.0':  
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')  
    
# This is needed to display the images.  
from object_detection.utils import label_map_util  
  
from object_detection.utils import visualization_utils as vis_util  
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
##Model preparation##  
  
# What model to download.  
MODEL_NAME = r'my_models' # 结合35行
MODEL_FILE = MODEL_NAME + '.tar.gz'  
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  
  
# Path to frozen detection graph. This is the actual model that is used for the object detection.  
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb') # 改成你自己pb文件的路径
  
# List of the strings that is used to add correct label for each box.  
PATH_TO_LABELS = os.path.join('data', 'my_pascal_label_map.pbtxt')  # 改成你自己的label_map.pbtxt路径
  
NUM_CLASSES = 1
  
## Download Model##  
#opener = urllib.request.URLopener()  
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)  
# tar_file = tarfile.open(MODEL_FILE)  
# for file in tar_file.getmembers():  
#   file_name = os.path.basename(file.name)  
#   if 'frozen_inference_graph.pb' in file_name:  
#     tar_file.extract(file, os.getcwd())  
      
## Load a (frozen) Tensorflow model into memory.  
detection_graph = tf.Graph()  
with detection_graph.as_default():  
  od_graph_def = tf.GraphDef()  
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  
    serialized_graph = fid.read()  
    od_graph_def.ParseFromString(serialized_graph)  
    tf.import_graph_def(od_graph_def, name='')  
      
## Loading label map  
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)  
category_index = label_map_util.create_category_index(categories)  
  
import imageio  
imageio.plugins.ffmpeg.download() 
  
from moviepy.editor import VideoFileClip  
from IPython.display import HTML  
  
def detect_objects(image_np, sess, detection_graph):  
    # 扩展维度，应为模型期待: [1, None, None, 3]  
    image_np_expanded = np.expand_dims(image_np, axis=0)  
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  
  
    # 每个框代表一个物体被侦测到  
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  
  
    #每个分值代表侦测到物体的可信度.    
    scores = detection_graph.get_tensor_by_name('detection_scores:0')  
    classes = detection_graph.get_tensor_by_name('detection_classes:0')  
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')  
  
    # 执行侦测任务.    
    (boxes, scores, classes, num_detections) = sess.run(  
        [boxes, scores, classes, num_detections],  
        feed_dict={image_tensor: image_np_expanded})  
  
    # 检测结果的可视化  
    vis_util.visualize_boxes_and_labels_on_image_array(  
        image_np,  
        np.squeeze(boxes),  
        np.squeeze(classes).astype(np.int32),  
        np.squeeze(scores),  
        category_index,  
        use_normalized_coordinates=True,  
        line_thickness=8)  
    return image_np  
  
def process_image(image):  
    # NOTE: The output you return should be a color image (3 channel) for processing video below  
    # you should return the final output (image with lines are drawn on lanes)  
    with detection_graph.as_default():  
        with tf.Session(graph=detection_graph) as sess:  
            image_process = detect_objects(image, sess, detection_graph)  
            return image_process  
          
white_output = 'test_out.mp4' 
# clip1 = VideoFileClip("video1.mp4").subclip(10,20)
clip1 = VideoFileClip("test.mp4") # 测试顺序数，第4个视频，但是不支持264格式，所以我给转成了mp4
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s  
white_clip.write_videofile(white_output, audio=False)  
  
from moviepy.editor import *  
clip1 = VideoFileClip("test_out.mp4")  # 读取识别后的视频流 {其实可以直接手动播放输出的视频}
clip1.write_gif("test_out.gif")  # 制作gif图