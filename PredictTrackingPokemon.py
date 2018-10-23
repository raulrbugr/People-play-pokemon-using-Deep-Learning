import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from pynput.keyboard import Key, Controller

keyboard = Controller()

import cv2
cap = cv2.VideoCapture(1);

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      image_np=cv2.resize(image_np, (1280,720))
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      #print([category_index.get(i) for i in classes[0]])
      #print(category_index.get(1))
      #print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])

      your_list = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
      name = [item['name'] for item in your_list]
      #print(name)
      wordwanted="person"
      
      if wordwanted in name:
          print(wordwanted)
          width, height,channels = image_np.shape
          #width, height = image_np.size
          xmin = boxes[0][0][0]*width
          ymin = boxes[0][0][1]*height
          xmax = boxes[0][0][2]*width
          ymax = boxes[0][0][3]*height
          xcenter=(xmax-((xmax-xmin)/2))
          ycenter=(ymax-((ymax-ymin)/2))
          #print (height,width)
          #print 'Top left'
          #print (xmin,ymin,)
          #print 'Bottom right'
          #print (xmax,ymax)
          print 'Center'
          print (xcenter,ycenter)
          
          cv2.circle(image_np,(int(ycenter),int(xcenter)),10,(0,0,255),-1)

          #GENERAL
          #if( (xcenter >= 480 and xcenter < 720) and (ycenter >= 0 and ycenter < 640)):
           #keyboard.press('u') #arriba
           #time.sleep( 0.01 )
           #keyboard.release('u')
          #elif((xcenter >= 240 and xcenter < 480) and (ycenter >= 0 and ycenter < 640)):
           #keyboard.press('j') #abajo
           #time.sleep( 0.01 )
           #keyboard.release('j')
          #elif((xcenter >= 480 and xcenter < 720) and (ycenter >= 640 and ycenter < 1280)):
           #keyboard.press('h') #izquierda
           #time.sleep( 0.01 )
           #keyboard.release('h')
          #elif((xcenter >= 0 and xcenter < 240) and (ycenter >= 0 and ycenter < 640)):
           #keyboard.press('a') #A
           #time.sleep( 0.01 )
           #keyboard.release('a')
          #elif((xcenter >= 0 and xcenter < 240) and (ycenter >= 640 and ycenter < 1280)):
           #keyboard.press('h') #B
           #time.sleep( 0.01 )
           #keyboard.release('h')
          #else:
           #keyboard.press('k') #derecha
           #time.sleep( 0.01 )
           #keyboard.release('k')
          despx=320
          despy=0
          despx2=640
          despy2=180
          despx3=320
          despy3=240
          

          if( ((ycenter >= 106 and ycenter < 213) and (xcenter >= 240 and xcenter < 360)) or ((ycenter >= 106+despx and ycenter < 213+despx) and (xcenter >= 240+despy and xcenter < 360+despy)) or ((ycenter >= 106+despx2 and ycenter < 213+despx2) and (xcenter >= 240+despy2 and xcenter < 360+despy2)) or ((ycenter >= 106+despx3 and ycenter < 213+despx3) and (xcenter >= 240+despy3 and xcenter < 360+despy3)) ):
           keyboard.press('u') #arriba
           time.sleep( 0.1 )
           keyboard.release('u')
           print "ARRIBA"
          elif(((ycenter >= 106 and ycenter < 213) and (xcenter >= 360 and xcenter < 480)) or ((ycenter >= 106+despx and ycenter < 213+despx) and (xcenter >= 360+despy and xcenter < 480+despy)) or ((ycenter >= 106+despx2 and ycenter < 213+despx2) and (xcenter >= 360+despy2 and xcenter < 480+despy2)) or ((ycenter >= 106+despx3 and ycenter < 213+despx3) and (xcenter >= 360+despy3 and xcenter < 480+despy3))):
           keyboard.press('j') #abajo
           time.sleep( 0.1 )
           keyboard.release('j')
           print "ABAJO"
          elif(((ycenter >= 0 and ycenter < 106) and (xcenter >= 360 and xcenter < 480)) or ((ycenter >= 0+despx and ycenter < 106+despx) and (xcenter >= 360+despy and xcenter < 480+despy)) or ((ycenter >= 0+despx2 and ycenter < 106+despx2) and (xcenter >= 360+despy2 and xcenter < 480+despy2)) or ((ycenter >= 0+despx3 and ycenter < 106+despx3) and (xcenter >= 360+despy3 and xcenter < 480+despy3))):
           keyboard.press('h') #izquierda
           time.sleep( 0.1 )
           keyboard.release('h')
           print "IZQUIERDA"
          elif(((ycenter >= 0 and ycenter < 106) and (xcenter >= 240 and xcenter < 360)) or ((ycenter >= 0+despx and ycenter < 160+despx) and (xcenter >= 240+despy and xcenter < 360+despy)) or ((ycenter >= 0+despx2 and ycenter < 160+despx2) and (xcenter >= 240+despy2 and xcenter < 360+despy2)) or ((ycenter >= 0+despx3 and ycenter < 160+despx3) and (xcenter >= 240+despy3 and xcenter < 360+despy3))):
           keyboard.press('z') #A
           time.sleep( 0.01 )
           keyboard.release('z')
           print "A"
          elif(((ycenter >= 213 and ycenter < 320) and (xcenter >= 240 and xcenter < 360)) or ((ycenter >= 213+despx and ycenter < 320+despx) and (xcenter >= 240+despy and xcenter < 360+despy)) or ((ycenter >= 213+despx2 and ycenter < 320+despx2) and (xcenter >= 240+despy2 and xcenter < 360+despy2)) or ((ycenter >= 213+despx3 and ycenter < 320+despx3) and (xcenter >= 240+despy3 and xcenter < 360+despy3))):
           keyboard.press('x') #B
           time.sleep( 0.01 )
           keyboard.release('x')
           print "B"
          elif(((ycenter >= 213 and ycenter < 320) and (xcenter >= 360 and xcenter < 480)) or ((ycenter >= 213+despx and ycenter < 320+despx) and (xcenter >= 360+despy and xcenter < 480+despy)) or ((ycenter >= 213+despx2 and ycenter < 320+despx2) and (xcenter >= 360+despy2 and xcenter < 480+despy2)) or ((ycenter >= 213+despx3 and ycenter < 320+despx3) and (xcenter >= 360+despy3 and xcenter < 480+despy3))):
           keyboard.press('k') #derecha
           time.sleep( 0.1 )
           keyboard.release('k')
           print "DERECHA"
          elif((ycenter >= 640 and ycenter < 1280) ):
           keyboard.press('z') #A
           time.sleep( 0.01 )
           keyboard.release('z')
           print "A"
          elif((ycenter >= 0 and ycenter < 640) ):
           keyboard.press('x') #B
           time.sleep( 0.01 )
           keyboard.release('x')
           print "B"
      
      #GENERAL
      #cv2.line(image_np, (0, 240), (1279, 240), (0,255,0), 4)
      #cv2.line(image_np, (0, 480), (1279, 480), (0,255,0), 4)
      #cv2.line(image_np, (640, 0), (640, 719), (0,255,0), 4)
     
      cv2.line(image_np, (0, 240), (320, 240), (0,255,0), 4)#horizontal
      cv2.line(image_np, (0, 360), (320, 360), (0,255,0), 4)
      cv2.line(image_np, (0, 480), (320, 480), (0,255,0), 4)
      cv2.line(image_np, (320, 240), (320, 480), (0,255,0), 4)#vertical
      cv2.line(image_np, (213, 240), (213, 480), (0,255,0), 4)
      cv2.line(image_np, (106, 240), (106, 480), (0,255,0), 4)


      despx=320
      despy=0
      cv2.line(image_np, (0+despx, 240+despy), (320+despx, 240+despy), (0,255,0), 4)#horizontal
      cv2.line(image_np, (0+despx, 360+despy), (320+despx, 360+despy), (0,255,0), 4)
      cv2.line(image_np, (0+despx, 480+despy), (320+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (320+despx, 240+despy), (320+despx, 480+despy), (0,255,0), 4)#vertical
      cv2.line(image_np, (213+despx, 240+despy), (213+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (106+despx, 240+despy), (106+despx, 480+despy), (0,255,0), 4)

      despx=640
      despy=180
      cv2.line(image_np, (0+despx, 240+despy), (320+despx, 240+despy), (0,255,0), 4)#horizontal
      cv2.line(image_np, (0+despx, 360+despy), (320+despx, 360+despy), (0,255,0), 4)
      cv2.line(image_np, (0+despx, 480+despy), (320+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (320+despx, 240+despy), (320+despx, 480+despy), (0,255,0), 4)#vertical
      cv2.line(image_np, (213+despx, 240+despy), (213+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (106+despx, 240+despy), (106+despx, 480+despy), (0,255,0), 4) 

      despx=320
      despy=240
      cv2.line(image_np, (0+despx, 240+despy), (320+despx, 240+despy), (0,255,0), 4)#horizontal
      cv2.line(image_np, (0+despx, 360+despy), (320+despx, 360+despy), (0,255,0), 4)
      cv2.line(image_np, (0+despx, 480+despy), (320+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (320+despx, 240+despy), (320+despx, 480+despy), (0,255,0), 4)#vertical
      cv2.line(image_np, (213+despx, 240+despy), (213+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (106+despx, 240+despy), (106+despx, 480+despy), (0,255,0), 4)
      cv2.line(image_np, (0+despx, 240+despy), (0+despx, 480+despy), (0,255,0), 4)            

      #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
