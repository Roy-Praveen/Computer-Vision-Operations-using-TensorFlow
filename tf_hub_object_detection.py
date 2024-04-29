'''These scripts contain various utilities absorbed from tensorflow hub 
   for object detection workflows'''

'''Installation of Object Detection API

git clone --depth 1 https://github.com/tensorflow/models

%%bash
sudo apt install -y protobuf-compiler
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

'''

'''There is a bonus code for carrying out instance Segementation uisng Mask RCNN
'''

'''Requirements
numpy==1.24.3
protobuf==3.20.3
'''

'''Import Packages'''
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

 ####################################################

'''Loading the image and converting into a numpy array

'''
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    Actually the below function converts the image into a tensor of 4 dimensions
    First dimension is the index of the image itself
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

  ################################################

'''Importing 3 of the most important util classes'''
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

 #################################################

'''If you label map was a key value pair stored in pbtxt format use this code
   to  bring it to category index dictionary format'''
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

 #################################################


'''Model Selection and Loading'''
model_display_name = 'CenterNet HourGlass104 Keypoints 512x512'
model_handle = ALL_MODELS[model_display_name]         #For ALL_MODELS check the all_models.txt file in this repo

print('Selected model:'+ model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))

hub_model = hub.load(model_handle)
print('model loaded!')

 #################################################

'''Loading Image with some operations which are usually done like flipping and converting to grayscale'''

image_np = load_image_into_numpy_array(image_path)

# Flip horizontally
if(flip_image_horizontally):
  image_np[0] = np.fliplr(image_np[0]).copy()

# Convert image to grayscale
if(convert_image_to_grayscale):
  image_np[0] = np.tile(
    np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

plt.figure(figsize=(24,32))
plt.imshow(image_np[0])
plt.show()

 ##################################################

 '''Running Inference on the loaded image'''

results = hub_model(image_np)

# different object detection models have additional results
# all of them are explained in the documentation
result = {key:value.numpy() for key,value in results.items()}
print(result.keys())

 ###################################################
'''Visualizing results'''
label_id_offset = 0 # Offset set to zero as there is no offset between label id of model and category Index
image_np_with_detections = image_np.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in result:
  keypoints = result['detection_keypoints'][0]
  keypoint_scores = result['detection_keypoint_scores'][0]

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections[0],
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False)
      '''Keypoints are points of interest which are areas of interest in the object class
         For example, in human face, it is nose, eyes, etc'''
      #keypoints=keypoints,                   
      #keypoint_scores=keypoint_scores,
      #keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

plt.figure(figsize=(24,32))
plt.imshow(image_np_with_detections[0])
plt.show()

 ###################################################










