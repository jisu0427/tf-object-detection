import tensorflow as tf
import os
import pathlib

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

# 내 로컬에 설치된 레이블 파일을, 인덱스와 연결시킨다.
PATH_TO_LABELS = 'E:\\학원2\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS)

print(category_index)