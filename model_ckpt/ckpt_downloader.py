import os
import tensorflow as tf
from models.research.slim.datasets import dataset_utils

# url of checkpoint
url = "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz"
checkpoints_dir = './mobilenet_v1'
os.makedirs(checkpoints_dir, exist_ok=True)

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
