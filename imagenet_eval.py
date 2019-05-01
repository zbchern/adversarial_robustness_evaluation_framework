# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the ImageNet data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = 0.7874 recall @ 5 = 0.9436 [50000 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib import slim

import utils
import nets_factory
from imagenet_preprocessing import image_processing
from imagenet_preprocessing.imagenet_data import ImagenetData

# attackers
from attacks import cw
from attacks import deepfool
from attacks import saliency_map
from attacks import fast_gradient_method
from attacks import projected_gradient_descent

FLAGS = tf.app.flags.FLAGS

# Network model configuration
tf.app.flags.DEFINE_string('model_name', 'inception_v4',
                           """Model to be evaluated""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_boolean('bg_class', True,
                            'An offset for the labels in the dataset. This flag is primarily used to '
                            'evaluate the VGG and ResNet architectures which do not use a background '
                            'class for the ImageNet dataset.')

# Adversarial attack
tf.app.flags.DEFINE_string('attacker', 'jsma',
                           """Attack algorithm used to perturb trained models""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

utils.set_best_gpu(1)


def _eval_once(init_fn, top_1_op, top_5_op):
    with tf.Session() as sess:

        init_fn(sess)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            # Counts the number of correct predictions.
            count_top_1 = 0.0
            count_top_5 = 0.0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            print('%s: starting evaluation on (%s) with (%s).' % (datetime.now(), FLAGS.model_name, FLAGS.attacker))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                top_1, top_5 = sess.run([top_1_op, top_5_op])
                count_top_1 += np.sum(top_1)
                count_top_5 += np.sum(top_5)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # Compute precision @ 1.
            precision_at_1 = count_top_1 / total_sample_count * 100
            recall_at_5 = count_top_5 / total_sample_count
            print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
                  (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)


def evaluate(dataset):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        images, labels = image_processing.inputs(dataset)

        # Image rotation
        # images = tf.contrib.image.rotate(images, 90 * math.pi / 180)
        # Brightness Modification
        # images = tf.clip_by_value(images + 0.4, clip_value_min=-1.0, clip_value_max=1.0)

        num_classes = dataset.num_classes() + 1
        if not FLAGS.bg_class:
            labels = tf.subtract(labels, 1)
            num_classes -= 1

        # Build a Graph that computes the logits predictions
        model_fn = lambda x: nets_factory.get_network_fn(FLAGS.model_name,
                                                         num_classes,
                                                         is_training=False,
                                                         reuse=tf.AUTO_REUSE)(x)

        # adv_examples = projected_gradient_descent.pgd(x=images,
        #                                               model_fn=model_fn,
        #                                               one_hot_labels=tf.one_hot(labels, num_classes),
        #                                               attack_type=FLAGS.attacker)

        # adv_examples = fast_gradient_method.fgsm(x=images,
        #                                          model_fn=model_fn,
        #                                          one_hot_labels=tf.one_hot(labels, num_classes),
        #                                          attack_type=FLAGS.attacker)

        # adv_examples = deepfool.deepfool(x=images,
        #                                  model_fn=model_fn)

        adv_examples = saliency_map.jsma(x=images,
                                         model_fn=model_fn,
                                         n_classes=num_classes)

        logits, _ = model_fn(adv_examples)

        # Calculate predictions.
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join('./model_ckpt/' + FLAGS.model_name, FLAGS.model_name + '.ckpt'),
            # os.path.join('./model_ckpt/' + FLAGS.model_name, 'mobilenet_v1_1.0_224.ckpt'),
            # os.path.join('./model_ckpt/' + FLAGS.model_name, 'mobilenet_v2_1.4_224.ckpt'),
            slim.get_model_variables())

        _eval_once(init_fn, top_1_op, top_5_op)


def main(unused_argv=None):
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    evaluate(dataset)


if __name__ == '__main__':
    tf.app.run()
