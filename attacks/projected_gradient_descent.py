"""Library with adversarial gradient descent attacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf


with open('attacks/config.json') as config_file:
    config = json.load(config_file)


def generate_pgd_common(x,
                        model_fn,
                        one_hot_labels,
                        perturbation_multiplier):

    # todo: fix args
    epsilon = float(config['pgd_epsilon'])
    step_size = float(config['pgd_step_size'])
    niter = int(config['pgd_niter'])
    clip_min = float(config['pgd_clip_min'])
    clip_max = float(config['pgd_clip_max'])

    # clipping boundaries
    clip_min = tf.maximum(x - epsilon, clip_min)
    clip_max = tf.minimum(x + epsilon, clip_max)

    # compute starting point
    start_x = x + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    # main iteration of PGD
    loop_vars = [0, start_x]

    def loop_cond(idx, _):
        return idx < niter

    def loop_body(idx, adv_images):
        logits, _ = model_fn(adv_images)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot_labels,
                logits=logits))
        perturbation = step_size * tf.sign(tf.gradients(ys=loss, xs=adv_images)[0])
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_value_min=clip_min, clip_value_max=clip_max)
        return idx + 1, new_adv_images

    with tf.control_dependencies([start_x]):
        _, result = tf.while_loop(cond=loop_cond,
                                  body=loop_body,
                                  loop_vars=loop_vars,
                                  back_prop=False,
                                  parallel_iterations=1)
        return result


def pgd(x,
        model_fn,
        one_hot_labels,
        attack_type):
    # todo: add args

    if attack_type.startswith('vallina'):
        return x

    elif attack_type.startswith('pgd'):
        return generate_pgd_common(x,
                                   model_fn,
                                   one_hot_labels,
                                   perturbation_multiplier=1.0)

    else:
        raise ValueError('Invalid attack type: {0}'.format(attack_type))


