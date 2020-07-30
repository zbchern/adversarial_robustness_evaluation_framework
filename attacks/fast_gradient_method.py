import json
import tensorflow as tf


with open('attacks/config.json') as config_file:
    config = json.load(config_file)


def fgm(x,
        model_fn,
        one_hot_labels):

    epsilon = float(config['fgm_epsilon'])
    niter = config['fgm_niter']
    clip_min = float(config['fgm_clip_min'])
    clip_max = float(config['fgm_clip_max'])

    loop_vars = [0, tf.identity(x)]

    def loop_cond(idx, _):
        return idx < niter

    def loop_body(idx, adv_images):
        logits, _ = model_fn(adv_images)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,
                                                                        logits=logits))
        adv_images = tf.stop_gradient(adv_images + epsilon * tf.sign(tf.gradients(loss, adv_images)[0]))
        adv_images = tf.clip_by_value(adv_images, clip_min, clip_max)
        return idx + 1, adv_images

    _, result = tf.while_loop(loop_cond,
                              loop_body,
                              loop_vars,
                              back_prop=False,
                              name='fast_gradient_method')
    return result


def fgsm(x,
         model_fn,
         one_hot_labels,
         attack_type):

    if attack_type.startswith('fgsm'):
        return fgm(x,
                   model_fn,
                   one_hot_labels)
    else:
        raise ValueError('Invalid attack type: {0}'.format(attack_type))
