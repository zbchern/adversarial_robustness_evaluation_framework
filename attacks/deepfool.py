import json
import tensorflow as tf

with open('attacks/config.json') as config_file:
    config = json.load(config_file)


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def deepfool(x, model_fn):
    """DeepFool for multi-class classifiers in batch mode.
    """

    eta = float(config['df_eta'])
    niter = int(config['df_niter'])
    clip_min = float(config['df_clip_min'])
    clip_max = float(config['df_clip_max'])

    logits, _ = model_fn(x)
    logits = tf.stop_gradient(logits)
    B, ydim = tf.shape(logits)[0], logits.get_shape().as_list()[1]

    k0 = tf.argmax(logits, axis=1, output_type=tf.int32)
    k0 = tf.stack((tf.range(B), k0), axis=1)

    xshape = x.get_shape().as_list()[1:]
    xdim = _prod(xshape)

    perm = list(range(len(xshape) + 2))
    perm[0], perm[1] = perm[1], perm[0]

    loop_vars = [0, tf.zeros_like(x)]

    def loop_cond(idx, _):
        return idx < niter

    def loop_body(idx, z):
        xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
        y, _ = model_fn(xadv)
        y = tf.nn.softmax(y)

        print('1--->{0}'.format(idx))
        print(ydim)
        gs = [tf.gradients(y[:, i], xadv)[0] for i in range(ydim)]
        g = tf.stack(gs, axis=0)
        g = tf.transpose(g, perm)

        print('2--->{0}'.format(idx))
        yk = tf.expand_dims(tf.gather_nd(y, k0), axis=1)
        gk = tf.expand_dims(tf.gather_nd(g, k0), axis=1)

        print('33--->{0}'.format(idx))
        a = tf.abs(y - yk)
        b = g - gk
        c = tf.norm(tf.reshape(b, [-1, ydim, xdim]), axis=-1)

        # Assume 1) 0/0=tf.nan 2) tf.argmin ignores nan
        score = a / c

        print('4--->{0}'.format(idx))
        ind = tf.argmin(score, axis=1, output_type=tf.int32)
        ind = tf.stack((tf.range(B), ind), axis=1)

        print('5--->{0}'.format(idx))
        si, bi = tf.gather_nd(score, ind), tf.gather_nd(b, ind)
        si = tf.reshape(si, [-1] + [1] * len(xshape))
        dx = si * bi
        print('6--->{0}'.format(idx))
        return idx + 1, z + dx

    with tf.control_dependencies([k0]):
        _, noise = tf.while_loop(loop_cond,
                                 loop_body,
                                 loop_vars,
                                 name='deepfool',
                                 back_prop=False)
        xadv = tf.stop_gradient(x + noise * (1 + eta))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return xadv
