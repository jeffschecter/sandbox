import tensorflow as tf


FLOAT = tf.float32


# --------------------------------------------------------------------------- #
# Nonlinearities                                                              #
# --------------------------------------------------------------------------- #
def LeakyReLUMaker(leak):
    def LeakyReLU(x):
        return tf.maximum(x, 0) + leak * tf.minimum(x, 0)
    return LeakyReLU


def ELU(x):
    pos = tf.cast(tf.greater_equal(x, 0), FLOAT)
    return (pos * x) + ((1 - pos) * (tf.exp(x) - 1))


# --------------------------------------------------------------------------- #
# Simple Layer Construction                                                   #
# --------------------------------------------------------------------------- #
def Weight(shape, sd=0.02):
    initial = tf.random_normal(shape, mean=0, stddev=sd)
    return tf.Variable(initial, name="weights")


def Bias(shape, sd=0.02):
    initial = tf.random_normal(shape, mean=0, stddev=sd)
    return tf.Variable(initial, name="bias")


def HiddenLayer(inp, shape, nonlin=ELU, sd=0.001, scope="RBM"):
    with tf.name_scope(scope) as ns:
        W = Weight(shape, sd=sd)
        b = Bias([shape[1]], sd=sd)
        h = nonlin(tf.matmul(inp, W) + b)
        return W, b, h


# --------------------------------------------------------------------------- #
# Convolutional Layers                                                        #
# --------------------------------------------------------------------------- #
def Conv2D(x, W, name="Conv2D"):
    return tf.nn.conv2d(
        x, W,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name=name)


def ConvLayer(x, filter_size, out_channels, nonlinearity, name):
    in_channels = int(x.get_shape()[-1])
    with tf.name_scope(name) as scope:
        W = Weight([filter_size, filter_size, in_channels, out_channels])
        b = Bias([out_channels])
        h = nonlinearity(Conv2D(x, W) + b)
    return h


def MaxPool2x2(x, name="MaxPool"):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name=name)


def StridedConv(x, W):
    return tf.nn.conv2d(
        x, W,
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="strided_conv")


# The function tf.nn.deconv2d is an undecumented part of the TensorFlow API,
# and is pretty half baked. For instance, unlike just about every other similar
# function, deconv2d can't handle a batch size of -1. Even worse, instead of
# throwing a py Exception, it results in an uncaught C++ error that crashes the
# iPython kernel. And its signature or internals may unexpectedly change at any
# point in the future, since it's not a stable part of the API.

# Anyway, the crux of the matter is: it forces us to commit to a single batch
# size for the entire network architecture.
BATCH_SIZE = 100
def FractionallyStridedConv(x, W, output_channels=None, batch_size=BATCH_SIZE):
    _, height, width, kernels = x.get_shape().as_list()
    if output_channels is None:
        output_channels = kernels / 2
    return tf.nn.deconv2d(
        x, W,
        [batch_size, height * 2, width * 2, output_channels],
        [1, 2, 2, 1],
        name="fractionally_strided_conv")


# --------------------------------------------------------------------------- #
# A batch norm that doesn't require running separate update ops!              #
# Stolen from http://stackoverflow.com/a/34634291                             #
# --------------------------------------------------------------------------- #
def BatchNorm(should_train, inp, scope="BatchNorm", affine=True):
    with tf.variable_scope(scope):
        beta = tf.Variable(
            tf.constant(0.0, shape=inp.get_shape()[-1:]),
            name="beta", trainable="true")
        gamma = tf.Variable(
            tf.constant(1.0, shape=inp.get_shape()[-1:]),
            name="gamma", trainable="true")
        batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean = ema.average(batch_mean)
        ema_var = ema.average(batch_var)
        
        def MeanVarWithUpdate():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
            
        mean, var = tf.python.control_flow_ops.cond(
            should_train,
            MeanVarWithUpdate,
            lambda: (ema_mean, ema_var))

        return tf.nn.batch_norm_with_global_normalization(
            inp, mean, var, beta, gamma, 0.001, affine)

