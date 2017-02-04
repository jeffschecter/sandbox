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


def HiddenLayer(inp, shape, nonlin=ELU, sd=0.001, scope="Hidden"):
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


def FractionallyStridedConv(x, W, input_shape=None, output_channels=None):
    _, height, width, kernels = input_shape or tf.shape(x)
    if output_channels is None:
        output_channels = kernels / 2
    return tf.nn.conv2d_transpose(
        x, W,
        tf.pack([tf.shape(x)[0], height * 2, width * 2, output_channels]),
        [1, 2, 2, 1],
        name="fractionally_strided_conv",
        padding="VALID")


def FractionallyStridedConvLayer(x, nonlinearity, name, input_shape=None, output_channels=None):
    kernels = input_shape[-1] if input_shape else tf.shape(x)[-1]
    if output_channels is None:
        output_channels = kernels / 2
    with tf.name_scope(name) as scope:
        W = Weight([2, 2, int(output_channels), int(kernels)])
        b = Bias([int(output_channels)])
        h = nonlinearity(FractionallyStridedConv(
            x, W,
            input_shape=input_shape,
            output_channels=output_channels) + b)
    return h


def ResNetCell(inp, channels, name, nonlin=tf.nn.relu):
    inp_channels = int(inp.get_shape()[-1])
    scale = int(channels / inp_channels)
    if inp_channels * scale != channels:
        raise ValueError("Channels must be a whole number multiple of input's channels.")
    with tf.name_scope(name) as ns:
        c1 = ConvLayer(inp, 3, channels, nonlin, "ResCellConv1")
        c2 = ConvLayer(c1, 3, channels, nonlin, "ResCellConv2")
        return c2 + tf.tile(inp, [1, 1, 1, scale])


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

