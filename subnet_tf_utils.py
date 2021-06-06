import tensorflow as tf


def get_initializer(init_method):
    if init_method == 'xavier_normal':
        initializer = tf.glorot_normal_initializer()
    elif init_method == 'xavier_uniform':
        initializer = tf.glorot_uniform_initializer()
    elif init_method == 'he_normal':
        initializer = tf.keras.initializers.he_normal()
    elif init_method == 'he_uniform':
        initializer = tf.keras.initializers.he_uniform()
    elif init_method == 'lecun_normal':
        initializer = tf.keras.initializers.lecun_normal()
    elif init_method == 'lecun_uniform':
        initializer = tf.keras.initializers.lecun_uniform()
    else:
        raise Exception('Unknown initializer:', init_method)
    return initializer


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name) as scope:
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def batchnorm(input, name='batch_norm', init_method=None):
    if init_method is not None:
        initializer = get_initializer(init_method)
    else:
        initializer = tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32)

    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=initializer)
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def layernorm(input, name='layer_norm', init_method=None):
    if init_method is not None:
        initializer = get_initializer(init_method)
    else:
        initializer = tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32)

    with tf.variable_scope(name):
        n_neurons = input.get_shape()[3]
        offset = tf.get_variable("offset", [n_neurons], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [n_neurons], dtype=tf.float32,
                                initializer=initializer)
        offset = tf.reshape(offset, [1, 1, -1])
        scale = tf.reshape(scale, [1, 1, -1])
        mean, variance = tf.nn.moments(input, axes=[1, 2, 3], keep_dims=True)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def instance_norm(input, name="instance_norm", init_method=None):
    if init_method is not None:
        initializer = get_initializer(init_method)
    else:
        initializer = tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32)

    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=initializer)
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def linear1d(inputlin, inputdim, outputdim, name="linear1d", init_method=None):
    if init_method is not None:
        initializer = get_initializer(init_method)
    else:
        initializer = None

    with tf.variable_scope(name) as scope:
        weight = tf.get_variable("weight", [inputdim, outputdim], initializer=initializer)
        bias = tf.get_variable("bias", [outputdim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        return tf.matmul(inputlin, weight) + bias


def general_conv2d(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                   stddev=0.02, padding="SAME", name="conv2d", do_norm=True, norm_type='instance_norm', do_relu=True,
                   relufactor=0, is_training=True, init_method=None):
    if init_method is not None:
        initializer = get_initializer(init_method)
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(name) as scope:
        conv = tf.contrib.layers.conv2d(inputconv, output_dim, [filter_width, filter_height],
                                        [stride_width, stride_height], padding, activation_fn=None,
                                        weights_initializer=initializer,
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv, init_method=init_method)
                # conv = tf.contrib.layers.instance_norm(conv, epsilon=1e-05, center=True, scale=True,
                #                                        scope='instance_norm')
            elif norm_type == 'batch_norm':
                # conv = batchnorm(conv, init_method=init_method)
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, center=True, scale=True, scope="batch_norm")
            elif norm_type == 'layer_norm':
                # conv = layernorm(conv, init_method=init_method)
                conv = tf.contrib.layers.layer_norm(conv, center=True, scale=True, scope='layer_norm')

        if do_relu:
            if relufactor == 0:
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def generative_cnn_c3_encoder(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 256 * 4 * 4))
        tensor_x = linear1d(tensor_x, 256 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_deeper(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 512]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 512, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_combine33(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

    tensor_x = tf.concat([local_x, global_x], axis=-1)

    with tf.variable_scope('Combined_Encoder', reuse=tf.AUTO_REUSE):
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_combine43(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

    tensor_x = tf.concat([local_x, global_x], axis=-1)

    with tf.variable_scope('Combined_Encoder', reuse=tf.AUTO_REUSE):
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_combine53(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

    tensor_x = tf.concat([local_x, global_x], axis=-1)

    with tf.variable_scope('Combined_Encoder', reuse=tf.AUTO_REUSE):
        tensor_x_sp = tensor_x  # [N, h, w, 256]
        tensor_x = tf.reshape(tensor_x, (-1, 1024 * 4 * 4))
        tensor_x = linear1d(tensor_x, 1024 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_combineFC(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        local_x = tf.reshape(local_x, (-1, 512 * 4 * 4))
        local_x = linear1d(local_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            local_x = tf.nn.dropout(local_x, drop_keep_prob)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        global_x = tf.reshape(global_x, (-1, 512 * 4 * 4))
        global_x = linear1d(global_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            global_x = tf.nn.dropout(global_x, drop_keep_prob)

    tensor_x_sp = None
    tensor_x = tf.concat([local_x, global_x], axis=-1)
    return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_combineFC_jointAttn(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5,
                                                  init_method=None, combine_manner='attn'):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        share_x = local_x

        with tf.variable_scope('Attn_branch', reuse=tf.AUTO_REUSE):
            attn_x = general_conv2d(share_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                    is_training=is_training, name="CNN_ENC_1", init_method=init_method)
            attn_x = general_conv2d(attn_x, 32, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                                    is_training=is_training, name="CNN_ENC_2", init_method=init_method)
            attn_x = general_conv2d(attn_x, 1, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                                    is_training=is_training, name="CNN_ENC_3", init_method=init_method)
            attn_map = tf.nn.sigmoid(attn_x)  # (N, H/8, W/8, 1), [0.0, 1.0]

        if combine_manner == 'attn':
            attn_feat = attn_map * share_x + share_x
        else:
            raise Exception('Unknown combine_manner', combine_manner)

        local_x = general_conv2d(attn_feat, 256, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        local_x = tf.reshape(local_x, (-1, 512 * 4 * 4))
        local_x = linear1d(local_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            local_x = tf.nn.dropout(local_x, drop_keep_prob)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        global_x = tf.reshape(global_x, (-1, 512 * 4 * 4))
        global_x = linear1d(global_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            global_x = tf.nn.dropout(global_x, drop_keep_prob)

    tensor_x_sp = None
    tensor_x = tf.concat([local_x, global_x], axis=-1)
    return tensor_x, tensor_x_sp, attn_map


def generative_cnn_c3_encoder_combineFC_sepAttn(local_inputs, global_inputs, is_training=True, drop_keep_prob=0.5,
                                                  init_method=None, combine_manner='attn'):
    local_x = local_inputs
    global_x = global_inputs

    with tf.variable_scope('Attn_branch', reuse=tf.AUTO_REUSE):
        attn_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        attn_x = general_conv2d(attn_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        attn_x = general_conv2d(attn_x, 64, filter_height=3, filter_width=3,
                                is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        attn_x = general_conv2d(attn_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        attn_x = general_conv2d(attn_x, 128, filter_height=3, filter_width=3,
                                is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        attn_x = general_conv2d(attn_x, 32, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                                is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        attn_x = general_conv2d(attn_x, 1, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                                is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)
        attn_map = tf.nn.sigmoid(attn_x)  # (N, H/8, W/8, 1), [0.0, 1.0]

    with tf.variable_scope('Local_Encoder', reuse=tf.AUTO_REUSE):
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        local_x = general_conv2d(local_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        local_x = general_conv2d(local_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        local_x = general_conv2d(local_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        if combine_manner == 'attn':
            attn_feat = attn_map * local_x + local_x
        elif combine_manner == 'channel':
            attn_feat = tf.concat([local_x, attn_map], axis=-1)
        else:
            raise Exception('Unknown combine_manner', combine_manner)

        local_x = general_conv2d(attn_feat, 256, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        local_x = general_conv2d(local_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3,
                                 is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        local_x = general_conv2d(local_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                 is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        local_x = tf.reshape(local_x, (-1, 512 * 4 * 4))
        local_x = linear1d(local_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            local_x = tf.nn.dropout(local_x, drop_keep_prob)

    with tf.variable_scope('Global_Encoder', reuse=tf.AUTO_REUSE):
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        global_x = general_conv2d(global_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        global_x = general_conv2d(global_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        global_x = general_conv2d(global_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        global_x = general_conv2d(global_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        global_x = general_conv2d(global_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)

        global_x = tf.reshape(global_x, (-1, 512 * 4 * 4))
        global_x = linear1d(global_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            global_x = tf.nn.dropout(global_x, drop_keep_prob)

    tensor_x_sp = None
    tensor_x = tf.concat([local_x, global_x], axis=-1)
    return tensor_x, tensor_x_sp, attn_map


def generative_cnn_c3_encoder_deeper13(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_c3_encoder_deeper13_attn(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_1_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = self_attention(tensor_x, 64)

        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_3_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_4_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3,
                                  is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, filter_height=3, filter_width=3, stride_height=1, stride_width=1,
                                  is_training=is_training, name="CNN_ENC_5_3", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_encoder(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_1_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_3_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_5_2", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 256 * 4 * 4))
        tensor_x = linear1d(tensor_x, 256 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_encoder_deeper(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, is_training=is_training, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_1_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, is_training=is_training, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, is_training=is_training, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_3_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, is_training=is_training, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, is_training=is_training, name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 512, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_5_2", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 512]

        tensor_x = tf.reshape(tensor_x, (-1, 512 * 4 * 4))
        tensor_x = linear1d(tensor_x, 512 * 4 * 4, 512, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def generative_cnn_encoder_deeper13(inputs, is_training=True, drop_keep_prob=0.5, init_method=None):
    tensor_x = inputs

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, is_training=is_training,
                                  name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 32, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_1_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 64, is_training=is_training,
                                  name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_2_2", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 128, is_training=is_training,
                                  name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_3_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_3_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, is_training=is_training,
                                  name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_4_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_4_3", init_method=init_method)

        tensor_x = general_conv2d(tensor_x, 256, is_training=is_training,
                                  name="CNN_ENC_5", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_5_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 256, stride_height=1, stride_width=1, is_training=is_training,
                                  name="CNN_ENC_5_3", init_method=init_method)
        tensor_x_sp = tensor_x  # [N, h, w, 256]

        tensor_x = tf.reshape(tensor_x, (-1, 256 * 4 * 4))
        tensor_x = linear1d(tensor_x, 256 * 4 * 4, 128, name='CNN_ENC_FC', init_method=init_method)

        if is_training:
            tensor_x = tf.nn.dropout(tensor_x, drop_keep_prob)

        return tensor_x, tensor_x_sp


def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')


def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def self_attention(x, in_channel, name='self_attention'):
    with tf.variable_scope(name) as scope:
        f = general_conv2d(x, in_channel // 8, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                           do_norm=False, do_relu=False, name='f_conv')  # (N, h, w, c')
        f = max_pooling(f)  # (N, h', w', c')
        g = general_conv2d(x, in_channel // 8, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                           do_norm=False, do_relu=False, name='g_conv')  # (N, h, w, c')
        h = general_conv2d(x, in_channel, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                           do_norm=False, do_relu=False, name='h_conv')  # (N, h, w, c)
        h = max_pooling(h)  # (N, h', w', c)

        # M = h * w, M' = h' * w'
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # (N, M, M')
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # (N, M, c)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # (N, h, w, c)
        o = general_conv2d(o, in_channel, filter_height=1, filter_width=1, stride_height=1, stride_width=1,
                           do_norm=False, do_relu=False, name='attn_conv')

        x = gamma * o + x

    return x


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def cnn_discriminator_wgan_gp(discrim_inputs, discrim_targets, init_method=None):
    tensor_x = tf.concat([discrim_inputs, discrim_targets], axis=3)  # (N, H, W, 3 + 1)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        tensor_x = general_conv2d(tensor_x, 32, filter_height=3, filter_width=3,
                                  is_training=True, name="CNN_ENC_1", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 64, filter_height=3, filter_width=3,
                                  is_training=True, name="CNN_ENC_2", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=True, name="CNN_ENC_3", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 128, filter_height=3, filter_width=3,
                                  is_training=True, name="CNN_ENC_4", init_method=init_method)
        tensor_x = general_conv2d(tensor_x, 1, filter_height=3, filter_width=3,
                                  is_training=True, name="CNN_ENC_5", init_method=init_method)
        # (N, H/32, W/32, 1)

        d_out = global_avg_pooling(tensor_x)  # (N, 1)

    return d_out
