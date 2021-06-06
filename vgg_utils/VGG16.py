import tensorflow as tf


def vgg_net(x, n_classes, img_size, reuse, is_train=True, dropout_rate=0.5):
    # Define a scope for reusing the variables
    with tf.variable_scope('VGG16', reuse=reuse):
        x = tf.reshape(x, [-1, img_size, img_size, 1])

        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#1', x.shape)

        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#2', x.shape)

        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#3', x.shape)

        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#4', x.shape)

        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#5', x.shape)

        x_shape = x.get_shape().as_list()
        nodes = x_shape[1] * x_shape[2] * x_shape[3]
        x = tf.reshape(x, [-1, nodes])

        x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
        if is_train:
            x = tf.layers.dropout(x, dropout_rate)

        x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
        if is_train:
            x = tf.layers.dropout(x, dropout_rate)

        out = tf.layers.dense(x, n_classes)
        print(out)

    return out


def vgg_net_slim(x, img_size):
    return_map = {}
    # Define a scope for reusing the variables
    with tf.variable_scope('VGG16', reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1, img_size, img_size, 1])

        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU1_1'] = x
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU1_2'] = x
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#1', x.shape)  #1 (?, 64, 64, 64)

        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU2_1'] = x
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU2_2'] = x
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#2', x.shape)  #2 (?, 32, 32, 128)

        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU3_1'] = x
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU3_2'] = x
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU3_3'] = x
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#3', x.shape)  #3 (?, 16, 16, 256)

        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU4_1'] = x
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU4_2'] = x
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU4_3'] = x
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#4', x.shape)  #4 (?, 8, 8, 512)

        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU5_1'] = x
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU5_2'] = x
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3], strides=1,
                             padding='SAME', activation=tf.nn.relu)
        return_map['ReLU5_3'] = x
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        print('#5', x.shape)  #5 (?, 4, 4, 512)

    return return_map
