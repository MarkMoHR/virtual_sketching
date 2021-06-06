import tensorflow as tf


class RasterUnit(object):
    def __init__(self,
                 raster_size,
                 input_params,  # (N, 10)
                 reuse=False):
        self.raster_size = raster_size
        self.input_params = input_params

        with tf.variable_scope("raster_unit", reuse=reuse):
            self.build_unit()

    def build_unit(self):
        x = self.input_params  # (N, 10)
        x = self.fully_connected(x, 10, 512, scope='fc1')  # (N, 512)
        x = tf.nn.relu(x)
        x = self.fully_connected(x, 512, 1024, scope='fc2')  # (N, 1024)
        x = tf.nn.relu(x)
        x = self.fully_connected(x, 1024, 2048, scope='fc3')  # (N, 2048)
        x = tf.nn.relu(x)
        x = self.fully_connected(x, 2048, 4096, scope='fc4')  # (N, 4096)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (-1, 16, 16, 16))  # (N, 16, 16, 16)
        x = tf.transpose(x, (0, 2, 3, 1))

        x = self.conv2d(x, 32, 3, 1, scope='conv1')  # (N, 16, 16, 32)
        x = tf.nn.relu(x)
        x = self.conv2d(x, 32, 3, 1, scope='conv2')  # (N, 16, 16, 32)
        x = self.pixel_shuffle(x, upscale_factor=2)  # (N, 32, 32, 8)

        x = self.conv2d(x, 16, 3, 1, scope='conv3')  # (N, 32, 32, 16)
        x = tf.nn.relu(x)
        x = self.conv2d(x, 16, 3, 1, scope='conv4')  # (N, 32, 32, 16)
        x = self.pixel_shuffle(x, upscale_factor=2)  # (N, 64, 64, 4)

        x = self.conv2d(x, 8, 3, 1, scope='conv5')  # (N, 64, 64, 8)
        x = tf.nn.relu(x)
        x = self.conv2d(x, 4, 3, 1, scope='conv6')  # (N, 64, 64, 4)
        x = self.pixel_shuffle(x, upscale_factor=2)  # (N, 128, 128, 1)
        x = tf.sigmoid(x)

        # (N, 128, 128), [0.0-stroke, 1.0-BG]
        self.stroke_image = 1.0 - tf.reshape(x, (-1, self.raster_size, self.raster_size))

    def conv2d(self, input_tensor, out_channels, kernel_size, stride, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            output_tensor = tf.layers.conv2d(input_tensor, out_channels, kernel_size=kernel_size,
                                             strides=(stride, stride),
                                             padding="same", kernel_initializer=tf.keras.initializers.he_normal())
            return output_tensor

    def fully_connected(self, input_tensor, in_dim, out_dim, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            weight = tf.get_variable("weight", [in_dim, out_dim], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
            bias = tf.get_variable("bias", [out_dim], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer())
            output_tensor = tf.matmul(input_tensor, weight) + bias
            return output_tensor

    def pixel_shuffle(self, input_tensor, upscale_factor):
        params_shape = input_tensor.get_shape()
        n, h, w, c = params_shape
        input_tensor_proc = tf.reshape(input_tensor, (n, h, w, c // 4, 4))
        input_tensor_proc = tf.transpose(input_tensor_proc, (0, 1, 2, 4, 3))
        input_tensor_proc = tf.reshape(input_tensor_proc, (n, h, w, -1))
        output_tensor = tf.depth_to_space(input_tensor_proc, block_size=upscale_factor)
        return output_tensor


class NeuralRasterizor(object):
    def __init__(self,
                 raster_size,
                 seq_len,
                 position_format='abs',
                 raster_padding=10,
                 strokes_format=3):
        self.raster_size = raster_size
        self.seq_len = seq_len
        self.position_format = position_format
        self.raster_padding = raster_padding
        self.strokes_format = strokes_format

        assert position_format in ['abs', 'rel']

    def raster_func_abs(self, input_data, raster_seq_len=None):
        """
        x and y in absolute position.
        :param input_data: (N, seq_len, 10): [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]. All in [0.0, 1.0]
        :return:
        """
        seq_len = raster_seq_len if raster_seq_len is not None else self.seq_len

        raster_params = tf.transpose(input_data, [1, 0, 2])  # (seq_len, N, 10)

        seq_stroke_images = tf.map_fn(self.stroke_drawer_with_raster_unit, raster_params,
                                      parallel_iterations=32)  # (seq_len, N, raster_size, raster_size)
        seq_stroke_images = tf.transpose(seq_stroke_images, [1, 2, 3, 0])
        # (N, raster_size, raster_size, seq_len), [0.0-stroke, 1.0-BG]

        filter_seq_stroke_images = 1.0 - seq_stroke_images
        # (N, raster_size, raster_size, seq_len), [0.0-BG, 1.0-stroke]

        # stacking
        stroke_images_unclip = tf.reduce_sum(filter_seq_stroke_images, axis=-1)  # (N, raster_size, raster_size)
        stroke_images = tf.clip_by_value(stroke_images_unclip, 0.0, 1.0)  # [0.0-BG, 1.0-stroke]
        return stroke_images, stroke_images_unclip, seq_stroke_images

    def stroke_drawer_with_raster_unit(self, params_batch):
        """
        Convert two points into a raster stroke image with RasterUnit.
        :param params_batch: (N, 10)
        :return: (N, raster_size, raster_size)
        """
        raster_unit = RasterUnit(
            raster_size=self.raster_size,
            input_params=params_batch,
            reuse=tf.AUTO_REUSE
        )
        stroke_image = raster_unit.stroke_image  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        return stroke_image


class NeuralRasterizorStep(object):
    def __init__(self,
                 raster_size,
                 position_format='abs'):
        self.raster_size = raster_size
        self.position_format = position_format

        assert position_format in ['abs', 'rel']

    def raster_func_stroke_abs(self, input_data):
        """
        x and y in absolute position.
        :param input_data: (N, 8): [x0, y0, x1, y1, x2, y2, r0, r2]. All in [0.0, 1.0]
        :return:
        """
        w_in = tf.ones(shape=(input_data.shape[0], 2), dtype=tf.float32)
        raster_params = tf.concat([input_data, w_in], axis=-1)  # (N, 10)
        stroke_image = self.stroke_drawer_with_raster_unit(raster_params)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        stroke_image = 1.0 - stroke_image  # [0.0-BG, 1.0-stroke]

        return stroke_image

    def mask_ending_state(self, input_states):
        """
        Mask the ending state to be 1
        :param input_states: (N, seq_len, 1) in offset manner
        :param seq_len:
        :return:
        """
        ending_state_accu = tf.cumsum(input_states, axis=1)  # (N, seq_len, 1)
        ending_state_clip = tf.clip_by_value(ending_state_accu, 0.0, 1.0)  # (N, seq_len, 1)
        return ending_state_clip

    def stroke_drawer_with_raster_unit(self, params_batch):
        """
        Convert two points into a raster stroke image with RasterUnit.
        :param params_batch: (N, 10)
        :return: (N, raster_size, raster_size)
        """
        raster_unit = RasterUnit(
            raster_size=self.raster_size,
            input_params=params_batch,
            reuse=tf.AUTO_REUSE
        )
        stroke_image = raster_unit.stroke_image  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        return stroke_image
