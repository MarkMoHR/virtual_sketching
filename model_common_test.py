import rnn
import tensorflow as tf

from subnet_tf_utils import generative_cnn_encoder, generative_cnn_encoder_deeper, generative_cnn_encoder_deeper13, \
    generative_cnn_c3_encoder, generative_cnn_c3_encoder_deeper, generative_cnn_c3_encoder_deeper13, \
    generative_cnn_c3_encoder_combine33, generative_cnn_c3_encoder_combine43, \
    generative_cnn_c3_encoder_combine53, generative_cnn_c3_encoder_combineFC, \
    generative_cnn_c3_encoder_deeper13_attn


class DiffPastingV3(object):
    def __init__(self, raster_size):
        self.patch_canvas = tf.placeholder(dtype=tf.float32,
                                           shape=(None, None, 1))  # (raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        self.cursor_pos_a = tf.placeholder(dtype=tf.float32, shape=(2))  # (2), float32, in large size
        self.image_size_a = tf.placeholder(dtype=tf.int32, shape=())  # ()
        self.window_size_a = tf.placeholder(dtype=tf.float32, shape=())  # (), float32, with grad
        self.raster_size_a = float(raster_size)

        self.pasted_image = self.image_pasting_sampling_v3()
        # (image_size, image_size, 1), [0.0-BG, 1.0-stroke]

    def image_pasting_sampling_v3(self):
        padding_size = tf.cast(tf.ceil(self.window_size_a / 2.0), tf.int32)

        x1y1_a = self.cursor_pos_a - self.window_size_a / 2.0  # (2), float32
        x2y2_a = self.cursor_pos_a + self.window_size_a / 2.0  # (2), float32

        x1y1_a_floor = tf.floor(x1y1_a)  # (2)
        x2y2_a_ceil = tf.ceil(x2y2_a)  # (2)

        cursor_pos_b_oricoord = (x1y1_a_floor + x2y2_a_ceil) / 2.0  # (2)
        cursor_pos_b = (cursor_pos_b_oricoord - x1y1_a) / self.window_size_a * self.raster_size_a  # (2)
        raster_size_b = (x2y2_a_ceil - x1y1_a_floor)  # (x, y)
        image_size_b = self.raster_size_a
        window_size_b = self.raster_size_a * (raster_size_b / self.window_size_a)  # (x, y)

        cursor_b_x, cursor_b_y = tf.split(cursor_pos_b, 2, axis=-1)  # (1)

        y1_b = cursor_b_y - (window_size_b[1] - 1.) / 2.
        x1_b = cursor_b_x - (window_size_b[0] - 1.) / 2.
        y2_b = y1_b + (window_size_b[1] - 1.)
        x2_b = x1_b + (window_size_b[0] - 1.)
        boxes_b = tf.concat([y1_b, x1_b, y2_b, x2_b], axis=-1)  # (4)
        boxes_b = boxes_b / tf.cast(image_size_b - 1, tf.float32)  # with grad to window_size_a

        box_ind_b = tf.ones((1), dtype=tf.int32)  # (1)
        box_ind_b = tf.cumsum(box_ind_b) - 1

        patch_canvas = tf.expand_dims(self.patch_canvas,
                                      axis=0)  # (1, raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        boxes_b = tf.expand_dims(boxes_b, axis=0)  # (1, 4)

        valid_canvas = tf.image.crop_and_resize(patch_canvas, boxes_b, box_ind_b,
                                                crop_size=[raster_size_b[1], raster_size_b[0]])
        valid_canvas = valid_canvas[0]  # (raster_size_b, raster_size_b, 1)

        pad_up = tf.cast(x1y1_a_floor[1], tf.int32) + padding_size
        pad_down = self.image_size_a + padding_size - tf.cast(x2y2_a_ceil[1], tf.int32)
        pad_left = tf.cast(x1y1_a_floor[0], tf.int32) + padding_size
        pad_right = self.image_size_a + padding_size - tf.cast(x2y2_a_ceil[0], tf.int32)

        paddings = [[pad_up, pad_down],
                    [pad_left, pad_right],
                    [0, 0]]
        pad_img = tf.pad(valid_canvas, paddings=paddings, mode='CONSTANT',
                         constant_values=0.0)  # (H_p, W_p, 1), [0.0-BG, 1.0-stroke]

        pasted_image = pad_img[padding_size: padding_size + self.image_size_a,
                       padding_size: padding_size + self.image_size_a, :]
        # (image_size, image_size, 1), [0.0-BG, 1.0-stroke]
        return pasted_image


class VirtualSketchingModel(object):
    def __init__(self, hps, gpu_mode=True, reuse=False):
        """Initializer for the model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
        self.hps = hps
        assert hps.model_mode in ['train', 'eval', 'eval_sample', 'sample']
        # with tf.variable_scope('SCC', reuse=reuse):
        if not gpu_mode:
            with tf.device('/cpu:0'):
                print('Model using cpu.')
                self.build_model()
        else:
            print('-' * 100)
            print('model_mode:', hps.model_mode)
            print('Model using gpu.')
            self.build_model()

    def build_model(self):
        """Define model architecture."""
        self.config_model()

        initial_state = self.get_decoder_inputs()
        self.initial_state = initial_state

        ## use pred as the prev points
        other_params, pen_ras, final_state = self.get_points_and_raster_image(self.image_size)
        # other_params: (N * max_seq_len, 6)
        # pen_ras: (N * max_seq_len, 2), after softmax

        self.other_params = other_params  # (N * max_seq_len, 6)
        self.pen_ras = pen_ras  # (N * max_seq_len, 2), after softmax
        self.final_state = final_state

        if not self.hps.use_softargmax:
            pen_state_soft = pen_ras[:, 1:2]  # (N * max_seq_len, 1)
        else:
            pen_state_soft = self.differentiable_argmax(pen_ras, self.hps.soft_beta)  # (N * max_seq_len, 1)

        pred_params = tf.concat([pen_state_soft, other_params], axis=1)  # (N * max_seq_len, 7)
        self.pred_params = tf.reshape(pred_params, shape=[-1, self.hps.max_seq_len, 7])  # (N, max_seq_len, 7)
        # pred_params: (N, max_seq_len, 7)

    def config_model(self):
        if self.hps.model_mode == 'train':
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.hps.dec_model == 'lstm':
            dec_cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            dec_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            dec_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        use_recurrent_dropout = self.hps.use_recurrent_dropout
        use_input_dropout = self.hps.use_input_dropout
        use_output_dropout = self.hps.use_output_dropout

        dec_cell = dec_cell_fn(
            self.hps.dec_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        # dropout:
        # print('Input dropout mode = %s.' % use_input_dropout)
        # print('Output dropout mode = %s.' % use_output_dropout)
        # print('Recurrent dropout mode = %s.' % use_recurrent_dropout)
        if use_input_dropout:
            print('Dropout to input w/ keep_prob = %4.4f.' % self.hps.input_dropout_prob)
            dec_cell = tf.contrib.rnn.DropoutWrapper(
                dec_cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            print('Dropout to output w/ keep_prob = %4.4f.' % self.hps.output_dropout_prob)
            dec_cell = tf.contrib.rnn.DropoutWrapper(
                dec_cell, output_keep_prob=self.hps.output_dropout_prob)
        self.dec_cell = dec_cell

        self.input_photo = tf.placeholder(dtype=tf.float32,
                                          shape=[self.hps.batch_size, None, None, self.hps.input_channel])  # [0.0-stroke, 1.0-BG]
        self.init_cursor = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, 1, 2])  # (N, 1, 2), in size [0.0, 1.0)
        self.init_width = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size])  # (1), in [0.0, 1.0]
        self.init_scaling = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size])  # (N), in [0.0, 1.0]
        self.init_window_size = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size])  # (N)
        self.image_size = tf.placeholder(dtype=tf.int32, shape=())  # ()

    ###########################

    def normalize_image_m1to1(self, in_img_0to1):
        norm_img_m1to1 = tf.multiply(in_img_0to1, 2.0)
        norm_img_m1to1 = tf.subtract(norm_img_m1to1, 1.0)
        return norm_img_m1to1

    def add_coords(self, input_tensor):
        batch_size_tensor = tf.shape(input_tensor)[0]  # get N size

        xx_ones = tf.ones([batch_size_tensor, self.hps.raster_size], dtype=tf.int32)  # e.g. (N, raster_size)
        xx_ones = tf.expand_dims(xx_ones, -1)  # e.g. (N, raster_size, 1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.hps.raster_size), 0),
                           [batch_size_tensor, 1])  # e.g. (N, raster_size)
        xx_range = tf.expand_dims(xx_range, 1)  # e.g. (N, 1, raster_size)

        xx_channel = tf.matmul(xx_ones, xx_range)  # e.g. (N, raster_size, raster_size)
        xx_channel = tf.expand_dims(xx_channel, -1)  # e.g. (N, raster_size, raster_size, 1)

        yy_ones = tf.ones([batch_size_tensor, self.hps.raster_size], dtype=tf.int32)  # e.g. (N, raster_size)
        yy_ones = tf.expand_dims(yy_ones, 1)  # e.g. (N, 1, raster_size)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.hps.raster_size), 0),
                           [batch_size_tensor, 1])  # (N, raster_size)
        yy_range = tf.expand_dims(yy_range, -1)  # e.g. (N, raster_size, 1)

        yy_channel = tf.matmul(yy_range, yy_ones)  # e.g. (N, raster_size, raster_size)
        yy_channel = tf.expand_dims(yy_channel, -1)  # e.g. (N, raster_size, raster_size, 1)

        xx_channel = tf.cast(xx_channel, 'float32') / (self.hps.raster_size - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.hps.raster_size - 1)
        # xx_channel = xx_channel * 2 - 1  # [-1, 1]
        # yy_channel = yy_channel * 2 - 1

        ret = tf.concat([
            input_tensor,
            xx_channel,
            yy_channel,
        ], axis=-1)  # e.g. (N, raster_size, raster_size, 4)

        return ret

    def build_combined_encoder(self, patch_canvas, patch_photo, entire_canvas, entire_photo, cursor_pos,
                               image_size, window_size):
        """
        :param patch_canvas: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :param patch_photo: (N, raster_size, raster_size, 1/3), [-1.0-stroke, 1.0-BG]
        :param entire_canvas: (N, image_size, image_size, 1), [0.0-stroke, 1.0-BG]
        :param entire_photo: (N, image_size, image_size, 1/3), [0.0-stroke, 1.0-BG]
        :param cursor_pos: (N, 1, 2), in size [0.0, 1.0)
        :param window_size: (N, 1, 1), float, in large size
        :return:
        """
        if self.hps.resize_method == 'BILINEAR':
            resize_method = tf.image.ResizeMethod.BILINEAR
        elif self.hps.resize_method == 'NEAREST_NEIGHBOR':
            resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.hps.resize_method == 'BICUBIC':
            resize_method = tf.image.ResizeMethod.BICUBIC
        elif self.hps.resize_method == 'AREA':
            resize_method = tf.image.ResizeMethod.AREA
        else:
            raise Exception('unknown resize_method', self.hps.resize_method)

        patch_photo = tf.stop_gradient(patch_photo)
        patch_canvas = tf.stop_gradient(patch_canvas)
        cursor_pos = tf.stop_gradient(cursor_pos)
        window_size = tf.stop_gradient(window_size)

        entire_photo_small = tf.stop_gradient(tf.image.resize_images(entire_photo,
                                                                      (self.hps.raster_size, self.hps.raster_size),
                                                                      method=resize_method))
        entire_canvas_small = tf.stop_gradient(tf.image.resize_images(entire_canvas,
                                                                      (self.hps.raster_size, self.hps.raster_size),
                                                                      method=resize_method))
        entire_photo_small = self.normalize_image_m1to1(entire_photo_small)  # [-1.0-stroke, 1.0-BG]
        entire_canvas_small = self.normalize_image_m1to1(entire_canvas_small)  # [-1.0-stroke, 1.0-BG]

        if self.hps.encode_cursor_type == 'value':
            cursor_pos_norm = tf.expand_dims(cursor_pos, axis=1)  # (N, 1, 1, 2)
            cursor_pos_norm = tf.tile(cursor_pos_norm, [1, self.hps.raster_size, self.hps.raster_size, 1])
            cursor_info = cursor_pos_norm
        else:
            raise Exception('Unknown encode_cursor_type', self.hps.encode_cursor_type)

        batch_input_combined = tf.concat([patch_photo, patch_canvas, entire_photo_small, entire_canvas_small, cursor_info],
                                axis=-1)  # [N, raster_size, raster_size, 6/10]
        batch_input_local = tf.concat([patch_photo, patch_canvas], axis=-1)  # [N, raster_size, raster_size, 2/4]
        batch_input_global = tf.concat([entire_photo_small, entire_canvas_small, cursor_info],
                                       axis=-1)  # [N, raster_size, raster_size, 4/6]

        if self.hps.model_mode == 'train':
            is_training = True
            dropout_keep_prob = self.hps.pix_drop_kp
        else:
            is_training = False
            dropout_keep_prob = 1.0

        if self.hps.add_coordconv:
            batch_input_combined = self.add_coords(batch_input_combined)  # (N, in_H, in_W, in_dim + 2)
            batch_input_local = self.add_coords(batch_input_local)  # (N, in_H, in_W, in_dim + 2)
            batch_input_global = self.add_coords(batch_input_global)  # (N, in_H, in_W, in_dim + 2)

        if 'combine' in self.hps.encoder_type:
            if self.hps.encoder_type == 'combine33':
                image_embedding, _ = generative_cnn_c3_encoder_combine33(batch_input_local, batch_input_global,
                                                                         is_training, dropout_keep_prob)  # (N, 128)
            elif self.hps.encoder_type == 'combine43':
                image_embedding, _ = generative_cnn_c3_encoder_combine43(batch_input_local, batch_input_global,
                                                                         is_training, dropout_keep_prob)  # (N, 128)
            elif self.hps.encoder_type == 'combine53':
                image_embedding, _ = generative_cnn_c3_encoder_combine53(batch_input_local, batch_input_global,
                                                                         is_training, dropout_keep_prob)  # (N, 128)
            elif self.hps.encoder_type == 'combineFC':
                image_embedding, _ = generative_cnn_c3_encoder_combineFC(batch_input_local, batch_input_global,
                                                                         is_training, dropout_keep_prob)  # (N, 256)
            else:
                raise Exception('Unknown encoder_type', self.hps.encoder_type)
        else:
            with tf.variable_scope('Combined_Encoder', reuse=tf.AUTO_REUSE):
                if self.hps.encoder_type == 'conv10':
                    image_embedding, _ = generative_cnn_encoder(batch_input_combined, is_training, dropout_keep_prob)  # (N, 128)
                elif self.hps.encoder_type == 'conv10_deep':
                    image_embedding, _ = generative_cnn_encoder_deeper(batch_input_combined, is_training, dropout_keep_prob)  # (N, 512)
                elif self.hps.encoder_type == 'conv13':
                    image_embedding, _ = generative_cnn_encoder_deeper13(batch_input_combined, is_training, dropout_keep_prob)  # (N, 128)
                elif self.hps.encoder_type == 'conv10_c3':
                    image_embedding, _ = generative_cnn_c3_encoder(batch_input_combined, is_training, dropout_keep_prob)  # (N, 128)
                elif self.hps.encoder_type == 'conv10_deep_c3':
                    image_embedding, _ = generative_cnn_c3_encoder_deeper(batch_input_combined, is_training, dropout_keep_prob)  # (N, 512)
                elif self.hps.encoder_type == 'conv13_c3':
                    image_embedding, _ = generative_cnn_c3_encoder_deeper13(batch_input_combined, is_training, dropout_keep_prob)  # (N, 128)
                elif self.hps.encoder_type == 'conv13_c3_attn':
                    image_embedding, _ = generative_cnn_c3_encoder_deeper13_attn(batch_input_combined, is_training, dropout_keep_prob)  # (N, 128)
                else:
                    raise Exception('Unknown encoder_type', self.hps.encoder_type)
        return image_embedding

    def build_seq_decoder(self, dec_cell, actual_input_x, initial_state):
        rnn_output, last_state = self.rnn_decoder(dec_cell, initial_state, actual_input_x)
        rnn_output_flat = tf.reshape(rnn_output, [-1, self.hps.dec_rnn_size])

        pen_n_out = 2
        params_n_out = 6

        with tf.variable_scope('DEC_RNN_out_pen', reuse=tf.AUTO_REUSE):
            output_w_pen = tf.get_variable('output_w', [self.hps.dec_rnn_size, pen_n_out])
            output_b_pen = tf.get_variable('output_b', [pen_n_out], initializer=tf.constant_initializer(0.0))
            output_pen = tf.nn.xw_plus_b(rnn_output_flat, output_w_pen, output_b_pen)  # (N, pen_n_out)

        with tf.variable_scope('DEC_RNN_out_params', reuse=tf.AUTO_REUSE):
            output_w_params = tf.get_variable('output_w', [self.hps.dec_rnn_size, params_n_out])
            output_b_params = tf.get_variable('output_b', [params_n_out], initializer=tf.constant_initializer(0.0))
            output_params = tf.nn.xw_plus_b(rnn_output_flat, output_w_params, output_b_params)  # (N, params_n_out)

        output = tf.concat([output_pen, output_params], axis=1)  # (N, n_out)

        return output, last_state

    def get_mixture_coef(self, outputs):
        z = outputs
        z_pen_logits = z[:, 0:2]  # (N, 2), pen states
        z_other_params_logits = z[:, 2:]  # (N, 6)

        z_pen = tf.nn.softmax(z_pen_logits)  # (N, 2)
        if self.hps.position_format == 'abs':
            x1y1 = tf.nn.sigmoid(z_other_params_logits[:, 0:2])  # (N, 2)
            x2y2 = tf.tanh(z_other_params_logits[:, 2:4])  # (N, 2)
            widths = tf.nn.sigmoid(z_other_params_logits[:, 4:5])  # (N, 1)
            widths = tf.add(tf.multiply(widths, 1.0 - self.hps.min_width), self.hps.min_width)
            scaling = tf.nn.sigmoid(z_other_params_logits[:, 5:6]) * self.hps.max_scaling  # (N, 1), [0.0, max_scaling]
            # scaling = tf.add(tf.multiply(scaling, (self.hps.max_scaling - self.hps.min_scaling) / self.hps.max_scaling),
            #                  self.hps.min_scaling)
            z_other_params = tf.concat([x1y1, x2y2, widths, scaling], axis=-1)  # (N, 6)
        else:  # "rel"
            raise Exception('Unknown position_format', self.hps.position_format)

        r = [z_other_params, z_pen]
        return r

    ###########################

    def get_decoder_inputs(self):
        initial_state = self.dec_cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
        return initial_state

    def rnn_decoder(self, dec_cell, initial_state, actual_input_x):
        with tf.variable_scope("RNN_DEC", reuse=tf.AUTO_REUSE):
            output, last_state = tf.nn.dynamic_rnn(
                dec_cell,
                actual_input_x,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32)
        return output, last_state

    ###########################

    def image_padding(self, ori_image, window_size, pad_value):
        """
        Pad with (bg)
        :param ori_image:
        :return:
        """
        paddings = [[0, 0],
                    [window_size // 2, window_size // 2],
                    [window_size // 2, window_size // 2],
                    [0, 0]]
        pad_img = tf.pad(ori_image, paddings=paddings, mode='CONSTANT', constant_values=pad_value)  # (N, H_p, W_p, k)
        return pad_img

    def image_cropping_fn(self, fn_inputs):
        """
        crop the patch
        :return:
        """
        index_offset = self.hps.input_channel - 1
        input_image = fn_inputs[:, :, 0:2 + index_offset]  # (image_size, image_size, -), [0.0-BG, 1.0-stroke]
        cursor_pos = fn_inputs[0, 0, 2 + index_offset:4 + index_offset]  # (2), in [0.0, 1.0)
        image_size = fn_inputs[0, 0, 4 + index_offset]  # (), float32
        window_size = tf.cast(fn_inputs[0, 0, 5 + index_offset], tf.int32)  # ()

        input_img_reshape = tf.expand_dims(input_image, axis=0)
        pad_img = self.image_padding(input_img_reshape, window_size, pad_value=0.0)

        cursor_pos = tf.cast(tf.round(tf.multiply(cursor_pos, image_size)), dtype=tf.int32)
        x0, x1 = cursor_pos[0], cursor_pos[0] + window_size  # ()
        y0, y1 = cursor_pos[1], cursor_pos[1] + window_size  # ()
        patch_image = pad_img[:, y0:y1, x0:x1, :]  # (1, window_size, window_size, 2/4)

        # resize to raster_size
        patch_image_scaled = tf.image.resize_images(patch_image, (self.hps.raster_size, self.hps.raster_size),
                                                    method=tf.image.ResizeMethod.AREA)
        patch_image_scaled = tf.squeeze(patch_image_scaled, axis=0)
        # patch_canvas_scaled: (raster_size, raster_size, 2/4), [0.0-BG, 1.0-stroke]

        return patch_image_scaled

    def image_cropping(self, cursor_position, input_img, image_size, window_sizes):
        """
        :param cursor_position: (N, 1, 2), float type, in size [0.0, 1.0)
        :param input_img: (N, image_size, image_size, 2/4), [0.0-BG, 1.0-stroke]
        :param window_sizes: (N, 1, 1), float32, with grad
        """
        input_img_ = input_img
        window_sizes_non_grad = tf.stop_gradient(tf.round(window_sizes))  # (N, 1, 1), no grad

        cursor_position_ = tf.reshape(cursor_position, (-1, 1, 1, 2))  # (N, 1, 1, 2)
        cursor_position_ = tf.tile(cursor_position_, [1, image_size, image_size, 1])  # (N, image_size, image_size, 2)

        image_size_ = tf.reshape(tf.cast(image_size, tf.float32), (1, 1, 1, 1))  # (1, 1, 1, 1)
        image_size_ = tf.tile(image_size_, [self.hps.batch_size, image_size, image_size, 1])

        window_sizes_ = tf.reshape(window_sizes_non_grad, (-1, 1, 1, 1))  # (N, 1, 1, 1)
        window_sizes_ = tf.tile(window_sizes_, [1, image_size, image_size, 1])  # (N, image_size, image_size, 1)

        fn_inputs = tf.concat([input_img_, cursor_position_, image_size_, window_sizes_],
                              axis=-1)  # (N, image_size, image_size, 2/4 + 4)
        curr_patch_imgs = tf.map_fn(self.image_cropping_fn, fn_inputs, parallel_iterations=32)  # (N, raster_size, raster_size, -)
        return curr_patch_imgs

    def image_cropping_v3(self, cursor_position, input_img, image_size, window_sizes):
        """
        :param cursor_position: (N, 1, 2), float type, in size [0.0, 1.0)
        :param input_img: (N, image_size, image_size, k), [0.0-BG, 1.0-stroke]
        :param window_sizes: (N, 1, 1), float32, with grad
        """
        window_sizes_non_grad = tf.stop_gradient(window_sizes)  # (N, 1, 1), no grad

        cursor_pos = tf.multiply(cursor_position, tf.cast(image_size, tf.float32))
        cursor_x, cursor_y = tf.split(cursor_pos, 2, axis=-1)  # (N, 1, 1)

        y1 = cursor_y - (window_sizes_non_grad - 1.0) / 2
        x1 = cursor_x - (window_sizes_non_grad - 1.0) / 2
        y2 = y1 + (window_sizes_non_grad - 1.0)
        x2 = x1 + (window_sizes_non_grad - 1.0)
        boxes = tf.concat([y1, x1, y2, x2], axis=-1)  # (N, 1, 4)
        boxes = tf.squeeze(boxes, axis=1)  # (N, 4)
        boxes = boxes / tf.cast(image_size - 1, tf.float32)

        box_ind = tf.ones_like(cursor_x)[:, 0, 0]  # (N)
        box_ind = tf.cast(box_ind, dtype=tf.int32)
        box_ind = tf.cumsum(box_ind) - 1

        curr_patch_imgs = tf.image.crop_and_resize(input_img, boxes, box_ind,
                                                   crop_size=[self.hps.raster_size, self.hps.raster_size])
        #  (N, raster_size, raster_size, k), [0.0-BG, 1.0-stroke]
        return curr_patch_imgs

    def get_points_and_raster_image(self, image_size):
        ## generate the other_params and pen_ras and raster image for raster loss
        prev_state = self.initial_state  # (N, dec_rnn_size * 3)

        prev_width = self.init_width  # (N)
        prev_width = tf.expand_dims(tf.expand_dims(prev_width, axis=-1), axis=-1)  # (N, 1, 1)

        prev_scaling = self.init_scaling  # (N)
        prev_scaling = tf.reshape(prev_scaling, (-1, 1, 1))  # (N, 1, 1)

        prev_window_size = self.init_window_size  # (N)
        prev_window_size = tf.reshape(prev_window_size, (-1, 1, 1))  # (N, 1, 1)

        cursor_position_temp = self.init_cursor
        self.cursor_position = cursor_position_temp  # (N, 1, 2), in size [0.0, 1.0)
        cursor_position_loop = self.cursor_position

        other_params_list = []
        pen_ras_list = []

        curr_canvas_soft = tf.zeros_like(self.input_photo[:, :, :, 0])  # (N, image_size, image_size), [0.0-BG, 1.0-stroke]
        curr_canvas_hard = tf.zeros_like(curr_canvas_soft)  # [0.0-BG, 1.0-stroke]

        #### sampling part - start ####
        self.curr_canvas_hard = curr_canvas_hard

        if self.hps.cropping_type == 'v3':
            cropping_func = self.image_cropping_v3
        # elif self.hps.cropping_type == 'v2':
        #     cropping_func = self.image_cropping
        else:
            raise Exception('Unknown cropping_type', self.hps.cropping_type)

        for time_i in range(self.hps.max_seq_len):
            cursor_position_non_grad = tf.stop_gradient(cursor_position_loop)  # (N, 1, 2), in size [0.0, 1.0)

            curr_window_size = tf.multiply(prev_scaling, tf.stop_gradient(prev_window_size))  # float, with grad
            curr_window_size = tf.maximum(curr_window_size, tf.cast(self.hps.min_window_size, tf.float32))
            curr_window_size = tf.minimum(curr_window_size, tf.cast(image_size, tf.float32))

            ## patch-level encoding
            # Here, we make the gradients from canvas_z to curr_canvas_hard be None to avoid recurrent gradient propagation.
            curr_canvas_hard_non_grad = tf.stop_gradient(self.curr_canvas_hard)
            curr_canvas_hard_non_grad = tf.expand_dims(curr_canvas_hard_non_grad, axis=-1)

            # input_photo: (N, image_size, image_size, 1/3), [0.0-stroke, 1.0-BG]
            crop_inputs = tf.concat([1.0 - self.input_photo, curr_canvas_hard_non_grad], axis=-1)  # (N, H_p, W_p, 1+1)

            cropped_outputs = cropping_func(cursor_position_non_grad, crop_inputs, image_size, curr_window_size)
            index_offset = self.hps.input_channel - 1
            curr_patch_inputs = cropped_outputs[:, :, :, 0:1 + index_offset]  # [0.0-BG, 1.0-stroke]
            curr_patch_canvas_hard_non_grad = cropped_outputs[:, :, :, 1 + index_offset:2 + index_offset]
            # (N, raster_size, raster_size, 1/3), [0.0-BG, 1.0-stroke]

            curr_patch_inputs = 1.0 - curr_patch_inputs  # [0.0-stroke, 1.0-BG]
            curr_patch_inputs = self.normalize_image_m1to1(curr_patch_inputs)
            # (N, raster_size, raster_size, 1/3), [-1.0-stroke, 1.0-BG]

            # Normalizing image
            curr_patch_canvas_hard_non_grad = 1.0 - curr_patch_canvas_hard_non_grad  # [0.0-stroke, 1.0-BG]
            curr_patch_canvas_hard_non_grad = self.normalize_image_m1to1(curr_patch_canvas_hard_non_grad)  # [-1.0-stroke, 1.0-BG]

            ## image-level encoding
            combined_z = self.build_combined_encoder(
                curr_patch_canvas_hard_non_grad,
                curr_patch_inputs,
                1.0 - curr_canvas_hard_non_grad,
                self.input_photo,
                cursor_position_non_grad,
                image_size,
                curr_window_size)  # (N, z_size)
            combined_z = tf.expand_dims(combined_z, axis=1)  # (N, 1, z_size)

            curr_window_size_top_side_norm_non_grad = \
                tf.stop_gradient(curr_window_size / tf.cast(image_size, tf.float32))
            curr_window_size_bottom_side_norm_non_grad = \
                tf.stop_gradient(curr_window_size / tf.cast(self.hps.min_window_size, tf.float32))
            if not self.hps.concat_win_size:
                combined_z = tf.concat([tf.stop_gradient(prev_width), combined_z], 2)  # (N, 1, 2+z_size)
            else:
                combined_z = tf.concat([tf.stop_gradient(prev_width),
                                        curr_window_size_top_side_norm_non_grad,
                                        curr_window_size_bottom_side_norm_non_grad,
                                        combined_z],
                                       2)  # (N, 1, 2+z_size)

            if self.hps.concat_cursor:
                prev_input_x = tf.concat([cursor_position_non_grad, combined_z], 2)  # (N, 1, 2+2+z_size)
            else:
                prev_input_x = combined_z  # (N, 1, 2+z_size)

            h_output, next_state = self.build_seq_decoder(self.dec_cell, prev_input_x, prev_state)
            # h_output: (N * 1, n_out), next_state: (N, dec_rnn_size * 3)
            [o_other_params, o_pen_ras] = self.get_mixture_coef(h_output)
            # o_other_params: (N * 1, 6)
            # o_pen_ras: (N * 1, 2), after softmax

            o_other_params = tf.reshape(o_other_params, [-1, 1, 6])  # (N, 1, 6)
            o_pen_ras_raw = tf.reshape(o_pen_ras, [-1, 1, 2])  # (N, 1, 2)

            other_params_list.append(o_other_params)
            pen_ras_list.append(o_pen_ras_raw)

            #### sampling part - end ####

            prev_state = next_state

        other_params_ = tf.reshape(tf.concat(other_params_list, axis=1), [-1, 6])  # (N * max_seq_len, 6)
        pen_ras_ = tf.reshape(tf.concat(pen_ras_list, axis=1), [-1, 2])  # (N * max_seq_len, 2)

        return other_params_, pen_ras_, prev_state

    def differentiable_argmax(self, input_pen, soft_beta):
        """
        Differentiable argmax trick.
        :param input_pen: (N, n_class)
        :return: pen_state: (N, 1)
        """
        def sign_onehot(x):
            """
            :param x: (N, n_class)
            :return:  (N, n_class)
            """
            y = tf.sign(tf.reduce_max(x, axis=-1, keepdims=True) - x)
            y = (y - 1) * (-1)
            return y

        def softargmax(x, beta=1e2):
            """
            :param x: (N, n_class)
            :param beta: 1e10 is the best. 1e2 is acceptable.
            :return:  (N)
            """
            x_range = tf.cumsum(tf.ones_like(x), axis=1)  # (N, 2)
            return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=1) - 1

        ## Better to use softargmax(beta=1e2). The sign_onehot's gradient is close to zero.
        # pen_onehot = sign_onehot(input_pen)  # one-hot form, (N * max_seq_len, 2)
        # pen_state = pen_onehot[:, 1:2]  # (N * max_seq_len, 1)
        pen_state = softargmax(input_pen, soft_beta)
        pen_state = tf.expand_dims(pen_state, axis=1)  # (N * max_seq_len, 1)
        return pen_state
