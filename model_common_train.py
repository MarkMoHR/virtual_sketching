import rnn
import tensorflow as tf

from subnet_tf_utils import generative_cnn_encoder, generative_cnn_encoder_deeper, generative_cnn_encoder_deeper13, \
    generative_cnn_c3_encoder, generative_cnn_c3_encoder_deeper, generative_cnn_c3_encoder_deeper13, \
    generative_cnn_c3_encoder_combine33, generative_cnn_c3_encoder_combine43, \
    generative_cnn_c3_encoder_combine53, generative_cnn_c3_encoder_combineFC, \
    generative_cnn_c3_encoder_deeper13_attn
from rasterization_utils.NeuralRenderer import NeuralRasterizorStep
from vgg_utils.VGG16 import vgg_net_slim


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
        self.initial_state_list = tf.split(self.initial_state, self.total_loop, axis=0)

        total_loss_list = []
        ras_loss_list = []
        perc_relu_raw_list = []
        perc_relu_norm_list = []
        sn_loss_list = []
        cursor_outside_loss_list = []
        win_size_outside_loss_list = []
        early_state_loss_list = []

        tower_grads = []

        pred_raster_imgs_list = []
        pred_raster_imgs_rgb_list = []

        for t_i in range(self.total_loop):
            gpu_idx = t_i // self.hps.loop_per_gpu
            gpu_i = self.hps.gpus[gpu_idx]
            print(self.hps.model_mode, 'model, gpu:', gpu_i, ', loop:', t_i % self.hps.loop_per_gpu)
            with tf.device('/gpu:%d' % gpu_i):
                with tf.name_scope('GPU_%d' % gpu_i) as scope:
                    if t_i > 0:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        total_loss_list.clear()
                        ras_loss_list.clear()
                        perc_relu_raw_list.clear()
                        perc_relu_norm_list.clear()
                        sn_loss_list.clear()
                        cursor_outside_loss_list.clear()
                        win_size_outside_loss_list.clear()
                        early_state_loss_list.clear()
                        tower_grads.clear()
                        pred_raster_imgs_list.clear()
                        pred_raster_imgs_rgb_list.clear()

                    split_input_photo = self.input_photo_list[t_i]
                    split_image_size = self.image_size[t_i]
                    split_init_cursor = self.init_cursor_list[t_i]
                    split_initial_state = self.initial_state_list[t_i]
                    if self.hps.input_channel == 1:
                        split_target_sketch = split_input_photo
                    else:
                        split_target_sketch = self.target_sketch_list[t_i]

                    ## use pred as the prev points
                    other_params, pen_ras, final_state, pred_raster_images, pred_raster_images_rgb, \
                    pos_before_max_min, win_size_before_max_min \
                        = self.get_points_and_raster_image(split_initial_state, split_init_cursor, split_input_photo,
                                                           split_image_size)
                    # other_params: (N * max_seq_len, 6)
                    # pen_ras: (N * max_seq_len, 2), after softmax
                    # pos_before_max_min: (N, max_seq_len, 2), in image_size
                    # win_size_before_max_min: (N, max_seq_len, 1), in image_size

                    pred_raster_imgs = 1.0 - pred_raster_images  # (N, image_size, image_size), [0.0-stroke, 1.0-BG]
                    pred_raster_imgs_rgb = 1.0 - pred_raster_images_rgb  # (N, image_size, image_size, 3)
                    pred_raster_imgs_list.append(pred_raster_imgs)
                    pred_raster_imgs_rgb_list.append(pred_raster_imgs_rgb)

                    if not self.hps.use_softargmax:
                        pen_state_soft = pen_ras[:, 1:2]  # (N * max_seq_len, 1)
                    else:
                        pen_state_soft = self.differentiable_argmax(pen_ras, self.hps.soft_beta)  # (N * max_seq_len, 1)

                    pred_params = tf.concat([pen_state_soft, other_params], axis=1)  # (N * max_seq_len, 7)
                    pred_params = tf.reshape(pred_params, shape=[-1, self.hps.max_seq_len, 7])  # (N, max_seq_len, 7)
                    # pred_params: (N, max_seq_len, 7)

                    if self.hps.model_mode == 'train' or self.hps.model_mode == 'eval':
                        raster_cost, sn_cost, cursor_outside_cost, winsize_outside_cost, \
                        early_pen_states_cost, \
                        perc_relu_loss_raw, perc_relu_loss_norm = \
                            self.build_losses(split_target_sketch, pred_raster_imgs, pred_params,
                                              pos_before_max_min, win_size_before_max_min,
                                              split_image_size)
                        # perc_relu_loss_raw, perc_relu_loss_norm: (n_layers)

                        ras_loss_list.append(raster_cost)
                        perc_relu_raw_list.append(perc_relu_loss_raw)
                        perc_relu_norm_list.append(perc_relu_loss_norm)
                        sn_loss_list.append(sn_cost)
                        cursor_outside_loss_list.append(cursor_outside_cost)
                        win_size_outside_loss_list.append(winsize_outside_cost)
                        early_state_loss_list.append(early_pen_states_cost)

                        if self.hps.model_mode == 'train':
                            total_cost_split, grads_and_vars_split = self.build_training_op_split(
                                raster_cost, sn_cost, cursor_outside_cost, winsize_outside_cost,
                                early_pen_states_cost)
                            total_loss_list.append(total_cost_split)
                            tower_grads.append(grads_and_vars_split)

        self.raster_cost = tf.reduce_mean(tf.stack(ras_loss_list, axis=0))
        self.perc_relu_losses_raw = tf.reduce_mean(tf.stack(perc_relu_raw_list, axis=0), axis=0)  # (n_layers)
        self.perc_relu_losses_norm = tf.reduce_mean(tf.stack(perc_relu_norm_list, axis=0), axis=0)  # (n_layers)
        self.stroke_num_cost = tf.reduce_mean(tf.stack(sn_loss_list, axis=0))
        self.pos_outside_cost = tf.reduce_mean(tf.stack(cursor_outside_loss_list, axis=0))
        self.win_size_outside_cost = tf.reduce_mean(tf.stack(win_size_outside_loss_list, axis=0))
        self.early_pen_states_cost = tf.reduce_mean(tf.stack(early_state_loss_list, axis=0))
        self.cost = tf.reduce_mean(tf.stack(total_loss_list, axis=0))

        self.pred_raster_imgs = tf.concat(pred_raster_imgs_list, axis=0)  # (N, image_size, image_size), [0.0-stroke, 1.0-BG]
        self.pred_raster_imgs_rgb = tf.concat(pred_raster_imgs_rgb_list, axis=0)  # (N, image_size, image_size, 3)

        if self.hps.model_mode == 'train':
            self.build_training_op(tower_grads)

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

        self.total_loop = len(self.hps.gpus) * self.hps.loop_per_gpu

        self.init_cursor = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, 1, 2])  # (N, 1, 2), in size [0.0, 1.0)
        self.init_width = tf.placeholder(
            dtype=tf.float32,
            shape=[1])  # (1), in [0.0, 1.0]
        self.image_size = tf.placeholder(dtype=tf.int32, shape=(self.total_loop))  # ()

        self.init_cursor_list = tf.split(self.init_cursor, self.total_loop, axis=0)
        self.input_photo_list = []
        for loop_i in range(self.total_loop):
            input_photo_i = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.hps.input_channel])  # [0.0-stroke, 1.0-BG]
            self.input_photo_list.append(input_photo_i)

        if self.hps.input_channel == 3:
            self.target_sketch_list = []
            for loop_i in range(self.total_loop):
                target_sketch_i = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])  # [0.0-stroke, 1.0-BG]
                self.target_sketch_list.append(target_sketch_i)

        if self.hps.model_mode == 'train' or self.hps.model_mode == 'eval':
            self.stroke_num_loss_weight = tf.Variable(0.0, trainable=False)
            self.early_pen_loss_start_idx = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.early_pen_loss_end_idx = tf.Variable(0, dtype=tf.int32, trainable=False)

        if self.hps.model_mode == 'train':
            self.perc_loss_mean_list = []
            for loop_i in range(len(self.hps.perc_loss_layers)):
                relu_loss_mean = tf.Variable(0.0, trainable=False)
                self.perc_loss_mean_list.append(relu_loss_mean)
            self.last_step_num = tf.Variable(0.0, trainable=False)

            with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
                self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)

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
        input_image = fn_inputs[:, :, 0:2 + index_offset]  # (image_size, image_size, 2), [0.0-BG, 1.0-stroke]
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
        image_size_ = tf.tile(image_size_, [self.hps.batch_size // self.total_loop, image_size, image_size, 1])

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

    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate vectors x and y from a  4D tensor image.

        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B, H', W')
        - y: flattened tensor of shape (B, H', W')

        Returns
        -------
        - output: tensor of shape (B, H', W', C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    def image_pasting_nondiff_single(self, fn_inputs):
        patch_image = fn_inputs[:, :, 0:1]  # (raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        cursor_pos = fn_inputs[0, 0, 1:3]  # (2), in large size
        image_size = tf.cast(fn_inputs[0, 0, 3], tf.int32)  # ()
        window_size = tf.cast(fn_inputs[0, 0, 4], tf.int32)  # ()

        patch_image_scaled = tf.expand_dims(patch_image, axis=0)  # (1, raster_size, raster_size, 1)
        patch_image_scaled = tf.image.resize_images(patch_image_scaled, (window_size, window_size),
                                                    method=tf.image.ResizeMethod.BILINEAR)
        patch_image_scaled = tf.squeeze(patch_image_scaled, axis=0)
        # patch_canvas_scaled: (window_size, window_size, 1)

        cursor_pos = tf.cast(tf.round(cursor_pos), dtype=tf.int32)  # (2)
        cursor_x, cursor_y = cursor_pos[0], cursor_pos[1]

        pad_up = cursor_y
        pad_down = image_size - cursor_y
        pad_left = cursor_x
        pad_right = image_size - cursor_x

        paddings = [[pad_up, pad_down],
                    [pad_left, pad_right],
                    [0, 0]]
        pad_img = tf.pad(patch_image_scaled, paddings=paddings, mode='CONSTANT',
                         constant_values=0.0)  # (H_p, W_p, 1), [0.0-BG, 1.0-stroke]

        crop_start = window_size // 2
        pasted_image = pad_img[crop_start: crop_start + image_size, crop_start: crop_start + image_size, :]
        return pasted_image

    def image_pasting_diff_single(self, fn_inputs):
        patch_canvas = fn_inputs[:, :, 0:1]  # (raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        cursor_pos = fn_inputs[0, 0, 1:3]  # (2), in large size
        image_size = tf.cast(fn_inputs[0, 0, 3], tf.int32)  # ()
        window_size = tf.cast(fn_inputs[0, 0, 4], tf.int32)  # ()
        cursor_x, cursor_y = cursor_pos[0], cursor_pos[1]

        patch_canvas_scaled = tf.expand_dims(patch_canvas, axis=0)  # (1, raster_size, raster_size, 1)
        patch_canvas_scaled = tf.image.resize_images(patch_canvas_scaled, (window_size, window_size),
                                                     method=tf.image.ResizeMethod.BILINEAR)
        # patch_canvas_scaled: (1, window_size, window_size, 1)

        valid_canvas = self.image_pasting_diff_batch(patch_canvas_scaled,
                                                     tf.expand_dims(tf.expand_dims(cursor_pos, axis=0), axis=0),
                                                     window_size)
        valid_canvas = tf.squeeze(valid_canvas, axis=0)
        # (window_size + 1, window_size + 1, 1)

        pad_up = tf.cast(tf.floor(cursor_y), tf.int32)
        pad_down = image_size - 1 - tf.cast(tf.floor(cursor_y), tf.int32)
        pad_left = tf.cast(tf.floor(cursor_x), tf.int32)
        pad_right = image_size - 1 - tf.cast(tf.floor(cursor_x), tf.int32)

        paddings = [[pad_up, pad_down],
                    [pad_left, pad_right],
                    [0, 0]]
        pad_img = tf.pad(valid_canvas, paddings=paddings, mode='CONSTANT',
                         constant_values=0.0)  # (H_p, W_p, 1), [0.0-BG, 1.0-stroke]

        crop_start = window_size // 2
        pasted_image = pad_img[crop_start: crop_start + image_size, crop_start: crop_start + image_size, :]
        return pasted_image

    def image_pasting_diff_single_v3(self, fn_inputs):
        patch_canvas = fn_inputs[:, :, 0:1]  # (raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        cursor_pos_a = fn_inputs[0, 0, 1:3]  # (2), float32, in large size
        image_size_a = tf.cast(fn_inputs[0, 0, 3], tf.int32)  # ()
        window_size_a = fn_inputs[0, 0, 4]  # (), float32, with grad
        raster_size_a = float(self.hps.raster_size)

        padding_size = tf.cast(tf.ceil(window_size_a / 2.0), tf.int32)

        x1y1_a = cursor_pos_a - window_size_a / 2.0  # (2), float32
        x2y2_a = cursor_pos_a + window_size_a / 2.0  # (2), float32

        x1y1_a_floor = tf.floor(x1y1_a)  # (2)
        x2y2_a_ceil = tf.ceil(x2y2_a)  # (2)

        cursor_pos_b_oricoord = (x1y1_a_floor + x2y2_a_ceil) / 2.0  # (2)
        cursor_pos_b = (cursor_pos_b_oricoord - x1y1_a) / window_size_a * raster_size_a  # (2)
        raster_size_b = (x2y2_a_ceil - x1y1_a_floor)  # (x, y)
        image_size_b = raster_size_a
        window_size_b = raster_size_a * (raster_size_b / window_size_a)  # (x, y)

        cursor_b_x, cursor_b_y = tf.split(cursor_pos_b, 2, axis=-1)  # (1)

        y1_b = cursor_b_y - (window_size_b[1] - 1.) / 2.
        x1_b = cursor_b_x - (window_size_b[0] - 1.) / 2.
        y2_b = y1_b + (window_size_b[1] - 1.)
        x2_b = x1_b + (window_size_b[0] - 1.)
        boxes_b = tf.concat([y1_b, x1_b, y2_b, x2_b], axis=-1)  # (4)
        boxes_b = boxes_b / tf.cast(image_size_b - 1, tf.float32)  # with grad to window_size_a

        box_ind_b = tf.ones((1), dtype=tf.int32)  # (1)
        box_ind_b = tf.cumsum(box_ind_b) - 1

        patch_canvas = tf.expand_dims(patch_canvas, axis=0)  # (1, raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]
        boxes_b = tf.expand_dims(boxes_b, axis=0)  # (1, 4)

        valid_canvas = tf.image.crop_and_resize(patch_canvas, boxes_b, box_ind_b,
                                                crop_size=[raster_size_b[1], raster_size_b[0]])
        valid_canvas = valid_canvas[0]  # (raster_size_b, raster_size_b, 1)

        pad_up = tf.cast(x1y1_a_floor[1], tf.int32) + padding_size
        pad_down = image_size_a + padding_size - tf.cast(x2y2_a_ceil[1], tf.int32)
        pad_left = tf.cast(x1y1_a_floor[0], tf.int32) + padding_size
        pad_right = image_size_a + padding_size - tf.cast(x2y2_a_ceil[0], tf.int32)

        paddings = [[pad_up, pad_down],
                    [pad_left, pad_right],
                    [0, 0]]
        pad_img = tf.pad(valid_canvas, paddings=paddings, mode='CONSTANT',
                         constant_values=0.0)  # (H_p, W_p, 1), [0.0-BG, 1.0-stroke]

        pasted_image = pad_img[padding_size: padding_size + image_size_a, padding_size: padding_size + image_size_a, :]
        return pasted_image

    def image_pasting_diff_batch(self, patch_image, cursor_position, window_size):
        """
        :param patch_img: (N, window_size, window_size, 1), [0.0-BG, 1.0-stroke]
        :param cursor_position: (N, 1, 2), in large size
        :return:
        """
        paddings1 = [[0, 0],
                     [1, 1],
                     [1, 1],
                     [0, 0]]
        patch_image_pad1 = tf.pad(patch_image, paddings=paddings1, mode='CONSTANT',
                                  constant_values=0.0)  # (N, window_size+2, window_size+2, 1), [0.0-BG, 1.0-stroke]

        cursor_x, cursor_y = cursor_position[:, :, 0:1], cursor_position[:, :, 1:2]  # (N, 1, 1)
        cursor_x_f, cursor_y_f = tf.floor(cursor_x), tf.floor(cursor_y)
        patch_x, patch_y = 1.0 - (cursor_x - cursor_x_f), 1.0 - (cursor_y - cursor_y_f)  # (N, 1, 1)

        x_ones = tf.ones_like(patch_x, dtype=tf.float32)  # (N, 1, 1)
        x_ones = tf.tile(x_ones, [1, 1, window_size])  # (N, 1, window_size)
        patch_x = tf.concat([patch_x, x_ones], axis=-1)  # (N, 1, window_size + 1)
        patch_x = tf.tile(patch_x, [1, window_size + 1, 1])  # (N, window_size + 1, window_size + 1)
        patch_x = tf.cumsum(patch_x, axis=-1)  # (N, window_size + 1, window_size + 1)
        patch_x0 = tf.cast(tf.floor(patch_x), tf.int32)  # (N, window_size + 1, window_size + 1)
        patch_x1 = patch_x0 + 1  # (N, window_size + 1, window_size + 1)

        y_ones = tf.ones_like(patch_y, dtype=tf.float32)  # (N, 1, 1)
        y_ones = tf.tile(y_ones, [1, window_size, 1])  # (N, window_size, 1)
        patch_y = tf.concat([patch_y, y_ones], axis=1)  # (N, window_size + 1, 1)
        patch_y = tf.tile(patch_y, [1, 1, window_size + 1])  # (N, window_size + 1, window_size + 1)
        patch_y = tf.cumsum(patch_y, axis=1)  # (N, window_size + 1, window_size + 1)
        patch_y0 = tf.cast(tf.floor(patch_y), tf.int32)  # (N, window_size + 1, window_size + 1)
        patch_y1 = patch_y0 + 1  # (N, window_size + 1, window_size + 1)

        # get pixel value at corner coords
        valid_canvas_patch_a = self.get_pixel_value(patch_image_pad1, patch_x0, patch_y0)
        valid_canvas_patch_b = self.get_pixel_value(patch_image_pad1, patch_x0, patch_y1)
        valid_canvas_patch_c = self.get_pixel_value(patch_image_pad1, patch_x1, patch_y0)
        valid_canvas_patch_d = self.get_pixel_value(patch_image_pad1, patch_x1, patch_y1)
        # (N, window_size + 1, window_size + 1, 1)

        patch_x0 = tf.cast(patch_x0, tf.float32)
        patch_x1 = tf.cast(patch_x1, tf.float32)
        patch_y0 = tf.cast(patch_y0, tf.float32)
        patch_y1 = tf.cast(patch_y1, tf.float32)

        # calculate deltas
        wa = (patch_x1 - patch_x) * (patch_y1 - patch_y)
        wb = (patch_x1 - patch_x) * (patch_y - patch_y0)
        wc = (patch_x - patch_x0) * (patch_y1 - patch_y)
        wd = (patch_x - patch_x0) * (patch_y - patch_y0)
        # (N, window_size + 1, window_size + 1)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)
        # (N, window_size + 1, window_size + 1, 1)

        # compute output
        valid_canvas_patch_ = tf.add_n([wa * valid_canvas_patch_a,
                                        wb * valid_canvas_patch_b,
                                        wc * valid_canvas_patch_c,
                                        wd * valid_canvas_patch_d])  # (N, window_size + 1, window_size + 1, 1)
        return valid_canvas_patch_

    def image_pasting(self, cursor_position_norm, patch_img, image_size, window_sizes, is_differentiable=False):
        """
        paste the patch_img to padded size based on cursor_position
        :param cursor_position_norm: (N, 1, 2), float type, in size [0.0, 1.0)
        :param patch_img: (N, raster_size, raster_size), [0.0-BG, 1.0-stroke]
        :param window_sizes: (N, 1, 1), float32, with grad
        :return:
        """
        cursor_position = tf.multiply(cursor_position_norm, tf.cast(image_size, tf.float32))  # in large size
        window_sizes_r = tf.round(window_sizes)  # (N, 1, 1), no grad

        patch_img_ = tf.expand_dims(patch_img, axis=-1)  # (N, raster_size, raster_size, 1)
        cursor_position_step = tf.reshape(cursor_position, (-1, 1, 1, 2))  # (N, 1, 1, 2)
        cursor_position_step = tf.tile(cursor_position_step, [1, self.hps.raster_size, self.hps.raster_size,
                                                              1])  # (N, raster_size, raster_size, 2)
        image_size_tile = tf.reshape(tf.cast(image_size, tf.float32), (1, 1, 1, 1))  # (N, 1, 1, 1)
        image_size_tile = tf.tile(image_size_tile, [self.hps.batch_size // self.total_loop, self.hps.raster_size,
                                                    self.hps.raster_size, 1])
        window_sizes_tile = tf.reshape(window_sizes_r, (-1, 1, 1, 1))  # (N, 1, 1, 1)
        window_sizes_tile = tf.tile(window_sizes_tile, [1, self.hps.raster_size, self.hps.raster_size, 1])

        pasting_inputs = tf.concat([patch_img_, cursor_position_step, image_size_tile, window_sizes_tile],
                                   axis=-1)  # (N, raster_size, raster_size, 5)

        if is_differentiable:
            curr_paste_imgs = tf.map_fn(self.image_pasting_diff_single, pasting_inputs,
                                        parallel_iterations=32)  # (N, image_size, image_size, 1)
        else:
            curr_paste_imgs = tf.map_fn(self.image_pasting_nondiff_single, pasting_inputs,
                                        parallel_iterations=32)  # (N, image_size, image_size, 1)
        curr_paste_imgs = tf.squeeze(curr_paste_imgs, axis=-1)  # (N, image_size, image_size)
        return curr_paste_imgs

    def image_pasting_v3(self, cursor_position_norm, patch_img, image_size, window_sizes, is_differentiable=False):
        """
        paste the patch_img to padded size based on cursor_position
        :param cursor_position_norm: (N, 1, 2), float type, in size [0.0, 1.0)
        :param patch_img: (N, raster_size, raster_size), [0.0-BG, 1.0-stroke]
        :param window_sizes: (N, 1, 1), float32, with grad
        :return:
        """
        cursor_position = tf.multiply(cursor_position_norm, tf.cast(image_size, tf.float32))  # in large size

        if is_differentiable:
            patch_img_ = tf.expand_dims(patch_img, axis=-1)  # (N, raster_size, raster_size, 1)
            cursor_position_step = tf.reshape(cursor_position, (-1, 1, 1, 2))  # (N, 1, 1, 2)
            cursor_position_step = tf.tile(cursor_position_step, [1, self.hps.raster_size, self.hps.raster_size,
                                           1])  # (N, raster_size, raster_size, 2)
            image_size_tile = tf.reshape(tf.cast(image_size, tf.float32), (1, 1, 1, 1))  # (N, 1, 1, 1)
            image_size_tile = tf.tile(image_size_tile, [self.hps.batch_size // self.total_loop, self.hps.raster_size,
                                      self.hps.raster_size, 1])
            window_sizes_tile = tf.reshape(window_sizes, (-1, 1, 1, 1))  # (N, 1, 1, 1)
            window_sizes_tile = tf.tile(window_sizes_tile, [1, self.hps.raster_size, self.hps.raster_size, 1])

            pasting_inputs = tf.concat([patch_img_, cursor_position_step, image_size_tile, window_sizes_tile],
                                       axis=-1)  # (N, raster_size, raster_size, 5)
            curr_paste_imgs = tf.map_fn(self.image_pasting_diff_single_v3, pasting_inputs,
                                        parallel_iterations=32)  # (N, image_size, image_size, 1)
        else:
            raise Exception('Unfinished...')
        curr_paste_imgs = tf.squeeze(curr_paste_imgs, axis=-1)  # (N, image_size, image_size)
        return curr_paste_imgs

    def get_points_and_raster_image(self, initial_state, init_cursor, input_photo, image_size):
        ## generate the other_params and pen_ras and raster image for raster loss
        prev_state = initial_state  # (N, dec_rnn_size * 3)

        prev_width = self.init_width  # (1)
        prev_width = tf.expand_dims(tf.expand_dims(prev_width, axis=0), axis=0)  # (1, 1, 1)
        prev_width = tf.tile(prev_width, [self.hps.batch_size // self.total_loop, 1, 1])  # (N, 1, 1)

        prev_scaling = tf.ones((self.hps.batch_size // self.total_loop, 1, 1))  # (N, 1, 1)
        prev_window_size = tf.ones((self.hps.batch_size // self.total_loop, 1, 1),
                                   dtype=tf.float32) * float(self.hps.raster_size)  # (N, 1, 1)

        cursor_position_temp = init_cursor
        self.cursor_position = cursor_position_temp  # (N, 1, 2), in size [0.0, 1.0)
        cursor_position_loop = self.cursor_position

        other_params_list = []
        pen_ras_list = []

        pos_before_max_min_list = []
        win_size_before_max_min_list = []

        curr_canvas_soft = tf.zeros_like(input_photo[:, :, :, 0])  # (N, image_size, image_size), [0.0-BG, 1.0-stroke]
        curr_canvas_soft_rgb = tf.tile(tf.zeros_like(input_photo[:, :, :, 0:1]), [1, 1, 1, 3])  # (N, image_size, image_size, 3), [0.0-BG, 1.0-stroke]
        curr_canvas_hard = tf.zeros_like(curr_canvas_soft)  # [0.0-BG, 1.0-stroke]

        #### sampling part - start ####
        self.curr_canvas_hard = curr_canvas_hard

        rasterizor_st = NeuralRasterizorStep(
            raster_size=self.hps.raster_size,
            position_format=self.hps.position_format)

        if self.hps.cropping_type == 'v3':
            cropping_func = self.image_cropping_v3
        # elif self.hps.cropping_type == 'v2':
        #     cropping_func = self.image_cropping
        else:
            raise Exception('Unknown cropping_type', self.hps.cropping_type)

        if self.hps.pasting_type == 'v3':
            pasting_func = self.image_pasting_v3
        # elif self.hps.pasting_type == 'v2':
        #     pasting_func = self.image_pasting
        else:
            raise Exception('Unknown pasting_type', self.hps.pasting_type)

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
            crop_inputs = tf.concat([1.0 - input_photo, curr_canvas_hard_non_grad], axis=-1)  # (N, H_p, W_p, 1/3+1)

            cropped_outputs = cropping_func(cursor_position_non_grad, crop_inputs, image_size, curr_window_size)
            index_offset = self.hps.input_channel - 1
            curr_patch_inputs = cropped_outputs[:, :, :, 0:1 + index_offset]  # [0.0-BG, 1.0-stroke]
            curr_patch_canvas_hard_non_grad = cropped_outputs[:, :, :, 1 + index_offset:2 + index_offset]
            # (N, raster_size, raster_size, 1), [0.0-BG, 1.0-stroke]

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
                input_photo,
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

            if self.hps.model_mode == 'train' or self.hps.model_mode == 'eval' or self.hps.model_mode == 'eval_sample':
                # use renderer here to convert the strokes to image
                curr_other_params = tf.squeeze(o_other_params, axis=1)  # (N, 6), (x1, y1)=[0.0, 1.0], (x2, y2)=[-1.0, 1.0]
                x1y1, x2y2, width2, scaling = curr_other_params[:, 0:2], curr_other_params[:, 2:4],\
                                              curr_other_params[:, 4:5], curr_other_params[:, 5:6]
                x0y0 = tf.zeros_like(x2y2)  # (N, 2), [-1.0, 1.0]
                x0y0 = tf.div(tf.add(x0y0, 1.0), 2.0)  # (N, 2), [0.0, 1.0]
                x2y2 = tf.div(tf.add(x2y2, 1.0), 2.0)  # (N, 2), [0.0, 1.0]
                widths = tf.concat([tf.squeeze(prev_width, axis=1), width2], axis=1)  # (N, 2)
                curr_other_params = tf.concat([x0y0, x1y1, x2y2, widths], axis=-1)  # (N, 8), (x0, y0)&(x2, y2)=[0.0, 1.0]
                curr_stroke_image = rasterizor_st.raster_func_stroke_abs(curr_other_params)
                # (N, raster_size, raster_size), [0.0-BG, 1.0-stroke]

                curr_stroke_image_large = pasting_func(cursor_position_loop, curr_stroke_image,
                                                             image_size, curr_window_size,
                                                             is_differentiable=self.hps.pasting_diff)
                # (N, image_size, image_size), [0.0-BG, 1.0-stroke]

                ## soft
                if not self.hps.use_softargmax:
                    curr_state_soft = o_pen_ras[:, 1:2]  # (N, 1)
                else:
                    curr_state_soft = self.differentiable_argmax(o_pen_ras, self.hps.soft_beta)  # (N, 1)

                curr_state_soft = tf.expand_dims(curr_state_soft, axis=1)  # (N, 1, 1)

                filter_curr_stroke_image_soft = tf.multiply(tf.subtract(1.0, curr_state_soft), curr_stroke_image_large)
                # (N, image_size, image_size), [0.0-BG, 1.0-stroke]
                curr_canvas_soft = tf.add(curr_canvas_soft, filter_curr_stroke_image_soft)  # [0.0-BG, 1.0-stroke]

                ## hard
                curr_state_hard = tf.expand_dims(tf.cast(tf.argmax(o_pen_ras_raw, axis=-1), dtype=tf.float32),
                                                     axis=-1)  # (N, 1, 1)
                filter_curr_stroke_image_hard = tf.multiply(tf.subtract(1.0, curr_state_hard), curr_stroke_image_large)
                # (N, image_size, image_size), [0.0-BG, 1.0-stroke]
                self.curr_canvas_hard = tf.add(self.curr_canvas_hard, filter_curr_stroke_image_hard)  # [0.0-BG, 1.0-stroke]
                self.curr_canvas_hard = tf.clip_by_value(self.curr_canvas_hard, 0.0, 1.0)  # [0.0-BG, 1.0-stroke]

            next_width = o_other_params[:, :, 4:5]
            next_scaling = o_other_params[:, :, 5:6]
            next_window_size = tf.multiply(next_scaling, tf.stop_gradient(curr_window_size))  # float, with grad
            window_size_before_max_min = next_window_size  # (N, 1, 1), large-level
            win_size_before_max_min_list.append(window_size_before_max_min)
            next_window_size = tf.maximum(next_window_size, tf.cast(self.hps.min_window_size, tf.float32))
            next_window_size = tf.minimum(next_window_size, tf.cast(image_size, tf.float32))

            prev_state = next_state
            prev_width = next_width * curr_window_size / next_window_size  # (N, 1, 1)
            prev_scaling = next_scaling  # (N, 1, 1))
            prev_window_size = curr_window_size

            # update the cursor position
            new_cursor_offsets = tf.multiply(o_other_params[:, :, 2:4],
                                             tf.divide(curr_window_size, 2.0))  # (N, 1, 2), window-level
            new_cursor_offset_next = new_cursor_offsets
            new_cursor_offset_next = tf.concat([new_cursor_offset_next[:, :, 1:2], new_cursor_offset_next[:, :, 0:1]], axis=-1)

            cursor_position_loop_large = tf.multiply(cursor_position_loop, tf.cast(image_size, tf.float32))

            if self.hps.stop_accu_grad:
                stroke_position_next = tf.stop_gradient(cursor_position_loop_large) + new_cursor_offset_next  # (N, 1, 2), large-level
            else:
                stroke_position_next = cursor_position_loop_large + new_cursor_offset_next  # (N, 1, 2), large-level

            stroke_position_before_max_min = stroke_position_next  # (N, 1, 2), large-level
            pos_before_max_min_list.append(stroke_position_before_max_min)

            if self.hps.cursor_type == 'next':
                cursor_position_loop_large = stroke_position_next  # (N, 1, 2), large-level
            else:
                raise Exception('Unknown cursor_type')

            cursor_position_loop_large = tf.maximum(cursor_position_loop_large, 0.0)
            cursor_position_loop_large = tf.minimum(cursor_position_loop_large, tf.cast(image_size - 1, tf.float32))
            cursor_position_loop = tf.div(cursor_position_loop_large, tf.cast(image_size, tf.float32))

        curr_canvas_soft = tf.clip_by_value(curr_canvas_soft, 0.0, 1.0)  # (N, raster_size, raster_size), [0.0-BG, 1.0-stroke]

        other_params_ = tf.reshape(tf.concat(other_params_list, axis=1), [-1, 6])  # (N * max_seq_len, 6)
        pen_ras_ = tf.reshape(tf.concat(pen_ras_list, axis=1), [-1, 2])  # (N * max_seq_len, 2)
        pos_before_max_min_ = tf.concat(pos_before_max_min_list, axis=1)  # (N, max_seq_len, 2)
        win_size_before_max_min_ = tf.concat(win_size_before_max_min_list, axis=1)  # (N, max_seq_len, 1)

        return other_params_, pen_ras_, prev_state, curr_canvas_soft, curr_canvas_soft_rgb, \
               pos_before_max_min_, win_size_before_max_min_

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

    def build_losses(self, target_sketch, pred_raster_imgs, pred_params,
                     pos_before_max_min, win_size_before_max_min, image_size):
        def get_raster_loss(pred_imgs, gt_imgs, loss_type):
            perc_layer_losses_raw = []
            perc_layer_losses_weighted = []
            perc_layer_losses_norm = []

            if loss_type == 'l1':
                ras_cost = tf.reduce_mean(tf.abs(tf.subtract(gt_imgs, pred_imgs)))  # ()
            elif loss_type == 'l1_small':
                gt_imgs_small = tf.image.resize_images(tf.expand_dims(gt_imgs, axis=3), (32, 32))
                pred_imgs_small = tf.image.resize_images(tf.expand_dims(pred_imgs, axis=3), (32, 32))
                ras_cost = tf.reduce_mean(tf.abs(tf.subtract(gt_imgs_small, pred_imgs_small)))  # ()
            elif loss_type == 'mse':
                ras_cost = tf.reduce_mean(tf.pow(tf.subtract(gt_imgs, pred_imgs), 2))  # ()
            elif loss_type == 'perceptual':
                return_map_pred = vgg_net_slim(pred_imgs, image_size)
                return_map_gt = vgg_net_slim(gt_imgs, image_size)
                perc_loss_type = 'l1'  # [l1, mse]
                weighted_map = {'ReLU1_1': 100.0, 'ReLU1_2': 100.0,
                                'ReLU2_1': 100.0, 'ReLU2_2': 100.0,
                                'ReLU3_1': 10.0, 'ReLU3_2': 10.0, 'ReLU3_3': 10.0,
                                'ReLU4_1': 1.0, 'ReLU4_2': 1.0, 'ReLU4_3': 1.0,
                                'ReLU5_1': 1.0, 'ReLU5_2': 1.0, 'ReLU5_3': 1.0}

                for perc_layer in self.hps.perc_loss_layers:
                    if perc_loss_type == 'l1':
                        perc_layer_loss = tf.reduce_mean(tf.abs(tf.subtract(return_map_pred[perc_layer],
                                                                            return_map_gt[perc_layer])))  # ()
                    elif perc_loss_type == 'mse':
                        perc_layer_loss = tf.reduce_mean(tf.pow(tf.subtract(return_map_pred[perc_layer],
                                                                            return_map_gt[perc_layer]), 2))  # ()
                    else:
                        raise NameError('Unknown perceptual loss type:', perc_loss_type)
                    perc_layer_losses_raw.append(perc_layer_loss)

                    assert perc_layer in weighted_map
                    perc_layer_losses_weighted.append(perc_layer_loss * weighted_map[perc_layer])

                for loop_i in range(len(self.hps.perc_loss_layers)):
                    perc_relu_loss_raw = perc_layer_losses_raw[loop_i]  # ()

                    if self.hps.model_mode == 'train':
                        curr_relu_mean = (self.perc_loss_mean_list[loop_i] * self.last_step_num + perc_relu_loss_raw) / (self.last_step_num + 1.0)
                        relu_cost_norm = perc_relu_loss_raw / curr_relu_mean
                    else:
                        relu_cost_norm = perc_relu_loss_raw
                    perc_layer_losses_norm.append(relu_cost_norm)

                perc_layer_losses_raw = tf.stack(perc_layer_losses_raw, axis=0)
                perc_layer_losses_norm = tf.stack(perc_layer_losses_norm, axis=0)

                if self.hps.perc_loss_fuse_type == 'max':
                    ras_cost = tf.reduce_max(perc_layer_losses_norm)
                elif self.hps.perc_loss_fuse_type == 'add':
                    ras_cost = tf.reduce_mean(perc_layer_losses_norm)
                elif self.hps.perc_loss_fuse_type == 'raw_add':
                    ras_cost = tf.reduce_mean(perc_layer_losses_raw)
                elif self.hps.perc_loss_fuse_type == 'weighted_sum':
                    ras_cost = tf.reduce_mean(perc_layer_losses_weighted)
                else:
                    raise NameError('Unknown perc_loss_fuse_type:', self.hps.perc_loss_fuse_type)

            elif loss_type == 'triplet':
                raise Exception('Solution for triplet loss is coming soon.')
            else:
                raise NameError('Unknown loss type:', loss_type)

            if loss_type != 'perceptual':
                for perc_layer_i in self.hps.perc_loss_layers:
                    perc_layer_losses_raw.append(tf.constant(0.0))
                    perc_layer_losses_norm.append(tf.constant(0.0))

                perc_layer_losses_raw = tf.stack(perc_layer_losses_raw, axis=0)
                perc_layer_losses_norm = tf.stack(perc_layer_losses_norm, axis=0)

            return ras_cost, perc_layer_losses_raw, perc_layer_losses_norm

        gt_raster_images = tf.squeeze(target_sketch, axis=3)  # (N, raster_h, raster_w), [0.0-stroke, 1.0-BG]
        raster_cost, perc_relu_losses_raw, perc_relu_losses_norm = \
            get_raster_loss(pred_raster_imgs, gt_raster_images, loss_type=self.hps.raster_loss_base_type)

        def get_stroke_num_loss(input_strokes):
            ending_state = input_strokes[:, :, 0]  # (N, seq_len)
            stroke_num_loss_pre = tf.reduce_mean(ending_state)  # larger is better, [0.0, 1.0]
            stroke_num_loss = 1.0 - stroke_num_loss_pre  # lower is better, [0.0, 1.0]
            return stroke_num_loss

        stroke_num_cost = get_stroke_num_loss(pred_params)  # lower is better

        def get_pos_outside_loss(pos_before_max_min_):
            pos_after_max_min = tf.maximum(pos_before_max_min_, 0.0)
            pos_after_max_min = tf.minimum(pos_after_max_min, tf.cast(image_size - 1, tf.float32))  # (N, max_seq_len, 2)
            pos_outside_loss = tf.reduce_mean(tf.abs(pos_before_max_min_ - pos_after_max_min))
            return pos_outside_loss

        pos_outside_cost = get_pos_outside_loss(pos_before_max_min)  # lower is better

        def get_win_size_outside_loss(win_size_before_max_min_, min_window_size):
            win_size_outside_top_loss = tf.divide(
                tf.maximum(win_size_before_max_min_ - tf.cast(image_size, tf.float32), 0.0),
                tf.cast(image_size, tf.float32))  # (N, max_seq_len, 1)
            win_size_outside_bottom_loss = tf.divide(
                tf.maximum(tf.cast(min_window_size, tf.float32) - win_size_before_max_min_, 0.0),
                tf.cast(min_window_size, tf.float32))  # (N, max_seq_len, 1)
            win_size_outside_loss = tf.reduce_mean(win_size_outside_top_loss + win_size_outside_bottom_loss)
            return win_size_outside_loss

        win_size_outside_cost = get_win_size_outside_loss(win_size_before_max_min, self.hps.min_window_size)  # lower is better

        def get_early_pen_states_loss(input_strokes, curr_start, curr_end):
            # input_strokes: (N, max_seq_len, 7)
            pred_early_pen_states = input_strokes[:, curr_start:curr_end, 0]  # (N, curr_early_len)
            pred_early_pen_states_min = tf.reduce_min(pred_early_pen_states, axis=1)  # (N), should not be 1
            early_pen_states_loss = tf.reduce_mean(pred_early_pen_states_min)  # lower is better
            return early_pen_states_loss

        early_pen_states_cost = get_early_pen_states_loss(pred_params,
                                                          self.early_pen_loss_start_idx, self.early_pen_loss_end_idx)

        return raster_cost, stroke_num_cost, pos_outside_cost, win_size_outside_cost, \
               early_pen_states_cost, \
               perc_relu_losses_raw, perc_relu_losses_norm

    def build_training_op_split(self, raster_cost, sn_cost, cursor_outside_cost, win_size_outside_cost,
                                early_pen_states_cost):
        total_cost = self.hps.raster_loss_weight * raster_cost + \
                self.hps.early_pen_loss_weight * early_pen_states_cost + \
                self.stroke_num_loss_weight * sn_cost + \
                self.hps.outside_loss_weight * cursor_outside_cost + \
                self.hps.win_size_outside_loss_weight * win_size_outside_cost

        tvars = [var for var in tf.trainable_variables()
                 if 'raster_unit' not in var.op.name and 'VGG16' not in var.op.name]
        gvs = self.optimizer.compute_gradients(total_cost, var_list=tvars)
        return total_cost, gvs

    def build_training_op(self, grad_list):
        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
            gvs = self.average_gradients(grad_list)
            g = self.hps.grad_clip

            for grad, var in gvs:
                print('>>', var.op.name)
                if grad is None:
                    print('  >> None value')

            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]

            self.train_op = self.optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step, name='train_step')

    def average_gradients(self, grads_list):
        """
        Compute the average gradients.
        :param grads_list: list(of length N_GPU) of list(grad, var)
        :return:
        """
        avg_grads = []
        for grad_and_vars in zip(*grads_list):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            avg_grads.append(grad_and_var)

        return avg_grads