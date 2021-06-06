import json
import os
import time
import numpy as np
import six
import tensorflow as tf
from PIL import Image
import argparse

import model_common_train as sketch_image_model
from hyper_parameters import FLAGS, get_default_hparams_rough, get_default_hparams_normal
from utils import create_summary, save_model, reset_graph, load_checkpoint
from dataset_utils import load_dataset_training

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

tf.logging.set_verbosity(tf.logging.INFO)


def should_save_log_img(step_):
    if step_ % 500 == 0:
        return True
    else:
        return False


def save_log_images(sess, model, data_set, save_root, step_num, curr_photo_prob, interpolate_type, save_num=10):
    res_gap = (model.hps.image_size_large - model.hps.image_size_small) // (save_num - 1)
    log_img_resolutions = []
    for ii in range(save_num - 1):
        log_img_resolutions.append(model.hps.image_size_small + ii * res_gap)
    log_img_resolutions.append(model.hps.image_size_large)

    for res_i in range(len(log_img_resolutions)):
        resolution = log_img_resolutions[res_i]

        sub_save_root = os.path.join(save_root, 'res_' + str(resolution))
        os.makedirs(sub_save_root, exist_ok=True)

        input_photos, target_sketches, init_cursors, image_size_rand = \
            data_set.get_batch_from_memory(memory_idx=res_i,
                                           fixed_image_size=resolution,
                                           random_cursor=model.hps.random_cursor,
                                           photo_prob=curr_photo_prob,
                                           interpolate_type=interpolate_type)
        # input_photos: (N, image_size, image_size, 3), [0-stroke, 1-BG]
        # target_sketches: (N, image_size, image_size), [0-stroke, 1-BG]
        # init_cursors: (N, 1, 2), in size [0.0, 1.0)

        input_photo_val = input_photos

        init_cursor_input = [init_cursors for _ in range(model.total_loop)]
        init_cursor_input = np.concatenate(init_cursor_input, axis=0)
        image_size_input = [image_size_rand for _ in range(model.total_loop)]
        image_size_input = np.stack(image_size_input, axis=0)

        feed = {
            model.init_cursor: init_cursor_input,
            model.image_size: image_size_input,
            model.init_width: [model.hps.min_width],
        }
        for loop_i in range(model.total_loop):
            feed[model.input_photo_list[loop_i]] = input_photo_val

        raster_images_pred, raster_images_pred_rgb = sess.run([model.pred_raster_imgs, model.pred_raster_imgs_rgb],
                                                              feed)  # (N, image_size, image_size), [0.0-stroke, 1.0-BG]
        raster_images_pred = (np.array(raster_images_pred[0]) * 255.0).astype(np.uint8)
        input_photo = (np.array(input_photo_val[0, :, :, :]) * 255.0).astype(np.uint8)
        target_sketch = (np.array(target_sketches[0]) * 255.0).astype(np.uint8)
        raster_images_pred_rgb = (np.array(raster_images_pred_rgb[0]) * 255.0).astype(np.uint8)

        pred_save_path = os.path.join(sub_save_root, str(step_num) + '.png')
        input_save_path = os.path.join(sub_save_root, 'input.png')
        target_save_path = os.path.join(sub_save_root, 'gt.png')

        pred_rgb_save_root = os.path.join(sub_save_root, 'rgb')
        os.makedirs(pred_rgb_save_root, exist_ok=True)
        pred_rgb_save_path = os.path.join(pred_rgb_save_root, str(step_num) + '.png')

        raster_images_pred = Image.fromarray(raster_images_pred, 'L')
        raster_images_pred.save(pred_save_path, 'PNG')
        input_photo = Image.fromarray(input_photo, 'RGB')
        input_photo.save(input_save_path, 'PNG')
        target_sketch = Image.fromarray(target_sketch, 'L')
        target_sketch.save(target_save_path, 'PNG')
        raster_images_pred_rgb = Image.fromarray(raster_images_pred_rgb, 'RGB')
        raster_images_pred_rgb.save(pred_rgb_save_path, 'PNG')


def train(sess, train_model, eval_sample_model, train_set, valid_set, sub_log_root, sub_snapshot_root, sub_log_img_root):
    # Setup summary writer.
    summary_writer = tf.summary.FileWriter(sub_log_root)

    print('-' * 100)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i' % (var.name, str(var.get_shape()), num_param))
    print('Total trainable variables %i.' % count_t_vars)
    print('-' * 100)

    # main train loop

    hps = train_model.hps
    start = time.time()

    # create saver
    snapshot_save_vars = [var for var in tf.global_variables()
                          if 'raster_unit' not in var.op.name and 'VGG16' not in var.op.name]
    saver = tf.train.Saver(var_list=snapshot_save_vars, max_to_keep=20)

    start_step = 1
    print('start_step', start_step)

    mean_perc_relu_losses = [0.0 for _ in range(len(hps.perc_loss_layers))]

    for _ in range(start_step, hps.num_steps + 1):
        step = sess.run(train_model.global_step)  # start from 0

        count_step = min(step, hps.num_steps)
        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                              (1 - count_step / hps.num_steps) ** hps.decay_power + hps.min_learning_rate)

        if hps.sn_loss_type == 'decreasing':
            assert hps.decrease_stop_steps <= hps.num_steps
            assert hps.stroke_num_loss_weight_end <= hps.stroke_num_loss_weight
            curr_sn_k = (hps.stroke_num_loss_weight - hps.stroke_num_loss_weight_end) / float(hps.decrease_stop_steps)
            curr_stroke_num_loss_weight = hps.stroke_num_loss_weight - count_step * curr_sn_k
            curr_stroke_num_loss_weight = max(curr_stroke_num_loss_weight, hps.stroke_num_loss_weight_end)
        elif hps.sn_loss_type == 'fixed':
            curr_stroke_num_loss_weight = hps.stroke_num_loss_weight
        elif hps.sn_loss_type == 'increasing':
            curr_sn_k = hps.stroke_num_loss_weight / float(hps.num_steps - hps.increase_start_steps)
            curr_stroke_num_loss_weight = max(count_step - hps.increase_start_steps, 0) * curr_sn_k
        else:
            raise Exception('Unknown sn_loss_type', hps.sn_loss_type)

        if hps.early_pen_loss_type == 'head':
            curr_early_pen_k = (hps.max_seq_len - hps.early_pen_length) / float(hps.num_steps)
            curr_early_pen_loss_len = count_step * curr_early_pen_k + hps.early_pen_length

            curr_early_pen_loss_start = 1
            curr_early_pen_loss_end = curr_early_pen_loss_len
        elif hps.early_pen_loss_type == 'tail':
            curr_early_pen_k = (hps.max_seq_len // 2 - 1) / float(hps.num_steps)
            curr_early_pen_loss_len = count_step * curr_early_pen_k + hps.max_seq_len // 2

            curr_early_pen_loss_end = hps.max_seq_len
            curr_early_pen_loss_start = curr_early_pen_loss_end - curr_early_pen_loss_len
        elif hps.early_pen_loss_type == 'move':
            curr_early_pen_k = (hps.max_seq_len // 2 - 1) / float(hps.num_steps)
            curr_early_pen_loss_len = count_step * curr_early_pen_k + hps.max_seq_len // 2

            curr_early_pen_loss_start = hps.max_seq_len - curr_early_pen_loss_len
            curr_early_pen_loss_end = curr_early_pen_loss_start + hps.max_seq_len // 2
        else:
            raise Exception('Unknown early_pen_loss_type', hps.early_pen_loss_type)
        curr_early_pen_loss_start = int(round(curr_early_pen_loss_start))
        curr_early_pen_loss_end = int(round(curr_early_pen_loss_end))

        if hps.photo_prob_type == 'increasing' or hps.photo_prob_type == 'interpolate':
            assert hps.photo_prob_end_step >= hps.photo_prob_start_step
            curr_photo_prob_k = 1.0 / float(hps.photo_prob_end_step - hps.photo_prob_start_step)
            curr_photo_prob = (count_step - hps.photo_prob_start_step) * curr_photo_prob_k
            curr_photo_prob = max(0.0, curr_photo_prob)
            curr_photo_prob = min(1.0, curr_photo_prob)
            interpolate_type = 'prob' if hps.photo_prob_type == 'increasing' else 'image'
        elif hps.photo_prob_type == 'zero':
            curr_photo_prob = 0.0
            interpolate_type = 'prob'
        elif hps.photo_prob_type == 'one':
            curr_photo_prob = 1.0
            interpolate_type = 'prob'
        else:
            raise Exception('Unknown photo_prob_type', hps.photo_prob_type)

        input_photos, target_sketches, init_cursors, image_sizes = \
            train_set.get_batch_multi_res(loop_num=train_model.total_loop,
                                          random_cursor=hps.random_cursor,
                                          photo_prob=curr_photo_prob,
                                          interpolate_type=interpolate_type)
        # input_photos: list of (N, image_size, image_size, 3), [0-stroke, 1-BG]
        # target_sketches: list of (N, image_size, image_size), [0-stroke, 1-BG]
        # init_cursors: list of (N, 1, 2), in size [0.0, 1.0)

        init_cursors_input = np.concatenate(init_cursors, axis=0)
        image_size_input = np.stack(image_sizes, axis=0)

        feed = {
            train_model.init_cursor: init_cursors_input,
            train_model.image_size: image_size_input,
            train_model.init_width: [hps.min_width],

            train_model.lr: curr_learning_rate,
            train_model.stroke_num_loss_weight: curr_stroke_num_loss_weight,
            train_model.early_pen_loss_start_idx: curr_early_pen_loss_start,
            train_model.early_pen_loss_end_idx: curr_early_pen_loss_end,

            train_model.last_step_num: float(step),
        }
        for layer_i in range(len(hps.perc_loss_layers)):
            feed[train_model.perc_loss_mean_list[layer_i]] = mean_perc_relu_losses[layer_i]

        for loop_i in range(train_model.total_loop):
            input_photo_val = input_photos[loop_i]
            target_sketch_val = target_sketches[loop_i]
            feed[train_model.input_photo_list[loop_i]] = input_photo_val
            feed[train_model.target_sketch_list[loop_i]] = np.expand_dims(target_sketch_val, axis=-1)

        (train_cost, raster_cost, perc_relu_costs_raw, perc_relu_costs_norm,
         stroke_num_cost, early_pen_states_cost,
         pos_outside_cost, win_size_outside_cost,
         train_step) = sess.run([
            train_model.cost, train_model.raster_cost,
            train_model.perc_relu_losses_raw, train_model.perc_relu_losses_norm,
            train_model.stroke_num_cost,
            train_model.early_pen_states_cost,
            train_model.pos_outside_cost, train_model.win_size_outside_cost,
            train_model.global_step
         ], feed)

        ## update mean_raster_loss
        for layer_i in range(len(hps.perc_loss_layers)):
            perc_relu_cost_raw = perc_relu_costs_raw[layer_i]
            mean_perc_relu_loss = mean_perc_relu_losses[layer_i]
            mean_perc_relu_loss = (mean_perc_relu_loss * step + perc_relu_cost_raw) / float(step + 1)
            mean_perc_relu_losses[layer_i] = mean_perc_relu_loss

        _ = sess.run(train_model.train_op, feed)

        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            train_summary_map = {
                'Train_Cost': train_cost,
                'Train_raster_Cost': raster_cost,
                'Train_stroke_num_Cost': stroke_num_cost,
                'Train_early_pen_states_cost': early_pen_states_cost,
                'Train_pos_outside_Cost': pos_outside_cost,
                'Train_win_size_outside_Cost': win_size_outside_cost,
                'Learning_Rate': curr_learning_rate,
                'Time_Taken_Train': time_taken
            }
            for layer_i in range(len(hps.perc_loss_layers)):
                layer_name = hps.perc_loss_layers[layer_i]
                train_summary_map['Train_raster_Cost_' + layer_name] = perc_relu_costs_raw[layer_i]

            create_summary(summary_writer, train_summary_map, train_step)

            output_format = ('step: %d, lr: %.6f, '
                             'snw: %.3f, '
                             'cost: %.4f, '
                             'ras: %.4f, stroke_num: %.4f, early_pen: %.4f, '
                             'pos_outside: %.4f, win_outside: %.4f, '
                             'train_time_taken: %.1f')
            output_values = (step, curr_learning_rate,
                             curr_stroke_num_loss_weight,
                             train_cost,
                             raster_cost, stroke_num_cost, early_pen_states_cost,
                             pos_outside_cost, win_size_outside_cost,
                             time_taken)
            output_log = output_format % output_values
            # print(output_log)
            tf.logging.info(output_log)
            start = time.time()

        if should_save_log_img(step) and step > 0:
            save_log_images(sess, eval_sample_model, valid_set, sub_log_img_root, step, curr_photo_prob, interpolate_type)

        if step % hps.save_every == 0 and step > 0:
            save_model(sess, saver, sub_snapshot_root, step)


def trainer(model_params):
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('Loading data files.')
    print('-' * 100)

    datasets = load_dataset_training(FLAGS.dataset_dir, model_params)

    sub_snapshot_root = os.path.join(FLAGS.snapshot_root, model_params.program_name)
    sub_log_root = os.path.join(FLAGS.log_root, model_params.program_name)
    sub_log_img_root = os.path.join(FLAGS.log_img_root, model_params.program_name)

    train_set = datasets[0]
    valid_set = datasets[1]
    train_model_params = datasets[2]
    eval_sample_model_params = datasets[3]

    eval_sample_model_params.loop_per_gpu = 1
    eval_sample_model_params.batch_size = len(eval_sample_model_params.gpus) * eval_sample_model_params.loop_per_gpu

    reset_graph()
    train_model = sketch_image_model.VirtualSketchingModel(train_model_params)
    eval_sample_model = sketch_image_model.VirtualSketchingModel(eval_sample_model_params, reuse=True)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    load_checkpoint(sess, FLAGS.neural_renderer_path, ras_only=True)
    if train_model_params.raster_loss_base_type == 'perceptual':
        load_checkpoint(sess, FLAGS.perceptual_model_root, perceptual_only=True)

    # Write config file to json file.
    os.makedirs(sub_log_root, exist_ok=True)
    os.makedirs(sub_log_img_root, exist_ok=True)
    os.makedirs(sub_snapshot_root, exist_ok=True)
    with tf.gfile.Open(os.path.join(sub_snapshot_root, 'model_config.json'), 'w') as f:
        json.dump(train_model_params.values(), f, indent=True)

    train(sess, train_model, eval_sample_model, train_set, valid_set,
          sub_log_root, sub_snapshot_root, sub_log_img_root)


def main(dataset_type):
    if dataset_type == 'rough':
        model_params = get_default_hparams_rough()
    elif dataset_type == 'face':
        model_params = get_default_hparams_normal()
    else:
        raise Exception('Unknown dataset_type:', dataset_type)

    trainer(model_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='rough', choices=['rough', 'face'], help="The dataset type.")
    args = parser.parse_args()

    main(args.data)
