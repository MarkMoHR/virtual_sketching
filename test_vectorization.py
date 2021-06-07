import numpy as np
import random
import os
import tensorflow as tf
from six.moves import range
from PIL import Image
import time
import argparse

import hyper_parameters as hparams
from model_common_test import DiffPastingV3, VirtualSketchingModel
from utils import reset_graph, load_checkpoint, update_hyperparams, draw, \
    save_seq_data, image_pasting_v3_testing, draw_strokes
from dataset_utils import load_dataset_testing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def move_cursor_to_undrawn(current_canvas_list, input_image_, last_min_acc_list, grid_patch_size=128,
                           stroke_acc_threshold=0.95, stroke_num_threshold=5, continuous_min_acc_threshold=2):
    """
    :param current_canvas_list: (select_times, image_size, image_size), [0.0-BG, 1.0-stroke]
    :param input_image_: (1, image_size, image_size), [0-stroke, 1-BG]
    :return: new_cursor_pos: (select_times, 1, 2), [0.0, 1.0)
    """
    def split_images(in_img, image_size, grid_size):
        if image_size % grid_size == 0:
            paddings_ = 0
        else:
            paddings_ = grid_size - image_size % grid_size
        paddings = [[0, paddings_],
                    [0, paddings_]]
        image_pad = np.pad(in_img, paddings, mode='constant', constant_values=0.0)  # (H_p, W_p), [0.0-BG, 1.0-stroke]

        assert image_pad.shape[0] % grid_size == 0

        split_num = image_pad.shape[0] // grid_size

        images_h = np.hsplit(image_pad, split_num)
        image_patches = []
        for image_h in images_h:
            images_v = np.vsplit(image_h, split_num)
            image_patches += images_v
        image_patches = np.array(image_patches, dtype=np.float32)
        return image_patches, split_num

    def line_drawing_rounding(line_drawing):
        line_drawing_r = np.copy(line_drawing)  # [0.0-BG, 1.0-stroke]
        line_drawing_r[line_drawing_r != 0.0] = 1.0
        return line_drawing_r

    def cal_undrawn_pixels(in_canvas, in_sketch):
        in_canvas_round = line_drawing_rounding(in_canvas).astype(np.int32)  # (N, H, W), [0.0-BG, 1.0-stroke]
        in_sketch_round = line_drawing_rounding(in_sketch).astype(np.int32)

        intersection = np.bitwise_and(in_canvas_round, in_sketch_round)

        intersection_sum = np.sum(intersection, axis=(1, 2))
        gt_sum = np.sum(in_sketch_round, axis=(1, 2))  # (N)

        undrawn_num = gt_sum - intersection_sum
        return undrawn_num

    def cal_stroke_acc(in_canvas, in_sketch):
        in_canvas_round = line_drawing_rounding(in_canvas).astype(np.int32)  # (N, H, W), [0.0-BG, 1.0-stroke]
        in_sketch_round = line_drawing_rounding(in_sketch).astype(np.int32)

        intersection = np.bitwise_and(in_canvas_round, in_sketch_round)

        intersection_sum = np.sum(intersection, axis=(1, 2)).astype(np.float32)
        gt_sum = np.sum(in_sketch_round, axis=(1, 2)).astype(np.float32)  # (N)
        undrawn_num = gt_sum - intersection_sum  # (N)

        stroke_acc = intersection_sum / gt_sum  # (N)
        stroke_acc[gt_sum == 0.0] = 1.0
        stroke_acc[undrawn_num <= stroke_num_threshold] = 1.0
        return stroke_acc

    def get_cursor(patch_idx, img_size, grid_size, split_num):
        y_pos = patch_idx % split_num
        x_pos = patch_idx // split_num

        y_top = y_pos * grid_size + grid_size // 4
        y_bottom = y_top + grid_size // 2
        x_left = x_pos * grid_size + grid_size // 4
        x_right = x_left + grid_size // 2

        cursor_y = random.randint(y_top, y_bottom)
        cursor_x = random.randint(x_left, x_right)

        cursor_y = max(0, min(cursor_y, img_size - 1))
        cursor_x = max(0, min(cursor_x, img_size - 1))  # (2), in large size
        center = np.array([cursor_x, cursor_y], dtype=np.float32)

        return center / float(img_size)  # (2), in size [0.0, 1.0)

    input_image = 1.0 - input_image_[0]  # (image_size, image_size), [0-BG, 1-stroke]
    img_size = input_image.shape[0]

    input_image_patches, split_number = split_images(input_image, img_size, grid_patch_size)  # (N, grid_size, grid_size)

    new_cursor_pos = []
    last_min_acc_list_new = [item for item in last_min_acc_list]
    for canvas_i in range(current_canvas_list.shape[0]):
        curr_canvas = current_canvas_list[canvas_i]  # (image_size, image_size), [0.0-BG, 1.0-stroke]

        curr_canvas_patches, _ = split_images(curr_canvas, img_size, grid_patch_size)  # (N, grid_size, grid_size)

        # 1. detect ending flag by stroke accuracy
        stroke_accuracy = cal_stroke_acc(curr_canvas_patches, input_image_patches)
        min_acc_idx = np.argmin(stroke_accuracy)
        min_acc= stroke_accuracy[min_acc_idx]
        # print('min_acc_idx', min_acc_idx, ' | ', 'min_acc', min_acc)

        if min_acc >= stroke_acc_threshold:  # end of drawing
            return None, None

        # 2. detect undrawn pixels
        undrawn_pixel_num = cal_undrawn_pixels(curr_canvas_patches, input_image_patches)
        # undrawn_pixel_num_dis = np.reshape(undrawn_pixel_num, (split_number, split_number)).T
        # print('undrawn_pixel_num_dis')
        # print(undrawn_pixel_num_dis)

        max_undrawn_idx = np.argmax(undrawn_pixel_num)
        # max_undrawn = undrawn_pixel_num[max_undrawn_idx]
        # print('max_undrawn_idx', max_undrawn_idx, ' | ', 'max_undrawn', max_undrawn)

        # 3. select a random position
        last_min_acc_idx, last_min_acc_times = last_min_acc_list[canvas_i]
        if last_min_acc_times >= continuous_min_acc_threshold and last_min_acc_idx == min_acc_idx:
            selected_patch_idx = last_min_acc_idx
            new_min_acc_times = 1
        else:
            selected_patch_idx = max_undrawn_idx

            if min_acc_idx == last_min_acc_idx:
                new_min_acc_times = last_min_acc_times + 1
            else:
                new_min_acc_times = 1

        new_min_acc_idx = min_acc_idx
        last_min_acc_list_new[canvas_i] = (new_min_acc_idx, new_min_acc_times)
        # print('selected_patch_idx', selected_patch_idx)

        # 4. get cursor according to the selected_patch_idx
        rand_cursor = get_cursor(selected_patch_idx, img_size, grid_patch_size, split_number)  # (2), in size [0.0, 1.0)
        new_cursor_pos.append(rand_cursor)

    assert len(new_cursor_pos) == current_canvas_list.shape[0]
    new_cursor_pos = np.expand_dims(np.stack(new_cursor_pos, axis=0), axis=1)  # (select_times, 1, 2), [0.0, 1.0)
    return new_cursor_pos, last_min_acc_list_new


def sample(sess, model, input_photos, init_cursor, image_size, init_len, seq_lens, state_dependent,
           pasting_func, round_stop_state_num, stroke_acc_threshold):
    """Samples a sequence from a pre-trained model."""
    select_times = 1
    curr_canvas = np.zeros(dtype=np.float32,
                           shape=(select_times, image_size, image_size))  # [0.0-BG, 1.0-stroke]

    initial_state = sess.run(model.initial_state)
    prev_width = np.stack([model.hps.min_width for _ in range(select_times)], axis=0)

    params_list = [[] for _ in range(select_times)]
    state_raw_list = [[] for _ in range(select_times)]
    state_soft_list = [[] for _ in range(select_times)]
    window_size_list = [[] for _ in range(select_times)]

    last_min_stroke_acc_list = [(-1, 0) for _ in range(select_times)]

    round_cursor_list = []
    round_length_real_list = []

    input_photos_tiles = np.tile(input_photos, (select_times, 1, 1))

    for cursor_i, seq_len in enumerate(seq_lens):
        # print('\n')
        # print('@@ Round', cursor_i + 1)
        if cursor_i == 0:
            cursor_pos = np.squeeze(init_cursor, axis=0)  # (select_times, 1, 2)
        else:
            cursor_pos, last_min_stroke_acc_list_updated = \
                move_cursor_to_undrawn(curr_canvas, input_photos, last_min_stroke_acc_list,
                                       grid_patch_size=model.hps.raster_size,
                                       stroke_acc_threshold=stroke_acc_threshold)  # (select_times, 1, 2)
            if cursor_pos is not None:
                round_cursor_list.append(cursor_pos)
                last_min_stroke_acc_list = last_min_stroke_acc_list_updated
            else:
                break

        prev_state = initial_state
        if not model.hps.init_cursor_on_undrawn_pixel:
            prev_width = np.stack([model.hps.min_width for _ in range(select_times)], axis=0)
        prev_scaling = np.ones((select_times), dtype=np.float32)  # (N)
        prev_window_size = np.ones((select_times), dtype=np.float32) * model.hps.raster_size  # (N)

        continuous_one_state_num = 0

        for i in range(seq_len):
            if not state_dependent and i % init_len == 0:
                prev_state = initial_state

            curr_window_size = prev_scaling * prev_window_size  # (N)
            curr_window_size = np.maximum(curr_window_size, model.hps.min_window_size)
            curr_window_size = np.minimum(curr_window_size, image_size)

            feed = {
                model.initial_state: prev_state,
                model.input_photo: np.expand_dims(input_photos_tiles, axis=-1),
                model.curr_canvas_hard: curr_canvas.copy(),
                model.cursor_position: cursor_pos,
                model.image_size: image_size,
                model.init_width: prev_width,
                model.init_scaling: prev_scaling,
                model.init_window_size: prev_window_size,
            }

            o_other_params_list, o_pen_list, o_pred_params_list, next_state_list = \
                sess.run([model.other_params, model.pen_ras, model.pred_params, model.final_state], feed_dict=feed)
            # o_other_params: (N, 6), o_pen: (N, 2), pred_params: (N, 1, 7), next_state: (N, 1024)
            # o_other_params: [tanh*2, sigmoid*2, tanh*2, sigmoid*2]

            idx_eos_list = np.argmax(o_pen_list, axis=1)  # (N)

            output_i = 0
            idx_eos = idx_eos_list[output_i]

            eos = [0, 0]
            eos[idx_eos] = 1

            other_params = o_other_params_list[output_i].tolist()  # (6)
            params_list[output_i].append([eos[1]] + other_params)
            state_raw_list[output_i].append(o_pen_list[output_i][1])
            state_soft_list[output_i].append(o_pred_params_list[output_i, 0, 0])
            window_size_list[output_i].append(curr_window_size[output_i])

            # draw the stroke and add to the canvas
            x1y1, x2y2, width2 = o_other_params_list[output_i, 0:2], o_other_params_list[output_i, 2:4], \
                                 o_other_params_list[output_i, 4]
            x0y0 = np.zeros_like(x2y2)  # (2), [-1.0, 1.0]
            x0y0 = np.divide(np.add(x0y0, 1.0), 2.0)  # (2), [0.0, 1.0]
            x2y2 = np.divide(np.add(x2y2, 1.0), 2.0)  # (2), [0.0, 1.0]
            widths = np.stack([prev_width[output_i], width2], axis=0)  # (2)
            o_other_params_proc = np.concatenate([x0y0, x1y1, x2y2, widths], axis=-1).tolist()  # (8)

            if idx_eos == 0:
                f = o_other_params_proc + [1.0, 1.0]
                pred_stroke_img = draw(f)  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]
                pred_stroke_img_large = image_pasting_v3_testing(1.0 - pred_stroke_img, cursor_pos[output_i, 0],
                                                                  image_size,
                                                                  curr_window_size[output_i],
                                                                  pasting_func, sess)  # [0.0-BG, 1.0-stroke]
                curr_canvas[output_i] += pred_stroke_img_large  # [0.0-BG, 1.0-stroke]

                continuous_one_state_num = 0
            else:
                continuous_one_state_num += 1

            curr_canvas = np.clip(curr_canvas, 0.0, 1.0)

            next_width = o_other_params_list[:, 4]  # (N)
            next_scaling = o_other_params_list[:, 5]
            next_window_size = next_scaling * curr_window_size  # (N)
            next_window_size = np.maximum(next_window_size, model.hps.min_window_size)
            next_window_size = np.minimum(next_window_size, image_size)

            prev_state = next_state_list
            prev_width = next_width * curr_window_size / next_window_size  # (N,)
            prev_scaling = next_scaling  # (N)
            prev_window_size = curr_window_size

            # update cursor_pos based on hps.cursor_type
            new_cursor_offsets = o_other_params_list[:, 2:4] * (np.expand_dims(curr_window_size, axis=-1) / 2.0)  # (N, 2), patch-level
            new_cursor_offset_next = new_cursor_offsets

            # important!!!
            new_cursor_offset_next = np.concatenate([new_cursor_offset_next[:, 1:2], new_cursor_offset_next[:, 0:1]], axis=-1)

            cursor_pos_large = cursor_pos * float(image_size)
            stroke_position_next = cursor_pos_large[:, 0, :] + new_cursor_offset_next  # (N, 2), large-level

            if model.hps.cursor_type == 'next':
                cursor_pos_large = stroke_position_next  # (N, 2), large-level
            else:
                raise Exception('Unknown cursor_type')

            cursor_pos_large = np.minimum(np.maximum(cursor_pos_large, 0.0), float(image_size - 1))  # (N, 2), large-level
            cursor_pos_large = np.expand_dims(cursor_pos_large, axis=1)  # (N, 1, 2)
            cursor_pos = cursor_pos_large / float(image_size)

            if continuous_one_state_num >= round_stop_state_num or i == seq_len - 1:
                round_length_real_list.append(i + 1)
                break

    return params_list, state_raw_list, state_soft_list, curr_canvas, window_size_list, \
           round_cursor_list, round_length_real_list


def main_testing(test_image_base_dir, test_dataset, test_image_name,
                 sampling_base_dir, model_base_dir, model_name,
                 sampling_num,
                 longer_infer_lens,
                 round_stop_state_num, stroke_acc_threshold,
                 draw_seq=False, draw_order=False,
                 state_dependent=True):
    model_params_default = hparams.get_default_hparams_clean()
    model_params = update_hyperparams(model_params_default, model_base_dir, model_name, infer_dataset=test_dataset)

    [test_set, eval_hps_model, sample_hps_model] \
        = load_dataset_testing(test_image_base_dir, test_dataset, test_image_name, model_params)

    test_image_raw_name = test_image_name[:test_image_name.find('.')]
    model_dir = os.path.join(model_base_dir, model_name)

    reset_graph()
    sampling_model = VirtualSketchingModel(sample_hps_model)

    # differentiable pasting graph
    paste_v3_func = DiffPastingV3(sample_hps_model.raster_size)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    snapshot_step = load_checkpoint(sess, model_dir, gen_model_pretrain=True)
    print('snapshot_step', snapshot_step)
    sampling_dir = os.path.join(sampling_base_dir, test_dataset + '__' + model_name)
    os.makedirs(sampling_dir, exist_ok=True)

    stroke_number_list = []
    compute_time_list = []

    for sampling_i in range(sampling_num):
        start_time_point = time.time()
        input_photos, init_cursors, test_image_size = test_set.get_test_image()
        # input_photos: (1, image_size, image_size), [0-stroke, 1-BG]
        # init_cursors: (1, 1, 2), in size [0.0, 1.0)

        print()
        print(test_image_name, ', image_size:', test_image_size, ', sampling_i:', sampling_i)
        print('Processing ...')

        if init_cursors.ndim == 3:
            init_cursors = np.expand_dims(init_cursors, axis=0)

        input_photos = input_photos[0:1, :, :]
        ori_img = (input_photos.copy()[0] * 255.0).astype(np.uint8)
        ori_img = np.stack([ori_img for _ in range(3)], axis=2)
        ori_img_png = Image.fromarray(ori_img, 'RGB')
        ori_img_png.save(os.path.join(sampling_dir, test_image_raw_name + '_input.png'), 'PNG')

        data_loading_time_point = time.time()

        # decoding for sampling
        strokes_raw_out_list, states_raw_out_list, states_soft_out_list, pred_imgs_out, \
        window_size_out_list, round_new_cursors, round_new_lengths = sample(
            sess, sampling_model, input_photos, init_cursors, test_image_size,
            eval_hps_model.max_seq_len, longer_infer_lens, state_dependent,
            paste_v3_func, round_stop_state_num, stroke_acc_threshold)
        # pred_imgs_out: [0.0-BG, 1.0-stroke]

        print('## round_lengths:', len(round_new_lengths), ':', round_new_lengths)

        sampling_time_point = time.time()

        data_loading_time = data_loading_time_point - start_time_point
        sampling_time_total = sampling_time_point - start_time_point
        sampling_time_wo_data_loading = sampling_time_point - data_loading_time_point
        compute_time_list.append(sampling_time_total)
        # print('  >>> data_loading_time', data_loading_time)
        print('  >>> sampling_time_total', sampling_time_total)
        # print('  >>> sampling_time_wo_data_loading', sampling_time_wo_data_loading)

        best_result_idx = 0
        strokes_raw_out = np.stack(strokes_raw_out_list[best_result_idx], axis=0)
        states_raw_out = states_raw_out_list[best_result_idx]
        states_soft_out = states_soft_out_list[best_result_idx]
        window_size_out = window_size_out_list[best_result_idx]

        multi_cursors = [init_cursors[0, best_result_idx, 0]]
        for c_i in range(len(round_new_cursors)):
            best_cursor = round_new_cursors[c_i][best_result_idx, 0]  # (2)
            multi_cursors.append(best_cursor)
        assert len(multi_cursors) == len(round_new_lengths)

        print('strokes_raw_out', strokes_raw_out.shape)
        stroke_number_list.append(strokes_raw_out.shape[0])

        clean_states_soft_out = np.array(states_soft_out)  # (N)

        flag_list = strokes_raw_out[:, 0].astype(np.int32)  # (N)
        drawing_len = len(flag_list) - np.sum(flag_list)
        assert drawing_len >= 0

        # print('    flag  raw\t soft\t x1\t\t y1\t\t x2\t\t y2\t\t r2\t\t s2')
        for i in range(strokes_raw_out.shape[0]):
            flag, x1, y1, x2, y2, r2, s2 = strokes_raw_out[i]
            win_size = window_size_out[i]
            out_format = '#%d: %d  | %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
            out_values = (i, flag, states_raw_out[i], clean_states_soft_out[i], x1, y1, x2, y2, r2, s2)
            out_log = out_format % out_values
            # print(out_log)

        print('Saving results ...')
        save_seq_data(sampling_dir, test_image_raw_name + '_' + str(sampling_i),
                      strokes_raw_out, multi_cursors,
                      test_image_size, round_new_lengths, eval_hps_model.min_width)

        draw_strokes(strokes_raw_out, sampling_dir, test_image_raw_name + '_' + str(sampling_i) + '_pred.png', 
                     ori_img, test_image_size,
                     multi_cursors, round_new_lengths, eval_hps_model.min_width, eval_hps_model.cursor_type,
                     sample_hps_model.raster_size, sample_hps_model.min_window_size,
                     sess,
                     pasting_func=paste_v3_func,
                     save_seq=draw_seq, draw_order=draw_order)

    average_stroke_number = np.mean(stroke_number_list)
    average_compute_time = np.mean(compute_time_list)
    print()
    print('@@@ Total summary:')
    print('  >>> average_stroke_number', average_stroke_number)
    print('  >>> average_compute_time', average_compute_time)


def main(model_name, test_image_name, sampling_num):
    test_dataset = 'clean_line_drawings'
    test_image_base_dir = 'sample_inputs'

    sampling_base_dir = 'outputs/sampling'
    model_base_dir = 'outputs/snapshot'

    state_dependent = False
    longer_infer_lens = [500 for _ in range(10)]
    round_stop_state_num = 12
    stroke_acc_threshold = 0.95

    draw_seq = False
    draw_color_order = True

    # set numpy output to something sensible
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    main_testing(test_image_base_dir, test_dataset, test_image_name,
                 sampling_base_dir, model_base_dir, model_name, sampling_num,
                 draw_seq=draw_seq, draw_order=draw_color_order,
                 state_dependent=state_dependent, longer_infer_lens=longer_infer_lens,
                 round_stop_state_num=round_stop_state_num, stroke_acc_threshold=stroke_acc_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', help="The test image name.")
    parser.add_argument('--model', '-m', type=str, default='pretrain_clean_line_drawings', help="The trained model.")
    parser.add_argument('--sample', '-s', type=int, default=1, help="The number of outputs.")
    args = parser.parse_args()

    assert args.input != ''
    assert args.sample > 0

    main(args.model, args.input, args.sample)
