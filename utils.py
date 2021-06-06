import os
import cv2
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


#############################################
# Tensorflow utils
#############################################

def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_checkpoint(sess, checkpoint_path, ras_only=False, perceptual_only=False, gen_model_pretrain=False,
                    train_entire=False):
    if ras_only:
        load_var = {var.op.name: var for var in tf.global_variables() if 'raster_unit' in var.op.name}
    elif perceptual_only:
        load_var = {var.op.name: var for var in tf.global_variables() if 'VGG16' in var.op.name}
    elif train_entire:
        load_var = {var.op.name: var for var in tf.global_variables()
                    if 'discriminator' not in var.op.name
                    and 'raster_unit' not in var.op.name
                    and 'VGG16' not in var.op.name
                    and 'beta1' not in var.op.name
                    and 'beta2' not in var.op.name
                    and 'global_step' not in var.op.name
                    and 'Entire' not in var.op.name
                    }
    else:
        if gen_model_pretrain:
            load_var = {var.op.name: var for var in tf.global_variables()
                        if 'discriminator' not in var.op.name
                        and 'raster_unit' not in var.op.name
                        and 'VGG16' not in var.op.name
                        and 'beta1' not in var.op.name
                        and 'beta2' not in var.op.name
                        # and 'global_step' not in var.op.name
                        }
        else:
            load_var = tf.global_variables()

    restorer = tf.train.Saver(load_var)
    if not ras_only:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        model_checkpoint_path = ckpt.model_checkpoint_path
    else:
        model_checkpoint_path = checkpoint_path
    print('Loading model %s' % model_checkpoint_path)
    restorer.restore(sess, model_checkpoint_path)

    snapshot_step = model_checkpoint_path[model_checkpoint_path.rfind('-') + 1:]
    return snapshot_step


def create_summary(summary_writer, summ_map, step):
    for summ_key in summ_map:
        summ_value = summ_map[summ_key]
        summ = tf.summary.Summary()
        summ.value.add(tag=summ_key, simple_value=float(summ_value))
        summary_writer.add_summary(summ, step)
    summary_writer.flush()


def save_model(sess, saver, model_save_path, global_step):
    checkpoint_path = os.path.join(model_save_path, 'p2s')
    print('saving model %s.' % checkpoint_path)
    print('global_step %i.' % global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


#############################################
# Utils for basic image processing
#############################################


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))


def rgb_trans(split_num, break_values):
    slice_per_split = split_num // 8
    break_values_head, break_values_tail = break_values[:-1], break_values[1:]

    results = []

    for split_i in range(8):
        break_value_head = break_values_head[split_i]
        break_value_tail = break_values_tail[split_i]

        slice_gap = float(break_value_tail - break_value_head) / float(slice_per_split)
        for slice_i in range(slice_per_split):
            slice_val = break_value_head + slice_gap * slice_i
            slice_val = int(round(slice_val))
            results.append(slice_val)

    return results


def get_colors(color_num):
    split_num = (color_num // 8 + 1) * 8

    r_break_values = [0, 0, 0, 0, 128, 255, 255, 255, 128]
    g_break_values = [0, 0, 128, 255, 255, 255, 128, 0, 0]
    b_break_values = [128, 255, 255, 255, 128, 0, 0, 0, 0]

    r_rst_list = rgb_trans(split_num, r_break_values)
    g_rst_list = rgb_trans(split_num, g_break_values)
    b_rst_list = rgb_trans(split_num, b_break_values)

    assert len(r_rst_list) == len(g_rst_list)
    assert len(b_rst_list) == len(g_rst_list)

    rgb_color_list = [(r_rst_list[i], g_rst_list[i], b_rst_list[i]) for i in range(len(r_rst_list))]
    return rgb_color_list


#############################################
# Utils for testing
#############################################

def save_seq_data(save_root, save_filename, strokes_data, init_cursors, image_size, round_length, init_width):
    seq_save_root = os.path.join(save_root, 'seq_data')
    os.makedirs(seq_save_root, exist_ok=True)
    save_npz_path = os.path.join(seq_save_root, save_filename + '.npz')
    np.savez(save_npz_path, strokes_data=strokes_data, init_cursors=init_cursors,
             image_size=image_size, round_length=round_length, init_width=init_width)


def image_pasting_v3_testing(patch_image, cursor, image_size, window_size_f, pasting_func, sess):
    """
    :param patch_image:  (raster_size, raster_size), [0.0-BG, 1.0-stroke]
    :param cursor: (2), in size [0.0, 1.0)
    :param window_size_f: (), float32, [0.0, image_size)
    :return: (image_size, image_size), [0.0-BG, 1.0-stroke]
    """
    cursor_pos = cursor * float(image_size)
    pasted_image = sess.run(pasting_func.pasted_image,
                            feed_dict={pasting_func.patch_canvas: np.expand_dims(patch_image, axis=-1),
                                       pasting_func.cursor_pos_a: cursor_pos,
                                       pasting_func.image_size_a: image_size,
                                       pasting_func.window_size_a: window_size_f})
    # (image_size, image_size, 1), [0.0-BG, 1.0-stroke]
    pasted_image = pasted_image[:, :, 0]
    return pasted_image


def draw_strokes(data, save_root, save_filename, input_img, image_size, init_cursor, infer_lengths, init_width,
                 cursor_type, raster_size, min_window_size,
                 sess,
                 pasting_func=None,
                 save_seq=False, draw_order=False):
    """
    :param data: (N_strokes, 9): flag, x1, y1, x2, y2, r2, s2
    :return:
    """
    canvas = np.zeros((image_size, image_size), dtype=np.float32)  # [0.0-BG, 1.0-stroke]
    canvas_color = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas_color_with_moving = np.zeros((image_size, image_size, 3), dtype=np.float32)
    frames = []

    cursor_idx = 0

    stroke_count = len(data)
    color_rgb_set = get_colors(stroke_count)  # list of (3,) in [0, 255]
    color_idx = 0

    for round_idx in range(len(infer_lengths)):
        round_length = infer_lengths[round_idx]

        cursor_pos = init_cursor[cursor_idx]  # (2)
        cursor_idx += 1

        prev_width = init_width
        prev_scaling = 1.0
        prev_window_size = raster_size  # (1)

        for round_inner_i in range(round_length):
            stroke_idx = np.sum(infer_lengths[:round_idx]).astype(np.int32) + round_inner_i

            curr_window_size = prev_scaling * prev_window_size
            curr_window_size = np.maximum(curr_window_size, min_window_size)
            curr_window_size = np.minimum(curr_window_size, image_size)

            pen_state = data[stroke_idx, 0]
            stroke_params = data[stroke_idx, 1:]  # (8)

            x1y1, x2y2, width2, scaling2 = stroke_params[0:2], stroke_params[2:4], stroke_params[4], stroke_params[5]
            x0y0 = np.zeros_like(x2y2)  # (2), [-1.0, 1.0]
            x0y0 = np.divide(np.add(x0y0, 1.0), 2.0)  # (2), [0.0, 1.0]
            x2y2 = np.divide(np.add(x2y2, 1.0), 2.0)  # (2), [0.0, 1.0]
            widths = np.stack([prev_width, width2], axis=0)  # (2)
            stroke_params_proc = np.concatenate([x0y0, x1y1, x2y2, widths], axis=-1)  # (8)

            next_width = stroke_params[4]
            next_scaling = stroke_params[5]
            next_window_size = next_scaling * curr_window_size
            next_window_size = np.maximum(next_window_size, min_window_size)
            next_window_size = np.minimum(next_window_size, image_size)

            prev_width = next_width * curr_window_size / next_window_size
            prev_scaling = next_scaling
            prev_window_size = curr_window_size

            f = stroke_params_proc.tolist()  # (8)
            f += [1.0, 1.0]
            gt_stroke_img = draw(f)  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]
            gt_stroke_img_large = image_pasting_v3_testing(1.0 - gt_stroke_img, cursor_pos, image_size,
                                                            curr_window_size,
                                                            pasting_func, sess)  # [0.0-BG, 1.0-stroke]

            if pen_state == 0:
                canvas += gt_stroke_img_large  # [0.0-BG, 1.0-stroke]

            if draw_order:
                color_rgb = color_rgb_set[color_idx]  # (3) in [0, 255]
                color_idx += 1

                color_rgb = np.reshape(color_rgb, (1, 1, 3)).astype(np.float32)
                color_stroke = np.expand_dims(gt_stroke_img_large, axis=-1) * (1.0 - color_rgb / 255.0)
                canvas_color_with_moving = canvas_color_with_moving * np.expand_dims((1.0 - gt_stroke_img_large),
                                                                                     axis=-1) + color_stroke  # (H, W, 3)

                if pen_state == 0:
                    canvas_color = canvas_color * np.expand_dims((1.0 - gt_stroke_img_large),
                                                                 axis=-1) + color_stroke  # (H, W, 3)

            # update cursor_pos based on hps.cursor_type
            new_cursor_offsets = stroke_params[2:4] * (curr_window_size / 2.0)  # (1, 6), patch-level
            new_cursor_offset_next = new_cursor_offsets

            # important!!!
            new_cursor_offset_next = np.concatenate([new_cursor_offset_next[1:2], new_cursor_offset_next[0:1]], axis=-1)

            cursor_pos_large = cursor_pos * float(image_size)

            stroke_position_next = cursor_pos_large + new_cursor_offset_next  # (2), large-level

            if cursor_type == 'next':
                cursor_pos_large = stroke_position_next  # (2), large-level
            else:
                raise Exception('Unknown cursor_type')

            cursor_pos_large = np.minimum(np.maximum(cursor_pos_large, 0.0), float(image_size - 1))  # (2), large-level
            cursor_pos = cursor_pos_large / float(image_size)

            frames.append(canvas.copy())

    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = np.round((1.0 - canvas) * 255.0).astype(np.uint8)  # [0-stroke, 255-BG]

    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, save_filename)
    canvas_img = Image.fromarray(canvas, 'L')
    canvas_img.save(save_path, 'PNG')

    if save_seq:
        seq_save_root = os.path.join(save_root, 'seq', save_filename[:-4])
        os.makedirs(seq_save_root, exist_ok=True)
        for len_i in range(len(frames)):
            frame = frames[len_i]
            frame = np.round((1.0 - frame) * 255.0).astype(np.uint8)
            save_path = os.path.join(seq_save_root, str(len_i) + '.png')
            frame_img = Image.fromarray(frame, 'L')
            frame_img.save(save_path, 'PNG')

    if draw_order:
        order_save_root = os.path.join(save_root, 'order')
        order_comp_save_root = os.path.join(save_root, 'order-compare')
        os.makedirs(order_save_root, exist_ok=True)
        os.makedirs(order_comp_save_root, exist_ok=True)

        canvas_color = 255 - np.round(canvas_color * 255.0).astype(np.uint8)
        canvas_color_img = Image.fromarray(canvas_color, 'RGB')
        save_path = os.path.join(order_save_root, save_filename)
        canvas_color_img.save(save_path, 'PNG')

        canvas_color_with_moving = 255 - np.round(canvas_color_with_moving * 255.0).astype(np.uint8)

        # comparsions
        rows = 2
        cols = 3
        plt.figure(figsize=(5 * cols, 5 * rows))

        plt.subplot(rows, cols, 1)
        plt.title('Input', fontsize=12)
        # plt.axis('off')
        input_rgb = input_img
        plt.imshow(input_rgb)

        # plt.subplot(rows, cols, 2)
        # plt.title('GT', fontsize=12)
        # # plt.axis('off')
        # gt_rgb = np.stack([gt_img for _ in range(3)], axis=2)
        # plt.imshow(gt_rgb)

        plt.subplot(rows, cols, 2)
        plt.title('Sketch', fontsize=12)
        # plt.axis('off')
        canvas_rgb = np.stack([canvas for _ in range(3)], axis=2)
        plt.imshow(canvas_rgb)

        plt.subplot(rows, cols, 4)
        plt.title('Sketch Order', fontsize=12)
        # plt.axis('off')
        plt.imshow(canvas_color)

        plt.subplot(rows, cols, 5)
        plt.title('Sketch Order with moving', fontsize=12)
        # plt.axis('off')
        plt.imshow(canvas_color_with_moving)

        plt.subplot(rows, cols, 6)
        plt.title('Order', fontsize=12)
        plt.axis('off')

        img_h = 5
        img_w = 10
        color_array = np.zeros([len(color_rgb_set) * img_h, img_w, 3], dtype=np.uint8)
        for i in range(len(color_rgb_set)):
            color_array[i * img_h: i * img_h + img_h, :, :] = color_rgb_set[i]

        plt.imshow(color_array)

        comp_save_path = os.path.join(order_comp_save_root, save_filename)
        plt.savefig(comp_save_path)
        plt.close()
        # plt.show()


def update_hyperparams(model_params, model_base_dir, model_name, infer_dataset):
    with tf.gfile.Open(os.path.join(model_base_dir, model_name, 'model_config.json'), 'r') as f:
        data = json.load(f)

    ignored_keys = ['image_size_small', 'image_size_large', 'z_size', 'raster_perc_loss_layer', 'raster_loss_wk',
                    'decreasing_sn', 'raster_loss_weight']
    for name in model_params._hparam_types.keys():
        if name not in data and name not in ignored_keys:
            raise Exception(name, 'not in model_config.json')

    assert data['resize_method'] == 'AREA'
    data['data_set'] = infer_dataset
    fix_list = ['use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)

    pop_keys = ['gpus', 'image_size', 'resolution_type', 'loop_per_gpu', 'stroke_num_loss_weight_end',
                'perc_loss_fuse_type',
                'early_pen_length', 'early_pen_loss_type', 'early_pen_loss_weight',
                'increase_start_steps', 'perc_loss_layers', 'sn_loss_type', 'photo_prob_end_step',
                'sup_weight', 'gan_weight', 'base_raster_loss_base_type']
    for pop_key in pop_keys:
        if pop_key in data.keys():
            data.pop(pop_key)

    model_params.parse_json(json.dumps(data))

    return model_params
