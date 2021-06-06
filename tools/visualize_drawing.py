import os
import sys
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.append('./')
from utils import get_colors, draw, image_pasting_v3_testing
from model_common_test import DiffPastingV3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def display_strokes_final(sess, pasting_func, data, init_cursor, image_size, infer_lengths, init_width,
                          save_base,
                          cursor_type='next', min_window_size=32, raster_size=128):
    """
    :param data: (N_strokes, 9): flag, x0, y0, x1, y1, x2, y2, r0, r2
    :return:
    """
    canvas = np.zeros((image_size, image_size), dtype=np.float32)  # [0.0-BG, 1.0-stroke]
    drawn_region = np.zeros_like(canvas)
    overlap_region = np.zeros_like(canvas)
    canvas_color_with_overlap = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas_color_wo_overlap = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas_color_with_moving = np.zeros((image_size, image_size, 3), dtype=np.float32)

    cursor_idx = 0

    if init_cursor.ndim == 1:
        init_cursor = [init_cursor]

    stroke_count = len(data)
    color_rgb_set = get_colors(stroke_count)  # list of (3,) in [0, 255]
    color_idx = 0

    valid_stroke_count = stroke_count - np.sum(data[:, 0]).astype(np.int32) + len(init_cursor)
    valid_color_rgb_set = get_colors(valid_stroke_count)  # list of (3,) in [0, 255]
    valid_color_idx = -1

    # print('Drawn stroke number', valid_stroke_count)
    # print('    flag  x1\t\t y1\t\t x2\t\t y2\t\t r2\t\t s2')

    for round_idx in range(len(infer_lengths)):
        round_length = infer_lengths[round_idx]

        cursor_pos = init_cursor[cursor_idx]  # (2)
        cursor_idx += 1

        prev_width = init_width
        prev_scaling = 1.0
        prev_window_size = float(raster_size)  # (1)

        for round_inner_i in range(round_length):
            stroke_idx = np.sum(infer_lengths[:round_idx]).astype(np.int32) + round_inner_i

            curr_window_size_raw = prev_scaling * prev_window_size
            curr_window_size_raw = np.maximum(curr_window_size_raw, min_window_size)
            curr_window_size_raw = np.minimum(curr_window_size_raw, image_size)

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
            next_window_size = next_scaling * curr_window_size_raw
            next_window_size = np.maximum(next_window_size, min_window_size)
            next_window_size = np.minimum(next_window_size, image_size)

            prev_width = next_width * curr_window_size_raw / next_window_size
            prev_scaling = next_scaling
            prev_window_size = curr_window_size_raw

            f = stroke_params_proc.tolist()  # (8)
            f += [1.0, 1.0]
            gt_stroke_img = draw(f)  # (H, W), [0.0-stroke, 1.0-BG]

            gt_stroke_img_large = image_pasting_v3_testing(1.0 - gt_stroke_img, cursor_pos,
                                                           image_size,
                                                           curr_window_size_raw,
                                                           pasting_func, sess)  # [0.0-BG, 1.0-stroke]

            is_overlap = False

            if pen_state == 0:
                canvas += gt_stroke_img_large  # [0.0-BG, 1.0-stroke]

                curr_drawn_stroke_region = np.zeros_like(gt_stroke_img_large)
                curr_drawn_stroke_region[gt_stroke_img_large > 0.5] = 1
                intersection = drawn_region * curr_drawn_stroke_region
                # regard stroke with >50% overlap area as overlaped stroke
                if np.sum(intersection) / np.sum(curr_drawn_stroke_region) > 0.5:
                    # enlarge the stroke a bit for better visualization
                    overlap_region[gt_stroke_img_large > 0] += 1
                    is_overlap = True

                drawn_region[gt_stroke_img_large > 0.5] = 1

            color_rgb = color_rgb_set[color_idx]  # (3) in [0, 255]
            color_idx += 1

            color_rgb = np.reshape(color_rgb, (1, 1, 3)).astype(np.float32)
            color_stroke = np.expand_dims(gt_stroke_img_large, axis=-1) * (1.0 - color_rgb / 255.0)
            canvas_color_with_moving = canvas_color_with_moving * np.expand_dims((1.0 - gt_stroke_img_large),
                                                                                 axis=-1) + color_stroke  # (H, W, 3)

            if pen_state == 0:
                valid_color_idx += 1

            if pen_state == 0:
                valid_color_rgb = valid_color_rgb_set[valid_color_idx]  # (3) in [0, 255]
                # valid_color_idx += 1

                valid_color_rgb = np.reshape(valid_color_rgb, (1, 1, 3)).astype(np.float32)
                valid_color_stroke = np.expand_dims(gt_stroke_img_large, axis=-1) * (1.0 - valid_color_rgb / 255.0)
                canvas_color_with_overlap = canvas_color_with_overlap * np.expand_dims((1.0 - gt_stroke_img_large),
                                                                                       axis=-1) + valid_color_stroke  # (H, W, 3)
                if not is_overlap:
                    canvas_color_wo_overlap = canvas_color_wo_overlap * np.expand_dims((1.0 - gt_stroke_img_large),
                                                                                       axis=-1) + valid_color_stroke  # (H, W, 3)

            # update cursor_pos based on hps.cursor_type
            new_cursor_offsets = stroke_params[2:4] * (float(curr_window_size_raw) / 2.0)  # (1, 6), patch-level
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

    canvas_rgb = np.stack([np.clip(canvas, 0.0, 1.0) for _ in range(3)], axis=-1)
    canvas_black = 255 - np.round(canvas_rgb * 255.0).astype(np.uint8)
    canvas_color_with_overlap = 255 - np.round(canvas_color_with_overlap * 255.0).astype(np.uint8)
    canvas_color_wo_overlap = 255 - np.round(canvas_color_wo_overlap * 255.0).astype(np.uint8)
    canvas_color_with_moving = 255 - np.round(canvas_color_with_moving * 255.0).astype(np.uint8)

    canvas_black_png = Image.fromarray(canvas_black, 'RGB')
    canvas_black_save_path = os.path.join(save_base, 'output_rendered.png')
    canvas_black_png.save(canvas_black_save_path, 'PNG')

    canvas_color_png = Image.fromarray(canvas_color_with_overlap, 'RGB')
    canvas_color_save_path = os.path.join(save_base, 'output_order_with_overlap.png')
    canvas_color_png.save(canvas_color_save_path, 'PNG')

    canvas_color_wo_png = Image.fromarray(canvas_color_wo_overlap, 'RGB')
    canvas_color_wo_save_path = os.path.join(save_base, 'output_order_wo_overlap.png')
    canvas_color_wo_png.save(canvas_color_wo_save_path, 'PNG')

    canvas_color_m_png = Image.fromarray(canvas_color_with_moving, 'RGB')
    canvas_color_m_save_path = os.path.join(save_base, 'output_order_with_moving.png')
    canvas_color_m_png.save(canvas_color_m_save_path, 'PNG')


def visualize_drawing(npz_path):
    assert npz_path != ''

    min_window_size = 32
    raster_size = 128

    split_idx = npz_path.rfind('/')
    if split_idx == -1:
        file_base = './'
        file_name = npz_path[:-4]
    else:
        file_base = npz_path[:npz_path.rfind('/')]
        file_name = npz_path[npz_path.rfind('/') + 1: -4]

    regenerate_base = os.path.join(file_base, file_name)
    os.makedirs(regenerate_base, exist_ok=True)

    # differentiable pasting graph
    paste_v3_func = DiffPastingV3(raster_size)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    strokes_data = data['strokes_data']
    init_cursors = data['init_cursors']
    image_size = data['image_size']
    round_length = data['round_length']
    init_width = data['init_width']

    if round_length.ndim == 0:
        round_lengths = [round_length]
    else:
        round_lengths = round_length

    # print('round_lengths', round_lengths)

    print('Processing ...')
    display_strokes_final(sess, paste_v3_func,
                          strokes_data, init_cursors, image_size, round_lengths, init_width,
                          regenerate_base,
                          min_window_size=min_window_size, raster_size=raster_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='', help="define a npz path")
    args = parser.parse_args()

    visualize_drawing(args.file)
