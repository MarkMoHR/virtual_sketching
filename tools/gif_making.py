import os
import sys
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.append('./')
from utils import draw, image_pasting_v3_testing
from model_common_test import DiffPastingV3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def add_scaling_visualization(canvas_images, cursor, window_size, image_size):
    """
    :param canvas_images: (N, H, W, 3)
    :param cursor:
    :param window_size:
    :param image_size:
    :return:
    """
    cursor_pos = cursor * float(image_size)
    cursor_x, cursor_y = int(round(cursor_pos[0])), int(round(cursor_pos[1]))  # in large size

    vis_color = [255, 0, 0]
    cursor_width = 3
    box_width = 2

    canvas_imgs = 255 - np.round(canvas_images * 255.0).astype(np.uint8)

    # add cursor visualization
    canvas_imgs[:, cursor_y - cursor_width: cursor_y + cursor_width, cursor_x - cursor_width: cursor_x + cursor_width, :] = vis_color

    # add box visualization
    up = max(0, cursor_y - window_size // 2)
    down = min(image_size, cursor_y + window_size // 2)
    left = max(0, cursor_x - window_size // 2)
    right = min(image_size, cursor_x + window_size // 2)
    # up = cursor_y - window_size // 2
    # down = cursor_y + window_size // 2
    # left = cursor_x - window_size // 2
    # right = cursor_x + window_size // 2

    if up > 0:
        canvas_imgs[:, up: up + box_width, left: right, :] = vis_color
    if down < image_size:
        canvas_imgs[:, down - box_width: down, left: right, :] = vis_color
    if left > 0:
        canvas_imgs[:, up: down, left: left + box_width, :] = vis_color
    if right < image_size:
        canvas_imgs[:, up: down, right - box_width: right, :] = vis_color
    return canvas_imgs


def make_gif(sess, pasting_func, data, init_cursor, image_size, infer_lengths, init_width,
             save_base,
             cursor_type='next', min_window_size=32, raster_size=128, add_box=True):
    """
    :param data: (N_strokes, 9): flag, x0, y0, x1, y1, x2, y2, r0, r2
    :return:
    """
    canvas = np.zeros((image_size, image_size), dtype=np.float32)  # [0.0-BG, 1.0-stroke]
    gif_frames = []

    cursor_idx = 0

    if init_cursor.ndim == 1:
        init_cursor = [init_cursor]

    for round_idx in range(len(infer_lengths)):
        print('Making progress', round_idx + 1, '/', len(infer_lengths))
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
            curr_window_size = int(round(curr_window_size_raw))  # ()

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

            if pen_state == 0:
                canvas += gt_stroke_img_large  # [0.0-BG, 1.0-stroke]

            canvas_rgb = np.stack([np.clip(canvas, 0.0, 1.0) for _ in range(3)], axis=-1)

            if add_box:
                vis_inputs = np.expand_dims(canvas_rgb, axis=0)
                vis_outputs = add_scaling_visualization(vis_inputs, cursor_pos, curr_window_size, image_size)
                canvas_vis = vis_outputs[0]
            else:
                canvas_vis = canvas_rgb

            canvas_vis_png = Image.fromarray(canvas_vis, 'RGB')
            gif_frames.append(canvas_vis_png)

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

    print('Saving to GIF ...')
    save_path = os.path.join(save_base, 'dynamic.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)


def gif_making(npz_path):
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

    gif_base = os.path.join(file_base, file_name)
    os.makedirs(gif_base, exist_ok=True)

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

    make_gif(sess, paste_v3_func,
             strokes_data, init_cursors, image_size, round_lengths, init_width,
             gif_base,
             min_window_size=min_window_size, raster_size=raster_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='', help="define a npz path")
    args = parser.parse_args()

    gif_making(args.file)
