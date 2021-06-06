import os
import argparse
import numpy as np
from xml.dom import minidom


def write_svg_1(path_list, img_size, save_path):
    ''' A long curve consisting of several strokes forms a path. '''
    impl = minidom.getDOMImplementation()

    doc = impl.createDocument(None, None, None)

    rootElement = doc.createElement('svg')
    rootElement.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    rootElement.setAttribute('height', str(img_size))
    rootElement.setAttribute('width', str(img_size))

    path_num = len(path_list)
    for path_i in range(path_num):
        path_items = path_list[path_i]

        assert len(path_items) > 0
        if len(path_items) == 1:
            continue

        childElement = doc.createElement('path')
        childElement.setAttribute('id', 'curve_' + str(path_i))
        childElement.setAttribute('stroke', '#000000')
        childElement.setAttribute('stroke-width', '3.5')
        childElement.setAttribute('stroke-linejoin', 'round')
        childElement.setAttribute('stroke-linecap', 'round')
        childElement.setAttribute('fill', 'none')

        command_str = ''
        for stroke_i, stroke_item in enumerate(path_items):
            if stroke_i == 0:
                command_str += 'M '
                stroke_position = stroke_item[0]
                command_str += str(stroke_position[0]) + ', ' + str(stroke_position[1]) + ' '
            else:
                command_str += 'Q '
                ctrl_position, stroke_position, stroke_width = stroke_item[0], stroke_item[1], stroke_item[2]

                ctrl_position_0 = last_position[0] + (stroke_position[0] - last_position[0]) * ctrl_position[1]
                ctrl_position_1 = last_position[1] + (stroke_position[1] - last_position[1]) * ctrl_position[0]

                command_str += str(ctrl_position_0) + ', ' + str(ctrl_position_1) + ', ' + \
                               str(stroke_position[0]) + ', ' + str(stroke_position[1]) + ' '

            last_position = stroke_position

        childElement.setAttribute('d', command_str)
        rootElement.appendChild(childElement)

    doc.appendChild(rootElement)

    f = open(save_path, 'w')
    doc.writexml(f, addindent='  ', newl='\n')
    f.close()


def write_svg_2(path_list, img_size, save_path):
    ''' A single stroke forms a path. '''
    impl = minidom.getDOMImplementation()

    doc = impl.createDocument(None, None, None)

    rootElement = doc.createElement('svg')
    rootElement.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    rootElement.setAttribute('height', str(img_size))
    rootElement.setAttribute('width', str(img_size))

    path_num = len(path_list)
    for path_i in range(path_num):
        path_items = path_list[path_i]

        assert len(path_items) > 0
        if len(path_items) == 1:
            continue

        for stroke_i, stroke_item in enumerate(path_items):
            if stroke_i == 0:
                last_position = stroke_item[0]
            else:
                childElement = doc.createElement('path')
                childElement.setAttribute('id', 'curve_' + str(path_i))
                childElement.setAttribute('stroke', '#000000')
                childElement.setAttribute('stroke-linejoin', 'round')
                childElement.setAttribute('stroke-linecap', 'round')
                childElement.setAttribute('fill', 'none')

                command_str = 'M ' + str(last_position[0]) + ', ' + str(last_position[1]) + ' '
                command_str += 'Q '

                ctrl_position, stroke_position, stroke_width = stroke_item[0], stroke_item[1], stroke_item[2]

                ctrl_position_0 = last_position[0] + (stroke_position[0] - last_position[0]) * ctrl_position[1]
                ctrl_position_1 = last_position[1] + (stroke_position[1] - last_position[1]) * ctrl_position[0]

                command_str += str(ctrl_position_0) + ', ' + str(ctrl_position_1) + ', ' + \
                               str(stroke_position[0]) + ', ' + str(stroke_position[1]) + ' '

                last_position = stroke_position

                childElement.setAttribute('d', command_str)
                childElement.setAttribute('stroke-width', str(stroke_width * img_size / 1.66))
                rootElement.appendChild(childElement)

    doc.appendChild(rootElement)

    f = open(save_path, 'w')
    doc.writexml(f, addindent='  ', newl='\n')
    f.close()


def convert_strokes_to_svg(data, init_cursor, image_size, infer_lengths, init_width, save_path, svg_type,
                           cursor_type='next', min_window_size=32, raster_size=128):
    """
    :param data: (N_strokes, 7): flag, x_c, y_c, dx, dy, r, ds
    :return:
    """
    cursor_idx = 0

    absolute_strokes = []
    absolute_strokes_path = []

    if init_cursor.ndim == 1:
        init_cursor = [init_cursor]

    for round_idx in range(len(infer_lengths)):
        round_length = infer_lengths[round_idx]

        cursor_pos = init_cursor[cursor_idx]  # (2)
        cursor_idx += 1

        cursor_pos_large = cursor_pos * float(image_size)

        if len(absolute_strokes_path) > 0:
            absolute_strokes.append(absolute_strokes_path)
        absolute_strokes_path = [[cursor_pos_large]]

        prev_width = init_width
        prev_scaling = 1.0
        prev_window_size = float(raster_size)  # (1)

        for round_inner_i in range(round_length):
            stroke_idx = np.sum(infer_lengths[:round_idx]).astype(np.int32) + round_inner_i

            curr_window_size_raw = prev_scaling * prev_window_size
            curr_window_size_raw = np.maximum(curr_window_size_raw, min_window_size)
            curr_window_size_raw = np.minimum(curr_window_size_raw, image_size)
            # curr_window_size = int(round(curr_window_size_raw))  # ()

            stroke_params = data[stroke_idx, 1:]  # (6)
            pen_state = data[stroke_idx, 0]

            next_width = stroke_params[4]
            next_scaling = stroke_params[5]

            next_width_abs = next_width * curr_window_size_raw / float(image_size)

            prev_scaling = next_scaling
            prev_window_size = curr_window_size_raw

            # update cursor_pos based on hps.cursor_type
            new_cursor_offsets = stroke_params[2:4] * (float(curr_window_size_raw) / 2.0)  # (1, 6), patch-level
            new_cursor_offset_next = new_cursor_offsets

            # important!!!
            new_cursor_offset_next = np.concatenate([new_cursor_offset_next[1:2], new_cursor_offset_next[0:1]], axis=-1)
            cursor_pos_large = cursor_pos * float(image_size)
            stroke_position_next = cursor_pos_large + new_cursor_offset_next  # (2), large-level

            if pen_state == 0:
                absolute_strokes_path.append([stroke_params[0:2], stroke_position_next, next_width_abs])
            else:
                absolute_strokes.append(absolute_strokes_path)
                absolute_strokes_path = [[stroke_position_next]]

            if cursor_type == 'next':
                cursor_pos_large = stroke_position_next  # (2), large-level
            else:
                raise Exception('Unknown cursor_type')

            cursor_pos_large = np.minimum(np.maximum(cursor_pos_large, 0.0), float(image_size - 1))  # (2), large-level
            cursor_pos = cursor_pos_large / float(image_size)

    absolute_strokes.append(absolute_strokes_path)

    if svg_type == 'cluster':
        write_svg_1(absolute_strokes, image_size, save_path)
    elif svg_type == 'single':
        write_svg_2(absolute_strokes, image_size, save_path)
    else:
        raise Exception('Unknown svg_type', svg_type)


def data_convert_to_absolute(npz_path, svg_type):
    assert npz_path != ''
    assert svg_type in ['single', 'cluster']

    min_window_size = 32
    raster_size = 128

    split_idx = npz_path.rfind('/')
    if split_idx == -1:
        file_base = './'
        file_name = npz_path[:-4]
    else:
        file_base = npz_path[:npz_path.rfind('/')]
        file_name = npz_path[npz_path.rfind('/') + 1: -4]

    svg_data_base = os.path.join(file_base, file_name)
    os.makedirs(svg_data_base, exist_ok=True)

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

    save_path = os.path.join(svg_data_base, str(svg_type) + '.svg')

    convert_strokes_to_svg(strokes_data, init_cursors, image_size, round_lengths, init_width,
                           min_window_size=min_window_size, raster_size=raster_size, save_path=save_path,
                           svg_type=svg_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='', help="define a npz path")
    parser.add_argument('--svg_type', '-st', type=str, choices=['single', 'cluster'], default='single',
                        help="svg type")
    args = parser.parse_args()

    data_convert_to_absolute(args.file, args.svg_type)
