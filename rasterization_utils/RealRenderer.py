import numpy as np
import gizeh


class GizehRasterizor(object):
    def __init__(self):
        self.name = 'GizehRasterizor'

    def get_line_array_v2(self, image_size, seq_strokes, stroke_width, is_bin=True):
        """
        :param p1: (x, y)
        :param p2: (x, y)
        :return: line_arr: (image_size, image_size), {0, 1}, 0 for BG and 1 for strokes
        """
        surface = gizeh.Surface(width=image_size, height=image_size)  # in pixels
        shape_list = []
        for seq_i in range(len(seq_strokes) - 1):
            p1, p2 = seq_strokes[seq_i, :2], seq_strokes[seq_i + 1, :2]
            pen_state = seq_strokes[seq_i, 2]

            if pen_state == 0.0:
                line = gizeh.polyline(points=[p1, p2], stroke_width=stroke_width, stroke=(1, 1, 1), fill=(0, 0, 0))
                shape_list.append(line)

        group = gizeh.Group(shape_list)
        group.draw(surface)

        # Now export the surface
        line_arr = surface.get_npimage()[:, :, 0]  # returns a (width x height x 3) numpy array

        if is_bin:
            line_arr[line_arr <= 128] = 0
            line_arr[line_arr != 0] = 1  # (image_size, image_size)
        else:
            line_arr = np.array(line_arr, dtype=np.float32) / 255.0

        return line_arr

    def get_line_array(self, p1, p2, image_size, stroke_width, is_bin=True):
        """
        :param p1: (x, y)
        :param p2: (x, y)
        :return: line_arr: (image_size, image_size), {0, 1}, 0 for BG and 1 for strokes
        """
        surface = gizeh.Surface(width=image_size, height=image_size)  # in pixels
        line = gizeh.polyline(points=[p1, p2], stroke_width=stroke_width, stroke=(1, 1, 1), fill=(0, 0, 0))
        line.draw(surface)

        # Now export the surface
        line_arr = surface.get_npimage()[:, :, 0]  # returns a (width x height x 3) numpy array

        if is_bin:
            line_arr[line_arr <= 128] = 0
            line_arr[line_arr != 0] = 1  # (image_size, image_size)
        else:
            line_arr = np.array(line_arr, dtype=np.float32) / 255.0

        return line_arr

    def load_sketch_images_on_the_fly_v2(self, image_size, norm_strokes3, stroke_width, is_bin=True):
        """
        :param norm_strokes3: list (N_sketches,), each with (N_points, 3)
        :return: list (N_sketches,), each with (raster_size, raster_size), 0-BG and 1-strokes
        """
        assert type(norm_strokes3) is list
        sketch_imgs_list = []
        for stroke_i in range(len(norm_strokes3)):
            seq_strokes3 = norm_strokes3[stroke_i]  # (N_points, 3)
            sketch_img = self.get_line_array_v2(image_size, seq_strokes3, stroke_width=stroke_width, is_bin=is_bin)
            sketch_img = np.clip(sketch_img, 0.0, 1.0)  # (image_size, image_size), 0 for BG and 1 for strokes
            sketch_imgs_list.append(sketch_img)

        return sketch_imgs_list

    def load_sketch_images_on_the_fly(self, image_size, norm_strokes3, stroke_width, is_bin=True):
        """
        :param norm_strokes3: list (N_sketches,), each with (N_points, 3)
        :return: list (N_sketches,), each with (raster_size, raster_size), 0-BG and 1-strokes
        """
        assert type(norm_strokes3) is list
        sketch_imgs_list = []
        for stroke_i in range(len(norm_strokes3)):
            seq_strokes3 = norm_strokes3[stroke_i]  # (N_points, 3)
            seq_len = len(seq_strokes3)
            stroke_imgs_list = []

            for seq_i in range(seq_len - 1):
                stroke_img = self.get_line_array(seq_strokes3[seq_i, :2], seq_strokes3[seq_i + 1, :2], image_size,
                                                 stroke_width=stroke_width, is_bin=is_bin)
                pen_state = seq_strokes3[seq_i, 2]
                stroke_img = stroke_img.astype(np.float32) * (1. - pen_state)
                stroke_imgs_list.append(stroke_img)

            stroke_imgs_list = np.stack(stroke_imgs_list,
                                        axis=-1)  # (image_size, image_size, seq_len-1), 0 for BG and 1 for strokes
            stroke_imgs_list = np.sum(stroke_imgs_list, axis=-1)
            stroke_imgs_list = np.clip(stroke_imgs_list, 0.0, 1.0)  # (image_size, image_size), 0 for BG and 1 for strokes
            sketch_imgs_list.append(stroke_imgs_list)

        return sketch_imgs_list

    def normalize_coordinate_np(self, sx, sy, image_size, raster_padding=10.0):
        """
        Convert offset to normalized absolute points. The numpy version as in NeuralRasterizor.
        :param sx: (N, seq_len)
        :param sy: (N, seq_len)
        :return:
        """
        seq_len = sx.shape[1]

        # transfer to abs points
        abs_x = np.cumsum(sx, axis=1)  # (N, seq_len)
        abs_y = np.cumsum(sy, axis=1)

        min_x = np.min(abs_x, axis=1, keepdims=True)  # (N, 1)
        max_x = np.max(abs_x, axis=1, keepdims=True)
        min_y = np.min(abs_y, axis=1, keepdims=True)
        max_y = np.max(abs_y, axis=1, keepdims=True)

        # transform to positive coordinate
        abs_x = np.subtract(abs_x, np.tile(min_x, [1, seq_len]))  # (N, seq_len)
        abs_y = np.subtract(abs_y, np.tile(min_y, [1, seq_len]))

        # scaling to [0.0, raster_size - 2 * padding - 1]
        bbox_w = np.squeeze(np.subtract(max_x, min_x), axis=-1)  # (N)
        bbox_h = np.squeeze(np.subtract(max_y, min_y), axis=-1)

        unpad_raster_size = (image_size - 1.0) - 2.0 * raster_padding
        scaling = np.divide(unpad_raster_size, np.maximum(bbox_w, bbox_h))  # (N)
        scaling_tile = np.tile(np.expand_dims(scaling, axis=-1), [1, seq_len])  # (N, seq_len)
        abs_x = np.multiply(abs_x, scaling_tile)  # (N, seq_len)
        abs_y = np.multiply(abs_y, scaling_tile)

        # add padding
        abs_x = np.add(abs_x, raster_padding)  # (N, seq_len)
        abs_y = np.add(abs_y, raster_padding)

        # transform to the middle
        trans_x = np.divide(np.subtract(unpad_raster_size, np.multiply(bbox_w, scaling)), 2.0)  # (N)
        trans_y = np.divide(np.subtract(unpad_raster_size, np.multiply(bbox_h, scaling)), 2.0)
        trans_x = np.tile(np.expand_dims(trans_x, axis=-1), [1, seq_len])  # (N, seq_len)
        trans_y = np.tile(np.expand_dims(trans_y, axis=-1), [1, seq_len])  # (N, seq_len)
        abs_x = np.add(abs_x, trans_x)  # (N, seq_len)
        abs_y = np.add(abs_y, trans_y)

        return abs_x, abs_y

    def normalize_strokes_np(self, strokes_list, image_size):
        """

        :param strokes_list: list (N_sketches,), each with (N_points, 3)
        :return:
        """
        assert type(strokes_list) is list

        rst_list = []
        for i in range(len(strokes_list)):
            strokes_data = strokes_list[i]  # (N_points, 3)
            norm_x, norm_y = self.normalize_coordinate_np(np.expand_dims(strokes_data[:, 0], axis=0),
                                                          np.expand_dims(strokes_data[:, 1], axis=0),
                                                          image_size)  # (1, N_points)
            norm_strokes_data = np.stack([norm_x[0], norm_y[0], strokes_data[:, 2]], axis=-1)  # (N_points, 3)
            rst_list.append(norm_strokes_data)
        return rst_list

    def raster_func(self, input_data, image_size, stroke_width, is_bin=True, version='v2'):
        """
        :param input_data: (N_sketches,), each with (N_points, 3)
        :return: raster_image_array: list (N_sketches,), each with (raster_size, raster_size), 0-BG and 1-strokes
        """
        norm_test_strokes3 = self.normalize_strokes_np(input_data, image_size)
        if version == 'v1':
            raster_image_array = self.load_sketch_images_on_the_fly(image_size, norm_test_strokes3, stroke_width, is_bin=is_bin)
        else:
            raster_image_array = self.load_sketch_images_on_the_fly_v2(image_size, norm_test_strokes3, stroke_width, is_bin=is_bin)

        return raster_image_array
