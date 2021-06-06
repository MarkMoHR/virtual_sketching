import os
import math
import random
import scipy.io
import numpy as np
import tensorflow as tf
from PIL import Image

from rasterization_utils.RealRenderer import GizehRasterizor as RealRenderer


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())


class GeneralRawDataLoader(object):
    def __init__(self,
                 image_path,
                 raster_size,
                 test_dataset):
        self.image_path = image_path
        self.raster_size = raster_size
        self.test_dataset = test_dataset

    def get_test_image(self, random_cursor=True, init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        input_image_data, image_size_test = self.gen_input_images(self.image_path)
        input_image_data = np.array(input_image_data,
                                    dtype=np.float32)  # (1, image_size, image_size, (3)), [0.0-strokes, 1.0-BG]

        return input_image_data, \
               self.gen_init_cursors(input_image_data, random_cursor, init_cursor_on_undrawn_pixel, init_cursor_num), \
               image_size_test

    def gen_input_images(self, image_path):
        img = Image.open(image_path).convert('RGB')
        height, width = img.height, img.width
        max_dim = max(height, width)

        img = np.array(img, dtype=np.uint8)

        if height != width:
            # Padding to a square image
            if self.test_dataset == 'clean_line_drawings':
                pad_value = [255, 255, 255]
            elif self.test_dataset == 'faces':
                pad_value = [0, 0, 0]
            else:
                # TODO: find better padding pixel value
                pad_value = img[height - 10, width - 10]

            img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            pad_width = max_dim - width
            pad_height = max_dim - height

            pad_img_r = np.pad(img_r, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[0])
            pad_img_g = np.pad(img_g, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[1])
            pad_img_b = np.pad(img_b, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[2])
            image_array = np.stack([pad_img_r, pad_img_g, pad_img_b], axis=-1)
        else:
            image_array = img

        if self.test_dataset == 'faces' and max_dim != 256:
            image_array_resize = Image.fromarray(image_array, 'RGB')
            image_array_resize = image_array_resize.resize(size=(256, 256), resample=Image.BILINEAR)
            image_array = np.array(image_array_resize, dtype=np.uint8)

        assert image_array.shape[0] == image_array.shape[1]
        img_size = image_array.shape[0]
        image_array = image_array.astype(np.float32)
        if self.test_dataset == 'clean_line_drawings':
            image_array = image_array[:, :, 0] / 255.0  # [0.0-stroke, 1.0-BG]
        else:
            image_array = image_array / 255.0  # [0.0-stroke, 1.0-BG]
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, img_size

    def crop_patch(self, image, center, image_size, crop_size):
        x0 = center[0] - crop_size // 2
        x1 = x0 + crop_size
        y0 = center[1] - crop_size // 2
        y1 = y0 + crop_size
        x0 = max(0, min(x0, image_size))
        y0 = max(0, min(y0, image_size))
        x1 = max(0, min(x1, image_size))
        y1 = max(0, min(y1, image_size))
        patch = image[y0:y1, x0:x1]
        return patch

    def gen_init_cursor_single(self, sketch_image, init_cursor_on_undrawn_pixel, misalign_size=3):
        # sketch_image: [0.0-stroke, 1.0-BG]
        image_size = sketch_image.shape[0]
        if np.sum(1.0 - sketch_image) == 0:
            center = np.zeros((2), dtype=np.int32)
            return center
        else:
            while True:
                center = np.random.randint(0, image_size, size=(2))  # (2), in large size
                patch = 1.0 - self.crop_patch(sketch_image, center, image_size, self.raster_size)
                if np.sum(patch) != 0:
                    if not init_cursor_on_undrawn_pixel:
                        return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)
                    else:
                        center_patch = 1.0 - self.crop_patch(sketch_image, center, image_size, misalign_size)
                        if np.sum(center_patch) != 0:
                            return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)

    def gen_init_cursors(self, sketch_data, random_pos=True, init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        init_cursor_batch_list = []
        for cursor_i in range(init_cursor_num):
            if random_pos:
                init_cursor_batch = []
                for i in range(len(sketch_data)):
                    sketch_image = sketch_data[i].copy().astype(np.float32)  # [0.0-stroke, 1.0-BG]
                    center = self.gen_init_cursor_single(sketch_image, init_cursor_on_undrawn_pixel)
                    init_cursor_batch.append(center)

                init_cursor_batch = np.stack(init_cursor_batch, axis=0)  # (N, 2)
            else:
                raise Exception('Not finished')
            init_cursor_batch_list.append(init_cursor_batch)

        if init_cursor_num == 1:
            init_cursor_batch = init_cursor_batch_list[0]
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=1).astype(np.float32)  # (N, 1, 2)
        else:
            init_cursor_batch = np.stack(init_cursor_batch_list, axis=1)  # (N, init_cursor_num, 2)
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=2).astype(
                np.float32)  # (N, init_cursor_num, 1, 2)

        return init_cursor_batch


def load_dataset_testing(test_data_base_dir, test_dataset, test_img_name, model_params):
    assert test_dataset in ['clean_line_drawings', 'rough_sketches', 'faces']
    img_path = os.path.join(test_data_base_dir, test_dataset, test_img_name)
    print('Loaded {} from {}'.format(img_path, test_dataset))

    eval_model_params = copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.batch_size = 1
    eval_model_params.model_mode = 'sample'

    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    test_set = GeneralRawDataLoader(img_path, eval_model_params.raster_size, test_dataset=test_dataset)

    result = [test_set, eval_model_params, sample_model_params]
    return result


class GeneralMultiObjectDataLoader(object):
    def __init__(self,
                 stroke3_data,
                 batch_size,
                 raster_size,
                 image_size_small,
                 image_size_large,
                 is_bin,
                 is_train):
        self.batch_size = batch_size  # minibatch size
        self.raster_size = raster_size
        self.image_size_small = image_size_small
        self.image_size_large = image_size_large
        self.is_bin = is_bin
        self.is_train = is_train

        self.num_batches = len(stroke3_data) // self.batch_size
        self.batch_idx = -1
        print('batch_size', batch_size, ', num_batches', self.num_batches)

        self.rasterizor = RealRenderer()
        self.memory_sketch_data_batch = []

        assert type(stroke3_data) is list
        self.preprocess_rand_data(stroke3_data)

    def preprocess_rand_data(self, stroke3):
        if self.is_train:
            random.shuffle(stroke3)
        self.stroke3_data = stroke3

    def cal_dist(self, posA, posB):
        return np.sqrt(np.sum(np.power(posA - posB, 2)))

    def invalid_position(self, pos, obj_size, pos_list, size_list):
        if len(pos_list) == 0:
            return False

        pos_a = pos
        size_a = obj_size
        for i in range(len(pos_list)):
            pos_b = pos_list[i]
            size_b = size_list[i]

            if self.cal_dist(pos_a, pos_b) < ((size_a + size_b) // 4):
                return True

        return False

    def get_object_info(self, image_size, vary_thickness=True, try_total_times=3):
        if image_size <= 172:
            obj_num = 1
            obj_thickness_list = [3]
        elif image_size <= 225:
            obj_num = random.randint(1, 2)
            obj_thickness_list = np.random.randint(3, 4 + 1, size=(obj_num))
        elif image_size <= 278:
            obj_num = 2
            obj_thickness_list = np.random.randint(3, 4 + 1, size=(obj_num))
        elif image_size <= 331:
            obj_num = random.randint(2, 3)
            while True:
                obj_thickness_list = np.random.randint(3, 5 + 1, size=(obj_num))
                if np.sum(obj_thickness_list) / obj_num != 5 and np.sum(obj_thickness_list) < 13:
                    break
        elif image_size <= 384:
            obj_num = 3
            while True:
                obj_thickness_list = np.random.randint(3, 5 + 1, size=(obj_num))
                if np.sum(obj_thickness_list) / obj_num != 5 and np.sum(obj_thickness_list) < 13:
                    break
        else:
            raise Exception('Invalid image_size', image_size)

        if not vary_thickness:
            num_item = len(obj_thickness_list)
            obj_thickness_list = [3 for _ in range(num_item)]

        obj_pos_list = []
        obj_size_list = []
        if obj_num == 1:
            obj_size_list.append(image_size)
            center = (image_size // 2, image_size // 2)
            obj_pos_list.append(center)
        else:
            for obj_i in range(obj_num):
                for try_i in range(try_total_times):
                    obj_size = random.randint(128, image_size * 3 // 4)
                    obj_center = np.random.randint(obj_size // 3, image_size - (obj_size // 3) + 1, size=(2))

                    if not self.invalid_position(obj_center, obj_size, obj_pos_list,
                                                 obj_size_list) or try_i == try_total_times - 1:
                        obj_pos_list.append(obj_center)
                        obj_size_list.append(obj_size)
                        break

        assert len(obj_size_list) == len(obj_pos_list) == len(obj_thickness_list) == obj_num
        return obj_num, obj_size_list, obj_pos_list, obj_thickness_list

    def object_pasting(self, obj_img, canvas_img, center):
        c_y, c_x = center[0], center[1]
        obj_size = obj_img.shape[0]
        canvas_size = canvas_img.shape[0]
        box_left = max(0, c_x - obj_size // 2)
        box_right = min(canvas_size, c_x + obj_size // 2)
        box_up = max(0, c_y - obj_size // 2)
        box_bottom = min(canvas_size, c_y + obj_size // 2)

        box_canvas = canvas_img[box_up: box_bottom, box_left: box_right]

        obj_box_up = box_up - (c_y - obj_size // 2)
        obj_box_left = box_left - (c_x - obj_size // 2)
        box_obj = obj_img[obj_box_up: obj_box_up + (box_bottom - box_up),
                  obj_box_left: obj_box_left + (box_right - box_left)]

        box_canvas += box_obj

        rst_canvas = np.copy(canvas_img)
        rst_canvas[box_up: box_bottom, box_left: box_right] = box_canvas
        rst_canvas = np.clip(rst_canvas, 0.0, 1.0)

        return rst_canvas

    def get_multi_object_image(self, img_size, vary_thickness):
        object_num, object_size_list, object_pos_list, object_thickness_list = self.get_object_info(
            img_size, vary_thickness=vary_thickness)

        canvas = np.zeros(shape=(img_size, img_size), dtype=np.float32)

        for obj_i in range(object_num):
            rand_idx = np.random.randint(0, len(self.stroke3_data))
            rand_stroke3 = self.stroke3_data[rand_idx]  # (N_points, 3)

            object_size = object_size_list[obj_i]
            object_enter = object_pos_list[obj_i]
            object_thickness = object_thickness_list[obj_i]

            stroke_image = self.gen_stroke_images([rand_stroke3], object_size, object_thickness)
            stroke_image = 1.0 - stroke_image[0]  # (image_size, image_size), [0.0-BG, 1.0-strokes]

            canvas = self.object_pasting(stroke_image, canvas, object_enter)  # [0.0-BG, 1.0-strokes]

        canvas = 1.0 - canvas  # [0.0-strokes, 1.0-BG]
        return canvas

    def get_batch_from_memory(self, memory_idx, vary_thickness, fixed_image_size=-1, random_cursor=True,
                              init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        if len(self.memory_sketch_data_batch) >= memory_idx + 1:
            sketch_data_batch = self.memory_sketch_data_batch[memory_idx]
            sketch_data_batch = np.expand_dims(sketch_data_batch,
                                               axis=0)  # (1, image_size, image_size), [0.0-strokes, 1.0-BG]
            image_size_rand = sketch_data_batch.shape[1]
        else:
            if fixed_image_size == -1:
                image_size_rand = random.randint(self.image_size_small, self.image_size_large)
            else:
                image_size_rand = fixed_image_size

            multi_obj_image = self.get_multi_object_image(image_size_rand, vary_thickness)  # [0.0-strokes, 1.0-BG]
            self.memory_sketch_data_batch.append(multi_obj_image)
            sketch_data_batch = np.expand_dims(multi_obj_image,
                                               axis=0)  # (1, image_size, image_size), [0.0-strokes, 1.0-BG]

        return None, sketch_data_batch, \
               self.gen_init_cursors(sketch_data_batch, random_cursor, init_cursor_on_undrawn_pixel, init_cursor_num), \
               image_size_rand

    def get_batch_multi_res(self, loop_num, vary_thickness, random_cursor=True,
                            init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        sketch_data_batch = []
        init_cursors_batch = []
        image_size_batch = []
        batch_size_per_loop = self.batch_size // loop_num
        for loop_i in range(loop_num):
            image_size_rand = random.randint(self.image_size_small, self.image_size_large)
            sketch_data_sub_batch = []
            for batch_i in range(batch_size_per_loop):
                multi_obj_image = self.get_multi_object_image(image_size_rand, vary_thickness)  # [0.0-strokes, 1.0-BG]
                sketch_data_sub_batch.append(multi_obj_image)
            sketch_data_sub_batch = np.stack(sketch_data_sub_batch,
                                             axis=0)  # (N, image_size, image_size), [0.0-strokes, 1.0-BG]

            init_cursors_sub_batch = self.gen_init_cursors(sketch_data_sub_batch, random_cursor,
                                                           init_cursor_on_undrawn_pixel, init_cursor_num)
            sketch_data_batch.append(sketch_data_sub_batch)
            init_cursors_batch.append(init_cursors_sub_batch)
            image_size_batch.append(image_size_rand)

        return None, \
               sketch_data_batch, \
               init_cursors_batch, \
               image_size_batch

    def gen_stroke_images(self, stroke3_list, image_size, stroke_width):
        """
        :param stroke3_list: list of (batch_size,), each with (N_points, 3)
        :param image_size:
        :return:
        """
        gt_image_array = self.rasterizor.raster_func(stroke3_list, image_size, stroke_width=stroke_width,
                                                     is_bin=self.is_bin, version='v2')
        gt_image_array = np.stack(gt_image_array, axis=0)
        gt_image_array = 1.0 - gt_image_array  # (batch_size, image_size, image_size), [0.0-strokes, 1.0-BG]
        return gt_image_array

    def crop_patch(self, image, center, image_size, crop_size):
        x0 = center[0] - crop_size // 2
        x1 = x0 + crop_size
        y0 = center[1] - crop_size // 2
        y1 = y0 + crop_size
        x0 = max(0, min(x0, image_size))
        y0 = max(0, min(y0, image_size))
        x1 = max(0, min(x1, image_size))
        y1 = max(0, min(y1, image_size))
        patch = image[y0:y1, x0:x1]
        return patch

    def gen_init_cursor_single(self, sketch_image, init_cursor_on_undrawn_pixel, misalign_size=3):
        # sketch_image: [0.0-stroke, 1.0-BG]
        image_size = sketch_image.shape[0]
        if np.sum(1.0 - sketch_image) == 0:
            center = np.zeros((2), dtype=np.int32)
            return center
        else:
            while True:
                center = np.random.randint(0, image_size, size=(2))  # (2), in large size
                patch = 1.0 - self.crop_patch(sketch_image, center, image_size, self.raster_size)
                if np.sum(patch) != 0:
                    if not init_cursor_on_undrawn_pixel:
                        return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)
                    else:
                        center_patch = 1.0 - self.crop_patch(sketch_image, center, image_size, misalign_size)
                        if np.sum(center_patch) != 0:
                            return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)

    def gen_init_cursors(self, sketch_data, random_pos=True, init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        init_cursor_batch_list = []
        for cursor_i in range(init_cursor_num):
            if random_pos:
                init_cursor_batch = []
                for i in range(len(sketch_data)):
                    sketch_image = sketch_data[i].copy().astype(np.float32)  # [0.0-stroke, 1.0-BG]
                    center = self.gen_init_cursor_single(sketch_image, init_cursor_on_undrawn_pixel)
                    init_cursor_batch.append(center)

                init_cursor_batch = np.stack(init_cursor_batch, axis=0)  # (N, 2)
            else:
                raise Exception('Not finished')
            init_cursor_batch_list.append(init_cursor_batch)

        if init_cursor_num == 1:
            init_cursor_batch = init_cursor_batch_list[0]
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=1).astype(np.float32)  # (N, 1, 2)
        else:
            init_cursor_batch = np.stack(init_cursor_batch_list, axis=1)  # (N, init_cursor_num, 2)
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=2).astype(
                np.float32)  # (N, init_cursor_num, 1, 2)

        return init_cursor_batch


def load_dataset_multi_object(dataset_base_dir, model_params):
    train_stroke3_data = []
    val_stroke3_data = []

    if model_params.data_set == 'clean_line_drawings':
        def load_qd_npz_data(npz_path):
            data = np.load(npz_path, encoding='latin1', allow_pickle=True)
            selected_strokes3 = data['stroke3']  # (N_sketches,), each with (N_points, 3)
            selected_strokes3 = selected_strokes3.tolist()
            return selected_strokes3

        base_dir_clean = 'QuickDraw-clean'
        cates = ['airplane', 'bus', 'car', 'sailboat', 'bird', 'cat', 'dog',
                 # 'rabbit',
                 'tree', 'flower',
                 # 'circle', 'line',
                 'zigzag'
                 ]

        for cate in cates:
            train_cate_sketch_data_npz_path = os.path.join(dataset_base_dir, base_dir_clean, 'train', cate + '.npz')
            val_cate_sketch_data_npz_path = os.path.join(dataset_base_dir, base_dir_clean, 'test', cate + '.npz')
            print(train_cate_sketch_data_npz_path)

            train_cate_stroke3_data = load_qd_npz_data(
                train_cate_sketch_data_npz_path)  # list of (N_sketches,), each with (N_points, 3)
            val_cate_stroke3_data = load_qd_npz_data(val_cate_sketch_data_npz_path)
            train_stroke3_data += train_cate_stroke3_data
            val_stroke3_data += val_cate_stroke3_data
    else:
        raise Exception('Unknown data type:', model_params.data_set)

    print('Loaded {}/{} from {}'.format(len(train_stroke3_data), len(val_stroke3_data), model_params.data_set))
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_sample_model_params = copy_hparams(model_params)
    eval_sample_model_params.use_input_dropout = 0
    eval_sample_model_params.use_recurrent_dropout = 0
    eval_sample_model_params.use_output_dropout = 0
    eval_sample_model_params.batch_size = 1  # only sample one at a time
    eval_sample_model_params.model_mode = 'eval_sample'

    train_set = GeneralMultiObjectDataLoader(train_stroke3_data,
                                             model_params.batch_size, model_params.raster_size,
                                             model_params.image_size_small, model_params.image_size_large,
                                             model_params.bin_gt, is_train=True)
    val_set = GeneralMultiObjectDataLoader(val_stroke3_data,
                                           eval_sample_model_params.batch_size, eval_sample_model_params.raster_size,
                                           eval_sample_model_params.image_size_small,
                                           eval_sample_model_params.image_size_large,
                                           eval_sample_model_params.bin_gt, is_train=False)

    result = [train_set, val_set, model_params, eval_sample_model_params]
    return result


class GeneralDataLoaderMultiObjectRough(object):
    def __init__(self,
                 photo_data,
                 sketch_data,
                 texture_data,
                 shadow_data,
                 batch_size,
                 raster_size,
                 image_size_small,
                 image_size_large,
                 is_train):
        self.batch_size = batch_size  # minibatch size
        self.raster_size = raster_size
        self.image_size_small = image_size_small
        self.image_size_large = image_size_large
        self.is_train = is_train

        assert photo_data is not None
        assert len(photo_data) == len(sketch_data)
        # self.num_batches = len(sketch_data) // self.batch_size
        self.batch_idx = -1
        print('batch_size', batch_size)

        assert type(photo_data) is list
        assert type(sketch_data) is list
        assert type(texture_data) is list and len(texture_data) > 0
        assert type(shadow_data) is list and len(shadow_data) > 0
        self.photo_data = photo_data
        self.sketch_data = sketch_data
        self.texture_data = texture_data  # list of (H, W, 3), [0, 255], uint8
        self.shadow_data = shadow_data  # list of (H, W), [0, 255], uint8

        self.memory_photo_data_batch = []
        self.memory_sketch_data_batch = []

    def rough_augmentation(self, raw_photo, texture_prob=0.20, noise_prob=0.15, shadow_prob=0.20):
        # raw_photo: (H, W), [0.0-stroke, 1.0-BG]
        aug_photo_rgb = np.stack([raw_photo for _ in range(3)], axis=-1)

        def texture_generation(texture_list, image_shape):
            while True:
                random_texture_id = random.randint(0, len(texture_list) - 1)
                texture_large = texture_list[random_texture_id]
                t_w, t_h = texture_large.shape[1], texture_large.shape[0]
                i_w, i_h = image_shape[1], image_shape[0]

                if t_h >= i_h and t_w >= i_w:
                    texture_large = np.copy(texture_large).astype(np.float32)
                    crop_y = random.randint(0, t_h - i_h)
                    crop_x = random.randint(0, t_w - i_w)
                    crop_texture = texture_large[crop_y: crop_y + i_h, crop_x: crop_x + i_w, :]
                    return crop_texture

        def texture_change(rough_img_, all_textures):
            # rough_img_: (H, W, 3), [0.0-stroke, 1.0-BG]

            texture_image = texture_generation(all_textures, rough_img_.shape)  # (h, w, 3)
            texture_image /= 255.0

            rand_b = np.random.uniform(1.0, 2.0, size=rough_img_.shape)
            textured_img = rough_img_ * (texture_image / rand_b + (rand_b - 1.0) / rand_b)  # [0.0, 1.0]
            return textured_img

        def noise_change(rough_img_, noise_scale=25):
            # rough_img_: (H, W, 3), [0.0, 1.0]
            rough_img_255 = rough_img_ * 255.0

            rand_noise = np.random.uniform(-1.0, 1.0, size=rough_img_255.shape) * noise_scale
            # rand_noise = np.random.normal(size=rough_img.shape) * noise_scale
            noise_img = rough_img_255 + rand_noise
            noise_img = np.clip(noise_img, 0.0, 255.0)
            noise_img /= 255.0
            return noise_img

        def shadow_change(rough_img_, all_shadows):
            # rough_img_: (H, W, 3), [0.0, 1.0]
            rough_img_255 = rough_img_ * 255.0

            shadow_i = random.randint(0, len(all_shadows) - 1)
            shadow_full = all_shadows[shadow_i]  # (H, W), [0, 255]
            shadow_img_size = shadow_full.shape[0]

            while True:
                position = np.random.randint(-shadow_img_size // 2, shadow_img_size // 2, (2))
                if abs(position[0]) > (shadow_img_size // 8) and abs(position[1]) > (shadow_img_size // 8):
                    break
            position += (shadow_img_size // 2)

            crop_up = shadow_img_size - position[0]
            crop_left = shadow_img_size - position[1]

            shadow_image_large = shadow_full[crop_up: crop_up + shadow_img_size, crop_left: crop_left + shadow_img_size]
            shadow_bg = Image.fromarray(shadow_image_large, 'L')
            shadow_bg = shadow_bg.resize(size=(rough_img_255.shape[1], rough_img_255.shape[0]), resample=Image.BILINEAR)
            shadow_bg = np.array(shadow_bg, dtype=np.float32) / 255.0  # [0.0-shadow, 1.0-BG]
            shadow_bg = np.stack([shadow_bg for _ in range(3)], axis=-1)

            shadow_img = rough_img_255 * shadow_bg
            shadow_img /= 255.0
            return shadow_img

        if random.random() <= texture_prob:
            aug_photo_rgb = texture_change(aug_photo_rgb, self.texture_data)  # (H, W, 3), [0.0, 1.0]
        if random.random() <= noise_prob:
            aug_photo_rgb = noise_change(aug_photo_rgb)  # (H, W, 3), [0.0, 1.0]
        if random.random() <= shadow_prob:
            aug_photo_rgb = shadow_change(aug_photo_rgb, self.shadow_data)  # (H, W, 3), [0.0, 1.0]

        return aug_photo_rgb

    def image_interpolation(self, photo, sketch, photo_prob):
        interp_photo = photo * photo_prob + sketch * (1.0 - photo_prob)
        interp_photo = np.clip(interp_photo, 0.0, 1.0)
        return interp_photo

    def get_batch_from_memory(self, memory_idx, interpolate_type, fixed_image_size=-1, random_cursor=True,
                              photo_prob=1.0, init_cursor_num=1):
        if len(self.memory_sketch_data_batch) >= memory_idx + 1:
            photo_data_batch = self.memory_photo_data_batch[memory_idx]
            sketch_data_batch = self.memory_sketch_data_batch[memory_idx]
            image_size_rand = sketch_data_batch.shape[1]
        else:
            if fixed_image_size == -1:
                image_size_rand = random.randint(self.image_size_small, self.image_size_large)
            else:
                image_size_rand = fixed_image_size

            # photo_prob = 0.0 if photo_prob_type == 'zero' else 1.0
            photo_data_batch, sketch_data_batch = self.select_sketch(
                image_size_rand)  # both: (H, W), [0.0-stroke, 1.0-BG]
            photo_data_batch = self.rough_augmentation(photo_data_batch)  # (H, W, 3), [0.0-stroke, 1.0-BG]

            self.memory_photo_data_batch.append(photo_data_batch)
            self.memory_sketch_data_batch.append(sketch_data_batch)

        if interpolate_type == 'prob':
            if random.random() >= photo_prob:
                photo_data_batch = np.stack([sketch_data_batch for _ in range(3)],
                                            axis=-1)  # (H, W, 3), [0.0-stroke, 1.0-BG]
        elif interpolate_type == 'image':
            photo_data_batch = self.image_interpolation(
                photo_data_batch, np.stack([sketch_data_batch for _ in range(3)], axis=-1), photo_prob)
        else:
            raise Exception('Unknown interpolate_type', interpolate_type)

        photo_data_batch = np.expand_dims(photo_data_batch, axis=0)  # (1, image_size, image_size, 3)
        sketch_data_batch = np.expand_dims(sketch_data_batch,
                                           axis=0)  # (1, image_size, image_size), [0.0-strokes, 1.0-BG]

        return photo_data_batch, sketch_data_batch, \
               self.gen_init_cursors(sketch_data_batch, random_cursor, init_cursor_num), image_size_rand

    def select_sketch(self, image_size_rand):
        resolution_idx = image_size_rand - self.image_size_small
        img_idx = random.randint(0, len(self.sketch_data[resolution_idx]) - 1)
        assert img_idx != -1

        selected_sketch = self.sketch_data[resolution_idx][img_idx]  # [0-stroke, 255-BG], uint8
        selected_photo = self.photo_data[resolution_idx][img_idx]  # [0-stroke, 255-BG], uint8

        rst_sketch_image = selected_sketch.astype(np.float32) / 255.0  # [0.0-stroke, 1.0-BG]
        rst_photo_image = selected_photo.astype(np.float32) / 255.0  # [0.0-stroke, 1.0-BG]

        return rst_photo_image, rst_sketch_image

    def get_batch_multi_res(self, loop_num, interpolate_type, random_cursor=True, init_cursor_num=1, photo_prob=1.0):
        photo_data_batch = []
        sketch_data_batch = []
        init_cursors_batch = []
        image_size_batch = []
        batch_size_per_loop = self.batch_size // loop_num
        for loop_i in range(loop_num):
            image_size_rand = random.randint(self.image_size_small, self.image_size_large)

            photo_data_sub_batch = []
            sketch_data_sub_batch = []
            for img_i in range(batch_size_per_loop):
                photo_patch, sketch_patch = self.select_sketch(image_size_rand)  # both: (H, W), [0.0-stroke, 1.0-BG]
                photo_patch = self.rough_augmentation(photo_patch)  # (H, W, 3), [0.0-stroke, 1.0-BG]

                if interpolate_type == 'prob':
                    if random.random() >= photo_prob:
                        photo_patch = np.stack([sketch_patch for _ in range(3)],
                                               axis=-1)  # (H, W, 3), [0.0-stroke, 1.0-BG]
                elif interpolate_type == 'image':
                    photo_patch = self.image_interpolation(
                        photo_patch, np.stack([sketch_patch for _ in range(3)], axis=-1), photo_prob)
                else:
                    raise Exception('Unknown interpolate_type', interpolate_type)

                photo_data_sub_batch.append(photo_patch)
                sketch_data_sub_batch.append(sketch_patch)

            photo_data_sub_batch = np.stack(photo_data_sub_batch,
                                            axis=0)  # (N, image_size, image_size, 3), [0.0-strokes, 1.0-BG]
            sketch_data_sub_batch = np.stack(sketch_data_sub_batch,
                                             axis=0)  # (N, image_size, image_size), [0.0-strokes, 1.0-BG]
            init_cursors_sub_batch = self.gen_init_cursors(sketch_data_sub_batch, random_cursor, init_cursor_num)
            photo_data_batch.append(photo_data_sub_batch)
            sketch_data_batch.append(sketch_data_sub_batch)
            init_cursors_batch.append(init_cursors_sub_batch)
            image_size_batch.append(image_size_rand)

        return photo_data_batch, sketch_data_batch, init_cursors_batch, image_size_batch

    def crop_patch(self, image, center, image_size, crop_size):
        x0 = center[0] - crop_size // 2
        x1 = x0 + crop_size
        y0 = center[1] - crop_size // 2
        y1 = y0 + crop_size
        x0 = max(0, min(x0, image_size))
        y0 = max(0, min(y0, image_size))
        x1 = max(0, min(x1, image_size))
        y1 = max(0, min(y1, image_size))
        patch = image[y0:y1, x0:x1]
        return patch

    def gen_init_cursor_single(self, sketch_image):
        # sketch_image: [0.0-stroke, 1.0-BG]
        image_size = sketch_image.shape[0]
        if np.sum(1.0 - sketch_image) == 0:
            center = np.zeros((2), dtype=np.int32)
            return center
        else:
            while True:
                center = np.random.randint(0, image_size, size=(2))  # (2), in large size
                patch = 1.0 - self.crop_patch(sketch_image, center, image_size, self.raster_size)
                if np.sum(patch) != 0:
                    return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)

    def gen_init_cursors(self, sketch_data, random_pos=True, init_cursor_num=1):
        init_cursor_batch_list = []
        for cursor_i in range(init_cursor_num):
            if random_pos:
                init_cursor_batch = []
                for i in range(len(sketch_data)):
                    sketch_image = sketch_data[i].copy().astype(np.float32)  # [0.0-stroke, 1.0-BG]
                    center = self.gen_init_cursor_single(sketch_image)
                    init_cursor_batch.append(center)

                init_cursor_batch = np.stack(init_cursor_batch, axis=0)  # (N, 2)
            else:
                raise Exception('Not finished')
            init_cursor_batch_list.append(init_cursor_batch)

        if init_cursor_num == 1:
            init_cursor_batch = init_cursor_batch_list[0]
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=1).astype(np.float32)  # (N, 1, 2)
        else:
            init_cursor_batch = np.stack(init_cursor_batch_list, axis=1)  # (N, init_cursor_num, 2)
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=2).astype(
                np.float32)  # (N, init_cursor_num, 1, 2)

        return init_cursor_batch


def load_dataset_multi_object_rough(dataset_base_dir, model_params):
    train_photo_data = []
    train_sketch_data = []
    val_photo_data = []
    val_sketch_data = []
    texture_data = []
    shadow_data = []

    if model_params.data_set == 'rough_sketches':
        base_dir_rough = 'QuickDraw-rough'

        def load_sketch_data(mat_path):
            sketch_data_mat = scipy.io.loadmat(mat_path)
            sketch_data = sketch_data_mat['sketch_array']
            sketch_data = np.array(sketch_data, dtype=np.uint8)  # (N, resolution, resolution), [0-strokes, 255-BG]
            return sketch_data

        def load_photo_data(mat_path):
            photo_data_mat = scipy.io.loadmat(mat_path)
            photo_data = photo_data_mat['image_array']
            photo_data = np.array(photo_data, dtype=np.uint8)  # (N, resolution, resolution), [0-strokes, 255-BG]
            return photo_data

        def load_normal_data(img_path):
            assert '.png' in img_path or '.jpg'
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)  # (H, W, 3), [0-stroke, 255-BG], uint8
            return img

        ## Texture
        texture_base = os.path.join(dataset_base_dir, base_dir_rough, 'texture')
        all_texture = os.listdir(texture_base)
        all_texture.sort()

        for file_name in all_texture:
            texture_path = os.path.join(texture_base, file_name)
            texture_uint8 = load_normal_data(texture_path)
            texture_data.append(texture_uint8)

        ## shadow
        def process_angle(img, temp_size):
            padded_img = img.copy()
            padded_img[0, 0:temp_size] -= 1
            padded_img[0, -(temp_size + 1):-1] -= 1
            padded_img[-1, 0:temp_size] -= 1
            padded_img[-1, -(temp_size + 1):-1] -= 1

            padded_img[0:temp_size, 0] -= 1
            padded_img[0:temp_size, -1] -= 1
            padded_img[-(temp_size + 1):-1, 0] -= 1
            padded_img[-(temp_size + 1):-1, -1] -= 1
            return padded_img

        def pad_img(ori_img, pad_value):
            padded_img = np.pad(ori_img, 1, constant_values=pad_value)
            img_h, img_w = padded_img.shape[0], padded_img.shape[1]

            temp_size = img_h // 3
            padded_img = process_angle(padded_img, temp_size)

            temp_size = img_h // 9
            padded_img = process_angle(padded_img, temp_size)

            temp_size = img_h // 15
            padded_img = process_angle(padded_img, temp_size)

            temp_size = img_h // 21
            padded_img = process_angle(padded_img, temp_size)

            padded_img = np.clip(padded_img, 0, 255)

            return padded_img

        def shadow_generation(transparency, shadow_img_size=1024):
            deepest_value = int(255 * transparency)

            center_patch = np.zeros((shadow_img_size // 2, shadow_img_size // 2), dtype=np.uint8)
            center_patch.fill(255)

            pad_gap = shadow_img_size // 2
            shadow_patch = center_patch.copy()
            for i in range(pad_gap):
                curr_pad_value = 255.0 - float(255.0 - deepest_value) / float(pad_gap) * (i + 1)
                shadow_patch = pad_img(shadow_patch, pad_value=curr_pad_value)

            for i in range(shadow_img_size // 4):
                shadow_patch = pad_img(shadow_patch, pad_value=deepest_value)

            assert shadow_patch.shape[0] == shadow_img_size * 2, shadow_patch.shape[0]
            return shadow_patch

        for transparency_ in range(90, 95 + 1):
            transparency = transparency_ / 100.0
            shadow_full = shadow_generation(transparency)
            shadow_data.append(shadow_full)

        splits = ['train', 'test']

        resolutions = [model_params.image_size_small, model_params.image_size_large]

        for resolution in range(resolutions[0], resolutions[1] + 1):
            for split in splits:
                sketch_mat1_path = os.path.join(dataset_base_dir, base_dir_rough, 'model_pencil1',
                                                'sketch', split, 'res_' + str(resolution) + '.mat')
                photo_mat1_path = os.path.join(dataset_base_dir, base_dir_rough, 'model_pencil1',
                                               'photo', split, 'res_' + str(resolution) + '.mat')
                sketch_data1_uint8 = load_sketch_data(
                    sketch_mat1_path)  # (N, resolution, resolution), [0-strokes, 255-BG]
                photo_data1_uint8 = load_photo_data(photo_mat1_path)  # (N, resolution, resolution), [0-strokes, 255-BG]

                sketch_mat2_path = os.path.join(dataset_base_dir, base_dir_rough, 'model_pencil2',
                                                'sketch', split, 'res_' + str(resolution) + '.mat')
                photo_mat2_path = os.path.join(dataset_base_dir, base_dir_rough, 'model_pencil2',
                                               'photo', split, 'res_' + str(resolution) + '.mat')
                sketch_data2_uint8 = load_sketch_data(
                    sketch_mat2_path)  # (N, resolution, resolution), [0-strokes, 255-BG]
                photo_data2_uint8 = load_photo_data(photo_mat2_path)  # (N, resolution, resolution), [0-strokes, 255-BG]

                sketch_data_uint8 = np.concatenate([sketch_data1_uint8, sketch_data2_uint8],
                                                   axis=0)  # (N, resolution, resolution), [0-strokes, 255-BG]
                photo_data_uint8 = np.concatenate([photo_data1_uint8, photo_data2_uint8],
                                                  axis=0)  # (N, resolution, resolution), [0-strokes, 255-BG]

                if split == 'train':
                    train_photo_data.append(photo_data_uint8)
                    train_sketch_data.append(sketch_data_uint8)
                else:
                    val_photo_data.append(photo_data_uint8)
                    val_sketch_data.append(sketch_data_uint8)

        assert len(train_sketch_data) == len(train_photo_data)
        assert len(val_sketch_data) == len(val_photo_data)
    else:
        raise Exception('Unknown data type:', model_params.data_set)

    print('Loaded {}/{} from {}'.format(len(train_sketch_data), len(val_sketch_data), model_params.data_set))
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_sample_model_params = copy_hparams(model_params)
    eval_sample_model_params.use_input_dropout = 0
    eval_sample_model_params.use_recurrent_dropout = 0
    eval_sample_model_params.use_output_dropout = 0
    eval_sample_model_params.batch_size = 1  # only sample one at a time
    eval_sample_model_params.model_mode = 'eval_sample'

    train_set = GeneralDataLoaderMultiObjectRough(train_photo_data, train_sketch_data,
                                                  texture_data, shadow_data,
                                                  model_params.batch_size, model_params.raster_size,
                                                  model_params.image_size_small, model_params.image_size_large,
                                                  is_train=True)
    val_set = GeneralDataLoaderMultiObjectRough(val_photo_data, val_sketch_data,
                                                texture_data, shadow_data,
                                                eval_sample_model_params.batch_size,
                                                eval_sample_model_params.raster_size,
                                                eval_sample_model_params.image_size_small,
                                                eval_sample_model_params.image_size_large,
                                                is_train=False)

    result = [
        train_set, val_set, model_params, eval_sample_model_params
    ]
    return result


class GeneralDataLoaderNormalImageLinear(object):
    def __init__(self,
                 photo_data,
                 sketch_data,
                 sketch_shape,
                 batch_size,
                 raster_size,
                 image_size_small,
                 image_size_large,
                 random_image_size,
                 flip_prob,
                 rotate_prob,
                 is_train):
        self.batch_size = batch_size  # minibatch size
        self.raster_size = raster_size
        self.image_size_small = image_size_small
        self.image_size_large = image_size_large
        self.random_image_size = random_image_size
        self.is_train = is_train

        assert photo_data is not None
        assert len(photo_data) == len(sketch_data)
        self.num_batches = len(sketch_data) // self.batch_size
        self.batch_idx = -1
        print('batch_size', batch_size, ', num_batches', self.num_batches)

        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

        assert type(photo_data) is list
        assert type(sketch_data) is list
        self.photo_data = photo_data
        self.sketch_data = sketch_data
        self.sketch_shape = sketch_shape

    def get_batch_from_memory(self, memory_idx, interpolate_type, fixed_image_size=-1, random_cursor=True,
                              photo_prob=1.0,
                              init_cursor_num=1):
        if self.random_image_size:
            image_size_rand = fixed_image_size
        else:
            image_size_rand = self.image_size_large

        photo_data_batch, sketch_data_batch = self.select_sketch_and_crop(
            image_size_rand, data_idx=memory_idx, photo_prob=photo_prob,
            interpolate_type=interpolate_type)  # sketch_patch: [0.0-stroke, 1.0-BG]

        photo_data_batch = np.expand_dims(photo_data_batch, axis=0)  # (1, image_size, image_size, 3)
        sketch_data_batch = np.expand_dims(sketch_data_batch,
                                           axis=0)  # (1, image_size, image_size), [0.0-strokes, 1.0-BG]
        image_size_rand = sketch_data_batch.shape[1]

        return photo_data_batch, sketch_data_batch, \
               self.gen_init_cursors(sketch_data_batch, random_cursor, init_cursor_num), image_size_rand

    def crop_and_augment(self, photo, sketch, shape, crop_size, rotate_angle, stroke_cover=0.01):
        # img: [0-stroke, 255-BG], uint8

        def angle_convert(angle):
            return angle / 180.0 * math.pi

        img_h, img_w = shape[0], shape[1]

        if self.is_train:
            crop_up = random.randint(0, img_h - crop_size)
            crop_left = random.randint(0, img_w - crop_size)
        else:
            crop_up = (img_h - crop_size) // 2
            crop_left = (img_w - crop_size) // 2

        assert crop_up >= 0
        assert crop_left >= 0

        crop_box = (crop_left, crop_up, crop_left + crop_size, crop_up + crop_size)
        rst_sketch_image = sketch.crop(crop_box)
        rst_photo_image = photo.crop(crop_box)

        if random.random() <= self.flip_prob and self.is_train:
            rst_sketch_image = rst_sketch_image.transpose(Image.FLIP_LEFT_RIGHT)
            rst_photo_image = rst_photo_image.transpose(Image.FLIP_LEFT_RIGHT)

        if rotate_angle != 0 and self.is_train:
            rst_sketch_image = rst_sketch_image.rotate(rotate_angle, resample=Image.BILINEAR)
            rst_photo_image = rst_photo_image.rotate(rotate_angle, resample=Image.BILINEAR)
            rst_sketch_image = np.array(rst_sketch_image, dtype=np.uint8)
            rst_photo_image = np.array(rst_photo_image, dtype=np.uint8)

            center = rst_photo_image.shape[0] // 2

            new_dim = float(crop_size) / (
                        math.sin(angle_convert(abs(rotate_angle))) + math.cos(angle_convert(abs(rotate_angle))))
            new_dim = int(round(new_dim))

            start_pos = center - new_dim // 2
            end_pos = start_pos + new_dim
            rst_sketch_image = rst_sketch_image[start_pos:end_pos, start_pos:end_pos, :]
            rst_photo_image = rst_photo_image[start_pos:end_pos, start_pos:end_pos, :]

        rst_sketch_image = np.array(rst_sketch_image, dtype=np.float32) / 255.0  # [0.0-stroke, 1.0-BG]
        rst_sketch_image = rst_sketch_image[:, :, 0]
        rst_photo_image = np.array(rst_photo_image, dtype=np.float32) / 255.0  # [0.0-stroke, 1.0-BG]

        percentage = np.mean(1.0 - rst_sketch_image)
        valid = True
        if percentage < stroke_cover:
            valid = False

        return rst_photo_image, rst_sketch_image, valid

    def image_interpolation(self, photo, sketch, photo_prob):
        interp_photo = photo * photo_prob + sketch * (1.0 - photo_prob)
        interp_photo = np.clip(interp_photo, 0.0, 1.0)
        return interp_photo

    def select_sketch_and_crop(self, image_size_rand, interpolate_type, rotate_angle=0, photo_prob=1.0,
                               data_idx=-1, trial_times=10):
        if self.is_train:
            while True:
                rand_img_idx = random.randint(0, len(self.sketch_data) - 1)
                selected_sketch_shape = self.sketch_shape[rand_img_idx]
                if selected_sketch_shape[0] >= image_size_rand and selected_sketch_shape[1] >= image_size_rand:
                    img_idx = rand_img_idx
                    break
        else:
            assert data_idx != -1
            img_idx = data_idx

        assert img_idx != -1
        selected_sketch = self.sketch_data[img_idx]
        selected_photo = self.photo_data[img_idx]
        selected_shape = self.sketch_shape[img_idx]

        assert interpolate_type in ['prob', 'image']

        if interpolate_type == 'prob' and random.random() >= photo_prob:
            selected_photo = self.sketch_data[img_idx]

        for trial_i in range(trial_times):
            cropped_photo, cropped_sketch, valid = \
                self.crop_and_augment(selected_photo, selected_sketch, selected_shape, image_size_rand, rotate_angle)
            # cropped_photo, cropped_sketch: [0.0-stroke, 1.0-BG]

            if valid or trial_i == trial_times - 1:
                if interpolate_type == 'image':
                    cropped_photo = self.image_interpolation(cropped_photo,
                                                             np.stack([cropped_sketch for _ in range(3)], axis=-1),
                                                             photo_prob)

                return cropped_photo, cropped_sketch

    def get_batch_multi_res(self, loop_num, interpolate_type, random_cursor=True, init_cursor_num=1, photo_prob=1.0):
        photo_data_batch = []
        sketch_data_batch = []
        init_cursors_batch = []
        image_size_batch = []
        batch_size_per_loop = self.batch_size // loop_num
        for loop_i in range(loop_num):
            if self.random_image_size:
                image_size_rand = random.randint(self.image_size_small, self.image_size_large)
            else:
                image_size_rand = self.image_size_large

            rotate_angle = 0
            if random.random() <= self.rotate_prob:
                rotate_angle = random.randint(-45, 45)

            photo_data_sub_batch = []
            sketch_data_sub_batch = []
            for img_i in range(batch_size_per_loop):
                photo_patch, sketch_patch = \
                    self.select_sketch_and_crop(image_size_rand, rotate_angle=rotate_angle, photo_prob=photo_prob,
                                                interpolate_type=interpolate_type)  # sketch_patch: [0.0-stroke, 1.0-BG]
                photo_data_sub_batch.append(photo_patch)
                sketch_data_sub_batch.append(sketch_patch)

            photo_data_sub_batch = np.stack(photo_data_sub_batch,
                                            axis=0)  # (N, image_size, image_size, 3), [0.0-strokes, 1.0-BG]
            sketch_data_sub_batch = np.stack(sketch_data_sub_batch,
                                             axis=0)  # (N, image_size, image_size), [0.0-strokes, 1.0-BG]
            init_cursors_sub_batch = self.gen_init_cursors(sketch_data_sub_batch, random_cursor, init_cursor_num)

            photo_data_batch.append(photo_data_sub_batch)
            sketch_data_batch.append(sketch_data_sub_batch)
            init_cursors_batch.append(init_cursors_sub_batch)

            image_size_rand = photo_data_sub_batch.shape[1]
            image_size_batch.append(image_size_rand)

        return photo_data_batch, sketch_data_batch, init_cursors_batch, image_size_batch

    def crop_patch(self, image, center, image_size, crop_size):
        x0 = center[0] - crop_size // 2
        x1 = x0 + crop_size
        y0 = center[1] - crop_size // 2
        y1 = y0 + crop_size
        x0 = max(0, min(x0, image_size))
        y0 = max(0, min(y0, image_size))
        x1 = max(0, min(x1, image_size))
        y1 = max(0, min(y1, image_size))
        patch = image[y0:y1, x0:x1]
        return patch

    def gen_init_cursor_single(self, sketch_image):
        # sketch_image: [0.0-stroke, 1.0-BG]
        image_size = sketch_image.shape[0]
        if np.sum(1.0 - sketch_image) == 0:
            center = np.zeros((2), dtype=np.int32)
            return center
        else:
            while True:
                center = np.random.randint(0, image_size, size=(2))  # (2), in large size
                patch = 1.0 - self.crop_patch(sketch_image, center, image_size, self.raster_size)
                if np.sum(patch) != 0:
                    return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)

    def gen_init_cursors(self, sketch_data, random_pos=True, init_cursor_num=1):
        init_cursor_batch_list = []
        for cursor_i in range(init_cursor_num):
            if random_pos:
                init_cursor_batch = []
                for i in range(len(sketch_data)):
                    sketch_image = sketch_data[i].copy().astype(np.float32)  # [0.0-stroke, 1.0-BG]
                    center = self.gen_init_cursor_single(sketch_image)
                    init_cursor_batch.append(center)

                init_cursor_batch = np.stack(init_cursor_batch, axis=0)  # (N, 2)
            else:
                raise Exception('Not finished')
            init_cursor_batch_list.append(init_cursor_batch)

        if init_cursor_num == 1:
            init_cursor_batch = init_cursor_batch_list[0]
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=1).astype(np.float32)  # (N, 1, 2)
        else:
            init_cursor_batch = np.stack(init_cursor_batch_list, axis=1)  # (N, init_cursor_num, 2)
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=2).astype(
                np.float32)  # (N, init_cursor_num, 1, 2)

        return init_cursor_batch


def load_dataset_normal_images(dataset_base_dir, model_params):
    train_photo_data = []
    train_sketch_data = []
    train_data_shape = []
    val_photo_data = []
    val_sketch_data = []
    val_data_shape = []

    if model_params.data_set == 'faces':
        random_training_image_size = False
        flip_prob = -0.1
        rotate_prob = -0.1

        splits = ['train', 'val']

        database = os.path.join(dataset_base_dir, 'CelebAMask-faces')
        photo_base = os.path.join(database, 'CelebA-HQ-img256')
        edge_base = os.path.join(database, 'CelebAMask-HQ-edge256')

        train_split_txt_save_path = os.path.join(database, 'train.txt')
        val_split_txt_save_path = os.path.join(database, 'val.txt')
        celeba_train_txt = np.loadtxt(train_split_txt_save_path, dtype=str)
        celeba_val_txt = np.loadtxt(val_split_txt_save_path, dtype=str)
        splits_indices_map = {'train': celeba_train_txt, 'val': celeba_val_txt}

        for split in splits:
            split_indices = splits_indices_map[split]

            for i in range(len(split_indices)):
                file_idx = split_indices[i]
                img_file_path = os.path.join(photo_base, str(file_idx) + '.jpg')
                edge_img_path = os.path.join(edge_base, str(file_idx) + '.png')

                img_data = Image.open(img_file_path).convert('RGB')
                edge_data = Image.open(edge_img_path).convert('RGB')

                if split == 'train':
                    train_photo_data.append(img_data)
                    train_sketch_data.append(edge_data)
                    train_data_shape.append((img_data.height, img_data.width))
                else:  # split == 'val'
                    val_photo_data.append(img_data)
                    val_sketch_data.append(edge_data)
                    val_data_shape.append((img_data.height, img_data.width))

        assert len(train_sketch_data) == len(train_data_shape) == len(train_photo_data)
        assert len(val_sketch_data) == len(val_data_shape) == len(val_photo_data)
    else:
        raise Exception('Unknown data type:', model_params.data_set)

    print('Loaded {}/{} from {}'.format(len(train_sketch_data), len(val_sketch_data), model_params.data_set))
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_sample_model_params = copy_hparams(model_params)
    eval_sample_model_params.use_input_dropout = 0
    eval_sample_model_params.use_recurrent_dropout = 0
    eval_sample_model_params.use_output_dropout = 0
    eval_sample_model_params.batch_size = 1  # only sample one at a time
    eval_sample_model_params.model_mode = 'eval_sample'

    train_set = GeneralDataLoaderNormalImageLinear(train_photo_data, train_sketch_data, train_data_shape,
                                                   model_params.batch_size, model_params.raster_size,
                                                   image_size_small=model_params.image_size_small,
                                                   image_size_large=model_params.image_size_large,
                                                   random_image_size=random_training_image_size,
                                                   flip_prob=flip_prob, rotate_prob=rotate_prob,
                                                   is_train=True)
    val_set = GeneralDataLoaderNormalImageLinear(val_photo_data, val_sketch_data, val_data_shape,
                                                 eval_sample_model_params.batch_size,
                                                 eval_sample_model_params.raster_size,
                                                 image_size_small=eval_sample_model_params.image_size_small,
                                                 image_size_large=eval_sample_model_params.image_size_large,
                                                 random_image_size=random_training_image_size,
                                                 flip_prob=flip_prob, rotate_prob=rotate_prob,
                                                 is_train=False)

    result = [
        train_set, val_set, model_params, eval_sample_model_params
    ]
    return result


def load_dataset_training(dataset_base_dir, model_params):
    if model_params.data_set == 'clean_line_drawings':
        return load_dataset_multi_object(dataset_base_dir, model_params)
    elif model_params.data_set == 'rough_sketches':
        return load_dataset_multi_object_rough(dataset_base_dir, model_params)
    elif model_params.data_set == 'faces':
        return load_dataset_normal_images(dataset_base_dir, model_params)
    else:
        raise Exception('Unknown data_set', model_params.data_set)
