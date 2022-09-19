import cv2
import os
import numpy as np
import paddle


def check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    if os.path.isfile(img_file) and check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


class FDDataset(paddle.io.Dataset):
    def __init__(self,
                 data_dir=None,
                 img_size=[640, 640],
                 input_name='x2paddle_images'):

        self.input_name = input_name
        self.img_size = img_size
        self.data_list = get_image_file_list(data_dir)
        # self.func = func

    def __getitem__(self, index):

        # 根据索引，从data_list列表中取出一个图像
        image_path = self.data_list[index]
        # 读取图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 预处理
        img, scale_factor = self.image_preprocess(img, self.img_size)

        return {self.input_name: img}

    def __len__(self):
        return len(self.data_list)

    def _generate_scale(self, im, target_shape, keep_ratio=True):
        """
            Args:
                im (np.ndarray): image (np.ndarray)
            Returns:
                im_scale_x: the resize ratio of X
                im_scale_y: the resize ratio of Y
            """
        origin_shape = im.shape[:2]
        if keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(target_shape)
            target_size_max = np.max(target_shape)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = target_shape
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x

    def image_preprocess(self, img, target_shape):
        # Resize image
        im_scale_y, im_scale_x = self._generate_scale(img, target_shape)
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_LINEAR)
        # Pad
        im_h, im_w = img.shape[:2]
        h, w = target_shape[:]
        if h != im_h or w != im_w:
            canvas = np.ones((h, w, 3), dtype=np.float32)
            canvas *= np.array([114.0, 114.0, 114.0], dtype=np.float32)
            canvas[0:im_h, 0:im_w, :] = img.astype(np.float32)
            img = canvas
        img = np.transpose(img / 255, [2, 0, 1])
        scale_factor = np.array([im_scale_y, im_scale_x])
        return img.astype(np.float32), scale_factor
