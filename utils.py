# PyTorch

import os

import natsort
import numpy as np
import torch
from PIL import Image
from numpy.lib.stride_tricks import as_strided
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

semi_supervised_labels_path = '../dataset/semisupervised_labels/Labels'
planet_boxes_path = '../dataset/planet'
planet_test_boxes_path = '/media/HDD1/dataset/ai4eo/planet_test'
sentinel_boxes_path = '/media/HDD1/dataset/ai4eo/sentinel-2'
sentinel_test_boxes_path = '/media/HDD1/dataset/ai4eo/planet-2_test'


def cut_image_strided(image, new_size):
    """

    :param image:
    :param new_size:
    :return:
    """
    bands = image.shape[0]
    new_size_y, new_size_x = new_size
    old_size_y = image.shape[1]
    old_size_x = image.shape[2]
    nr_images_x = old_size_x // new_size[1]
    nr_images_y = old_size_y // new_size[0]
    if old_size_x % new_size_x != 0 or old_size_y % new_size_y != 0:
        print("The patch size is not a full multiple of the complete patch size")

    return as_strided(image,
                      shape=(nr_images_y, nr_images_x, bands, new_size_y, new_size_x),
                      strides=(image.strides[1] * new_size_y,
                               image.strides[2] * new_size_x,
                               image.strides[0],
                               image.strides[1],
                               image.strides[2]))


def get_matching_planet_path_image(one_location_cm_label_folder_name):
    parallel_number = one_location_cm_label_folder_name.split('_')[0]
    planet_east_west_folder_name = one_location_cm_label_folder_name[4:]
    planet_images_path = os.path.join(planet_boxes_path, parallel_number, planet_east_west_folder_name, 'L3H-SR')
    return planet_images_path


def read_planet_images_from_cm(cm_label_path: str):
    parallel_number = cm_label_path.split('_')[0]
    planet_east_west_folder_name = cm_label_path[4:-20]
    original_image_name = cm_label_path[-11:-4] + '-01.tif'
    changed_image_name = cm_label_path[-19:-12] + '-01.tif'
    original_image_path = os.path.join(planet_boxes_path, parallel_number, planet_east_west_folder_name,
                                       'L3H-SR', original_image_name)
    changed_image_path = os.path.join(planet_boxes_path, parallel_number, planet_east_west_folder_name,
                                      'L3H-SR', changed_image_name)
    I1, I2 = io.imread(original_image_path), io.imread(changed_image_path)
    # I1 = (I1 - I1.mean()) / I1.std()
    # I2 = (I2 - I2.mean()) / I2.std()
    new_min = -1
    new_max = 1
    I1 = (I1 - np.min(I1)) / (np.max(I1) - np.min(I1)) * (new_max - new_min) + new_min
    I2 = (I2 - np.min(I2)) / (np.max(I2) - np.min(I2)) * (new_max - new_min) + new_min
    return I1, I2


def mask2rgb(mask):
    color_label_dict = {0: (0, 0, 0),
                        1: (128, 0, 0),
                        2: (0, 128, 0),
                        3: (128, 128, 0),
                        4: (0, 0, 128),
                        5: (128, 0, 128),
                        6: (0, 128, 128),
                        7: (128, 128, 128)}

    maskRGB = np.empty((mask.shape[1], mask.shape[2], 3))
    mask = np.squeeze(mask)
    for key in color_label_dict.keys():
        pixel_value = color_label_dict[key]
        maskRGB[mask == key] = pixel_value

    return maskRGB.astype(np.uint8)


class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, transform=None, train=True):
        # basics
        self.color_label_dict = {0: (0, 0, 0),
                                 1: (128, 0, 0),
                                 2: (0, 128, 0),
                                 3: (128, 128, 0),
                                 4: (0, 0, 128),
                                 5: (128, 0, 128),
                                 6: (0, 128, 128),
                                 7: (128, 128, 128)}

        self.transform = transform
        self.path = path
        self.train = train

        # load images
        self.imgs_1 = []
        self.imgs_2 = []
        self.labels = []
        self.n_patches = 0
        n_pix = 0
        n_pix0 = 0
        n_pix1 = 0
        n_pix2 = 0
        n_pix3 = 0
        n_pix4 = 0
        n_pix5 = 0
        n_pix6 = 0
        n_pix7 = 0

        if train:
            for one_location_cm_label_folder_name in tqdm(os.listdir(path)):
                # load and store each image
                one_location_cm_label_path = os.path.join(path, one_location_cm_label_folder_name)
                for cm_label_file_name in os.listdir(one_location_cm_label_path):
                    cm_label_path = os.path.join(one_location_cm_label_path, cm_label_file_name)

                    I1, I2 = read_planet_images_from_cm(cm_label_file_name)

                    cm_rgb_np = np.asarray(Image.open(cm_label_path).convert('RGB'))
                    cm_encoded = self.rgb_to_onehot(cm_rgb_np)
                    cm_indices_np = np.argmax(cm_encoded, axis=2)
                    n_pix += np.prod(cm_indices_np.shape)
                    n_pix0 += np.count_nonzero(cm_indices_np == 0)
                    n_pix1 += np.count_nonzero(cm_indices_np == 1)
                    n_pix2 += np.count_nonzero(cm_indices_np == 2)
                    n_pix3 += np.count_nonzero(cm_indices_np == 3)
                    n_pix4 += np.count_nonzero(cm_indices_np == 4)
                    n_pix5 += np.count_nonzero(cm_indices_np == 5)
                    n_pix6 += np.count_nonzero(cm_indices_np == 6)
                    n_pix7 += np.count_nonzero(cm_indices_np == 7)
                    cm_reshape = cm_indices_np.reshape(16, 256, 256)
                    for num_patch in range(0, cm_reshape.shape[0]):
                        self.labels.append(torch.from_numpy(cm_reshape[num_patch]))

                    I1_patches = cut_image_strided(I1.transpose(2, 0, 1), (256, 256))
                    I1_patches_reshape = I1_patches.reshape(I1_patches.shape[0] * I1_patches.shape[1],
                                                            *I1_patches.shape[2:])
                    for num_patch in range(0, I1_patches_reshape.shape[0]):
                        self.imgs_1.append(torch.from_numpy(I1_patches_reshape[num_patch]))

                    I2_resized = cut_image_strided(I2.transpose(2, 0, 1), (256, 256))
                    I2_patches_reshape = I2_resized.reshape(I2_resized.shape[0] * I2_resized.shape[1],
                                                            *I2_resized.shape[2:])
                    for num_patch in range(0, I2_patches_reshape.shape[0]):
                        self.imgs_2.append(torch.from_numpy(I2_patches_reshape[num_patch]))
            print(n_pix0 / n_pix)
            print(n_pix1 / n_pix)
            print(n_pix2 / n_pix)
            print(n_pix3 / n_pix)
            print(n_pix4 / n_pix)
            print(n_pix5 / n_pix)
            print(n_pix6 / n_pix)
            print(n_pix7 / n_pix)
            self.weights = [1 / n_pix0, 1 / n_pix1, 1 / n_pix2, 1 / n_pix3, 1 / n_pix4, 1 / n_pix5, 1 / n_pix6,
                            1 / n_pix7]
        else:
            for planet_test_folder_name in tqdm(os.listdir(path)):
                # load and store each image
                one_lat_test_path = os.path.join(planet_test_boxes_path, planet_test_folder_name)
                if not os.path.isfile(one_lat_test_path):
                    for one_location_test_folder_name in os.listdir(one_lat_test_path):
                        test_folder_path = os.path.join(one_lat_test_path, one_location_test_folder_name, 'L3H-SR')
                        test_file_name_filtered = [test_file_name for test_file_name in os.listdir(test_folder_path) if
                                                   '-01.tif' in test_file_name]
                        test_file_name_filtered_and_sort = natsort.natsorted(test_file_name_filtered)
                        for idx, test_file_name in enumerate(test_file_name_filtered_and_sort):
                            if idx != 23:
                                original_image_path = os.path.join(test_folder_path, test_file_name)
                                changed_image_path = os.path.join(test_folder_path,
                                                                  test_file_name_filtered_and_sort[idx + 1])
                                I1, I2 = io.imread(original_image_path), io.imread(changed_image_path)
                                I1 = (I1 - I1.mean()) / I1.std()
                                I2 = (I2 - I2.mean()) / I2.std()
                                I1_patches = cut_image_strided(I1.transpose(2, 0, 1), (256, 256))
                                I1_patches_reshape = I1_patches.reshape(I1_patches.shape[0] * I1_patches.shape[1],
                                                                        *I1_patches.shape[2:])
                                for num_patch in range(0, I1_patches_reshape.shape[0]):
                                    self.imgs_1.append(torch.from_numpy(I1_patches_reshape[num_patch]))

                                I2_resized = cut_image_strided(I2.transpose(2, 0, 1), (256, 256))
                                I2_patches_reshape = I2_resized.reshape(I2_resized.shape[0] * I2_resized.shape[1],
                                                                        *I2_resized.shape[2:])
                                for num_patch in range(0, I2_patches_reshape.shape[0]):
                                    self.imgs_2.append(torch.from_numpy(I2_patches_reshape[num_patch]))

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.labels[im_name]

    def __len__(self):
        return len(self.imgs_1)

    def __getitem__(self, idx):

        if self.train:
            sample = {'I1': self.imgs_1[idx], 'I2': self.imgs_2[idx], 'label': self.labels[idx]}
        else:
            sample = {'I1': self.imgs_1[idx], 'I2': self.imgs_2[idx]}
        #
        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    def rgb_to_onehot(self, rgb_arr):
        num_classes = len(self.color_label_dict)
        shape = rgb_arr.shape[:2] + (num_classes,)
        arr = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(self.color_label_dict):
            arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == self.color_label_dict[i], axis=1).reshape(shape[:2])
        return arr

    def inverse_ohe(self, ohe_labels):
        """converts one-hot encoded mask to the multiclass mask"""
        inverse_ohe_img = np.zeros(ohe_labels.shape[:2] + (1,))
        for ch in range(ohe_labels.shape[-1]):
            ys, xs = np.where(ohe_labels[..., ch])
            inverse_ohe_img[ys, xs] = ch
        inverse_ohe_img = np.repeat(inverse_ohe_img, 3, axis=2).astype(int)
        return inverse_ohe_img
