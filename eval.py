# PyTorch
import os

import natsort
import numpy as np
import torch
from skimage import io
from tqdm import tqdm

from model.siamunet_diff import SiamUnet_diff
from utils import cut_image_strided, load_config

config = load_config()
net, net_name = SiamUnet_diff(4, 8), 'FC-Siam-diff'
planet_test_boxes_path = config['model-param']['n_classes']

net.load_state_dict(config['results']['save_model_path'])
net.eval()
net.cuda()

check = []

for planet_test_folder_name in tqdm(os.listdir(planet_test_boxes_path)):
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
                    new_min = -1
                    new_max = 1
                    I1 = (I1 - np.min(I1)) / (np.max(I1) - np.min(I1)) * (new_max - new_min) + new_min
                    I2 = (I2 - np.min(I2)) / (np.max(I2) - np.min(I2)) * (new_max - new_min) + new_min
                    I1_patches = cut_image_strided(I1.transpose(2, 0, 1), (256, 256))
                    I1_patches_reshape = I1_patches.reshape(I1_patches.shape[0] * I1_patches.shape[1],
                                                            *I1_patches.shape[2:])

                    I2_resized = cut_image_strided(I2.transpose(2, 0, 1), (256, 256))
                    I2_patches_reshape = I2_resized.reshape(I2_resized.shape[0] * I2_resized.shape[1],
                                                            *I2_resized.shape[2:])

                    I1 = torch.from_numpy(I1_patches_reshape).float().cuda()
                    I2 = torch.from_numpy(I2_patches_reshape).float().cuda()
                    output = net(I1, I2)
                    maxes, predicted = torch.max(output.data, 1, keepdim=True)
                    np_result = predicted.reshape(1, 1024, 1024).cpu().numpy()
                    result_file_name = original_image_path[-38:-35] + '_' + original_image_path[
                                                                            -34:-22] + '-' + changed_image_path[
                                                                                             -14:-7] + '-' + original_image_path[
                                                                                                             -14:-7] + '.png'
                    io.imsave('results_final/' + result_file_name, np.squeeze(np_result).astype(np.uint8))

                    check.append(np.squeeze(np_result).tolist())

