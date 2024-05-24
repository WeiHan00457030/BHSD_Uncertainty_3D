import json
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms

# get train and test file list
def load_train_test_list(json_path):

    f = open(json_path)
    data = json.load(f)

    trainset_list = []
    for file in data['training']:
        trainset_list.append(file['image'].split('/')[-1].split('.')[0])

    testset_list = []
    for file in data['test']:
        testset_list.append(file.split('/')[-1].split('.')[0])

    f.close()

    return trainset_list, testset_list


def truncate_hu_values(image, min_hu=-40, max_hu=120):
    truncated_image = np.clip(image, min_hu, max_hu)
    return truncated_image


def normalize_hu_values(image, min_hu=-40, max_hu=120):
    image = image - min_hu
    image = image / (max_hu + min_hu)
    return image

# slice and save to 2D
def slices_3D_to_2D(input_folder, file_name, output_folder, is_mask):

    full_path = f"{input_folder}/{file_name}.nii.gz"
    nifti = nib.load(full_path)
    data = nifti.get_fdata()

    if not is_mask:
        data = truncate_hu_values(data)
        data = normalize_hu_values(data)
        save_slices(data, output_folder, file_name, is_mask)
    else:
        save_slices(data, output_folder, file_name, is_mask)


def save_slices(data, output_folder, file_name, is_mask):
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        if not is_mask:
            slice_img = (slice_img * 255).astype(np.uint8)
        else:
            slice_img = slice_img.astype(np.uint8)
        img = Image.fromarray(slice_img)
        img.save(os.path.join(output_folder, f"{file_name}_{i}.png"))


def perprocess_2D(json_path, input_folder, output_folder, is_mask: bool):

    trainset_list, testset_list = load_train_test_list(json_path)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/train_set", exist_ok=True)
    os.makedirs(f"{output_folder}/test_set", exist_ok=True)

    for file in filter(lambda f: f.endswith(".nii.gz"), os.listdir(input_folder)):

        file_name = file.split('.')[0]  # ID_0c3eef60_ID_6994ad7df0
        if file_name in trainset_list:
            slices_3D_to_2D(input_folder, file_name,f"{output_folder}/train_set", is_mask)
        elif file_name in testset_list:
            slices_3D_to_2D(input_folder, file_name,f"{output_folder}/test_set", is_mask)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, target_class=None):

        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_class = target_class

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith('.png')])
        self.mask_files = sorted(
            [f for f in os.listdir(mask_folder) if f.endswith('.png')])

    def __getitem__(self, index):

        image_path = os.path.join(self.image_folder, self.image_files[index])
        mask_path = os.path.join(self.mask_folder, self.mask_files[index])

        # image normalization
        image = Image.open(image_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0

        image = self.transform(image)

        # mask to tensor
        mask = Image.open(mask_path)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # focus on single task
        if self.target_class is not None:
            mask[mask != self.target_class] = 0
            mask[mask == self.target_class] = 1

        return image, mask

    def __len__(self):
        # 返回數據集的大小
        return len(self.image_files)
