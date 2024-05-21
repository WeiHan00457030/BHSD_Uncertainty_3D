import json
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms


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


def slices_3D_to_2D(input_folder, file_name, output_folder, is_mask):

    full_path = f"{input_folder}/{file_name}.nii.gz"
    nifti = nib.load(full_path)

    data = nifti.get_fdata()

    if not is_mask:
        data = truncate_hu_values(data)
        data = normalize_hu_values(data)
        for i in range(data.shape[2]):
            slice_img = data[:, :, i]
            # Convert to 8-bit grayscale
            slice_img = (slice_img * 255).astype(np.uint8)
            img = Image.fromarray(slice_img)
            img.save(f"{output_folder}/{file_name}_{i}.png")
    else:
        for i in range(data.shape[2]):
            slice_mask = data[:, :, i]
            slice_mask = slice_mask.astype(
                np.uint8)  # Convert to 8-bit integers

            img = Image.fromarray(slice_mask)
            img.save(f"{output_folder}/{file_name}_{i}.png")


def perprocess_2D(json_path, input_folder, output_folder, is_mask: bool):

    trainset_list, testset_list = load_train_test_list(json_path)

    for file in os.listdir(input_folder):  # ID_0c3eef60_ID_6994ad7df0.nii.gz

        if file.endswith(".nii.gz"):
            file_name = file.split('.')[0]  # ID_0c3eef60_ID_6994ad7df0
            if file_name in trainset_list:
                slices_3D_to_2D(input_folder, file_name,
                                f"{output_folder}/train_set", is_mask)
            elif file_name in testset_list:
                slices_3D_to_2D(input_folder, file_name,
                                f"{output_folder}/test_set", is_mask)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, target_class=None, transform=None):

        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_class = target_class

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
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

        return image, mask

    def __len__(self):
        # 返回數據集的大小
        return len(self.image_files)
