import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader
import albumentations as album

SEED = 42

# Perform one hot encoding on label


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

# Perform reverse one-hot-encoding on labels / preds


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x

# Perform colour coding on the reverse-one-hot outputs


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# helper function for data visualization


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


class SkyDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(
            images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id)
                           for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read images and masks
        image = cv2.cvtColor(cv2.imread(
            self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # apply preprocessing

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None, height=128, width=128):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [album.Resize(height, width)]
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def get_train_val_test_datasets(
    x_dir,
    y_dir,
    select_class_rgb_values,
    batch_size=32,
    preprocessing=None
):
    """Get train, validation and test datasets & dataloaders    
    Args:
        x_dir (str): path to images folder
        y_dir (str): path to segmentation masks folder
        select_class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    Return:
        Tupel: DataLoaders, SkyDatasets
    """
    # Get train and val dataset instances
    dataset = SkyDataset(
        x_dir, y_dir,
        preprocessing=preprocessing,
        class_rgb_values=select_class_rgb_values,
    )
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [0.6, 0.2,
                                                                                   0.2],
                                                                               generator=torch.Generator().manual_seed(SEED))
    # Get train, val and test data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return ((train_dataset, valid_dataset, test_dataset),
            (train_loader, valid_loader, test_loader))


def get_vis_test_set(x_dir, y_dir, select_class_rgb_values, preprocessing=None):
    """Get test dataset for visualisation without preprocessing   
    Args:
        x_dir (str): path to images folder
        y_dir (str): path to segmentation masks folder
        select_class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    Return:
        SkyDataset
    """
    dataset = SkyDataset(
        x_dir, y_dir,
        preprocessing=preprocessing,
        class_rgb_values=select_class_rgb_values,
    )
    _, _, test_dataset_vis = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2],
                                                           generator=torch.Generator().manual_seed(SEED))
    return test_dataset_vis
