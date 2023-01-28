import os
import pandas as pd
import numpy as np
import torch
from data_utils import get_train_val_test_datasets, get_preprocessing
from unet import get_unit_model, unet_trainer
from deeplab import get_deeplabv3_model, deeplabv3_trainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 32
    train_unet = True
    train_deeplab = True

    DATA_DIR = './ADE20K_outdoor/'
    x_train_dir = os.path.join(DATA_DIR, 'images/training')
    y_train_dir = os.path.join(DATA_DIR, 'annotations/training_binary')
    class_dict = pd.read_csv("./label_class_dict_2.csv")
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['other', 'sky']
    # Get RGB values of required classes
    select_class_indices = [class_names.index(
        cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    if train_unet:
        # Get Unit Model
        encoder = 'resnet50'
        encoder_weights = 'imagenet'
        model, preprocessing_fn = get_unit_model(
            encoder, encoder_weights, len(class_names))

        # Get Dataloaders
        datasets, dataloaders = get_train_val_test_datasets(
            x_train_dir,
            y_train_dir,
            select_class_rgb_values,
            batch_size=batch_size,
            preprocessing=get_preprocessing(preprocessing_fn))
        train_loader, valid_loader, _ = dataloaders

        model, train_logs_list, valid_logs_list = unet_trainer(
            model,
            train_loader,
            valid_loader,
            device=device,
            epochs=epochs
        )
    if train_deeplab:
        # Get DeepLabv3 Model
        model = get_deeplabv3_model(len(class_names))

        # Get Dataloaders
        datasets, dataloaders = get_train_val_test_datasets(
            x_train_dir,
            y_train_dir,
            select_class_rgb_values,
            batch_size=batch_size,
            preprocessing=get_preprocessing())
        train_loader, valid_loader, _ = dataloaders

        model, train_logs_list, valid_logs_list = deeplabv3_trainer(
            model,
            train_loader,
            valid_loader,
            device=device,
            epochs=epochs
        )
