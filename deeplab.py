import copy
import torch
import csv
import os
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, roc_auc_score, jaccard_score
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def get_deeplabv3_model(outputchannels=2):
    """Get DeepLabv3 model with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet50 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                   progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)

    return model


def deeplabv3_trainer(model, train_loader, valid_loader, device, epochs=20):
    """Train DeepLabv3 model """
    # Set the model in training mode
    model.to(device)

    # Specify the loss function
    criterion = smp.losses.DiceLoss('binary')

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # define metrics
    metrics = {'IoU': jaccard_score}

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = {
        'DiceLoss': [], 'IoU': []}, {'DiceLoss': [], 'IoU': []}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        for mode in ['Train', 'Test']:
            if mode == 'Train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                logs = train_logs_list
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_loader
                logs = valid_logs_list

            # Iterate over data.
            for sample in tqdm(iter(dataloader)):
                sample_logs = []
                inputs = sample[0].to(device)
                masks = sample[1].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(mode == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    for name, metric in metrics.items():
                        if name == 'IoU':
                            sample_logs.append(
                                metric(y_true, y_pred > 0.5))
                    if mode == 'Train':
                        loss.backward()
                        optimizer.step()

            logs['Diceloss'].append(loss.item())
            logs['IoU'].append(np.mean(sample_logs))

            print('{} DiceLoss: {:.4f}'.format(mode, loss))
            print('{} IoU: {:.4f}'.format(mode, valid_logs_list['IoU'][-1]))

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs_list['IoU'][-1]:
            best_iou_score = valid_logs_list['IoU'][-1]
            torch.save(model, './saved_models/best_deeplabv3_model.pth')
            print('Model saved!')

    return model, train_logs_list, valid_logs_list
