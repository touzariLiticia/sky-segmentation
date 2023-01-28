import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import torch


def get_unit_model(encoder, encoder_weights, classes, activation='sigmoid'):
    """Get Unet model with encoder and encoder_weights
    Args:
        encoder (str): pretrained encoder name
        encoder_weights (str): pre-training on ImageNet

    Returns:
        model: Returns the Unet model with the ResNet50 encoder and 
        preprocessing function.
    """
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder, encoder_weights)
    return model, preprocessing_fn


def unet_trainer(model, train_loader, valid_loader, device, epochs=20):
    """Train Unet model """

    # define loss function
    loss = smp.losses.DiceLoss('binary')
    loss.__name__ = 'DissLoss'

    # define metrics
    iou_score = smp.utils.metrics.IoU(threshold=0.5)
    iou_score.__name__ = 'IoU'
    metrics = [iou_score]

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, epochs):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['IoU']:
            best_iou_score = valid_logs['IoU']
            torch.save(model, './saved_models/best_unet_model.pth')
            print('Model saved!')

    return model, train_logs_list, valid_logs_list
