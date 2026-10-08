# Training Curves

This folder contains the training and validation loss curves generated during U-Net training.

The model was trained for 50 epochs using `BCEWithLogitsLoss` and the Adam optimizer.

The loss curve shows that both training and validation loss generally decrease over time, indicating that the model learns to segment lesion regions from the input images.

The validation loss is slightly more variable than the training loss, which is expected because the validation set contains unseen images.

The main training curve included in this folder is:

`training_loss.png`
