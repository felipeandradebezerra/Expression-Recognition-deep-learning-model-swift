![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.9](https://img.shields.io/badge/Python-3.9-brightgreen.svg)![Tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)![Github](https://aleen42.github.io/badges/src/github.svg)

## Facial Expressions Recognition using Convolutional Neural Networks

### Description:
A deep learning model developed in PyTorch was successfully exported to Core ML and integrated into an iOS app built with Swift for real-time facial expression detection. The Core ML model is capable of recognizing emotions such as Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral, and Contempt, providing instant notifications based on the detected emotions.

To ensure the model's robustness, it was evaluated on both training and validation datasets at each epoch, helping to identify and mitigate issues like overfitting and underfitting. Metrics such as accuracy and loss were logged using TensorBoard's SummaryWriter for detailed visualization and analysis, enabling a more comprehensive understanding of the model's performance throughout the training process.

Once training was complete, the model was optimized for deployment by tracing it into a TorchScript format. This traced model was then converted into a Core ML model (.mlmodel) using coremltools, making it compatible with Apple devices and enhancing inference efficiency. The training and inference processes were accelerated by leveraging the Metal Performance Shaders (MPS) backend on a MacBook Air M3, which significantly improved performance through hardware acceleration.

Utilizing TensorBoard for tracking training and validation metrics was crucial for gaining insights into the model's behavior and ensuring its readiness for real-world deployment.

### Technologies:
- Python: PyTorch
- Swift: Core ML, Vision

## Main insights

The following architectures were used to train the model: efficientnet_b0, resnet101, resnet152, densenet121, densenet161, densenet201
The restnet152 model was the one that achieved the best result (almost 65% accuracy with only 100 epochs)

### Data Augmentation
Various data augmentation techniques were applied to the training data to improve generalization and avoid overfitting. 

These includes:
* Grayscale Conversion: Converts images to 3-channel grayscale.
* Random Resized Crop: Randomly crops the image and resizes it to the target size.
* Random Rotation: Rotates the image randomly within a specified range.
* Random Affine Transformations: Applies affine transformations like translation.
* Color Jitter: Randomly changes brightness, contrast, and saturation.
* Random Horizontal Flip: Horizontally flips images with a certain probability.
* Normalization: Normalizes the images using the mean and standard deviation of the dataset.

### Class Imbalance Handling
Set weights for each class based on the frequency of each class in the training set. This weight vector was used in the CrossEntropyLoss function to penalize misclassifications of minority classes more heavily, thus addressing class imbalance.

### Model Architecture
Pretrained resnet152 model as the base to speed up training and improve performance, especially because data is limited, by leveraging previously learned features.

### Fine-Tuning
Fine-tune only the classifier layers to learn the new task while keeping the pretrained feature extractor frozen.

### Gradient Clipping
During backpropagation, clip the gradients to a maximum norm of 1.0. This helped in preventing exploding gradients, which can destabilize training.

### Optimization
Set AdamW optimizer to prevent overfitting by penalizing large weights.
A ReduceLROnPlateau scheduler is used to reduce the learning rate when the validation loss plateaus. This can help the model converge to a better solution.

### Model Pruning
Applied a L1 unstructured pruning to both convolutional and linear layers to reduce model size and potentially improve generalization by removing less important connections.

### Datasets
[Kaggle Affectnet](https://www.kaggle.com/datasets/thienkhonghoc/affectnet)
