# Neural Methods of Image Processing

An educational course on computer vision and deep learning. Built around hands-on practice with PyTorch: from basic tensor operations to transfer learning on pre-trained architectures.

## What the course is about

Modern image processing is built on convolutional neural networks that learn a hierarchy of features directly from pixels — from simple edges and textures to complex objects. The course walks through the key stages of working with such models: data preparation, augmentation, and training strategies.

**Data.** An image in a neural network is a tensor in `C×H×W` format (channels × height × width), and a batch is `B×C×H×W`. Pixel normalization stabilizes training, while `DataLoader` feeds data in shuffled batches.

**Augmentation.** Limited data is one of the main challenges in CV. Geometric and color transformations (flip, crop, rotation, color jitter) artificially expand the dataset and reduce overfitting. For segmentation, geometric augmentations are applied synchronously to both image and mask, with the mask interpolated using NEAREST to avoid blending class labels.

**Transfer learning.** Training from scratch requires large datasets. Models pre-trained on ImageNet already contain general-purpose features, so they are adapted to a new task in two ways: *feature extraction* (frozen backbone, only the classifier is trained) and *fine-tuning* (the last block is unfrozen together with the head). Comparing these strategies against training from scratch clearly shows how knowledge transfer speeds up convergence and improves accuracy.

**Architectures.** ResNet uses residual connections to combat vanishing gradients, DenseNet uses dense connections for feature reuse, and EfficientNet scales depth, width, and resolution. Each balances accuracy and computational cost in its own way.