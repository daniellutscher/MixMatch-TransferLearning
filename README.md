# Combining MixMatch with Transfer Learning

This script combines the Semi-Supervised-Learning method MixMatch with transfer learning to fine-tune a pre-trained [Efficient-Net](https://github.com/lukemelas/EfficientNet-PyTorch) model on a chest x-ray images dataset.

The MixMatch method was proposed by the Google Research team, details here: [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch) and the pytorch implementation I based my project on is from [MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch).

Currently, the script only contains the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) as well as a [chest x-ray images](https://www.kaggle.com/nih-chest-xrays/sample) dataset.


## Requirements
- Python 3.6+
- PyTorch 1.0
- torchvision
- tensorboardX
- progress
- matplotlib
- numpy
- efficientnet_pytorch

## Usage
### Preprocess X-ray dataset
Download the X-ray dataset from the [kaggle website](https://www.kaggle.com/nih-chest-xrays/sample/downloads/sample.zip/4). 
Next, create the dataset and x_ray_images folder:
```
mkdir -p dataset/x_ray_images
```
Then extract the zip file into /dataset/x_ray_images and run make_x_ray_dataset.py


### Train
Train the EfficientNet model with 250 labeled data of the x-ray dataset, a batch size of 16, and a learning rate of 0.0001. Freeze all layers except the last for the first 5 epochs, then unfreeze all layers for fine-tuning:

```
python main.py --lr 0.0001 --out x_ray@250 --batch_size 16 --unfreeze 5 --dataset x_ray --n-labeled 250 --model efficient
```

Train the ResNet model by 4000 labeled data of CIFAR-10 dataset:

```
python main.py --lr 0.002 --batch-size 64 --dataset cifar --n-labeled 4000 --out cifar10@4000 --model resnet
```

## Current Performance
Given computational limitations, I've only been able to run the MixMatch script with the efficientNet-b0 model for around 50 epochs (see To-Do). Adding the SSL method largely improved on the classification performance, increasing from ±28% test set accuracy (transfer learning with all 2800 labels) to ±44.5% test accuracy using only 250 labeled images.

## TO-DO
- run models for full epoch duration
- add requirements.txt


## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```
