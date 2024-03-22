import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import os
# import ultralytics
sys.path.append('/mundus/mrahman528/thesis/github-thesis/Object-Detection-and-OOD-Detection')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ultralytics')))

from dataset.dataset import CustomYOLOv8Dataset, custom_collate_fn, SeaThru
from ultralytics.models.yolo.detect import DetectionTrainer

train_img_dir = '/mundus/mrahman528/thesis/detr/dataset/train'
train_label_dir = '/mundus/mrahman528/thesis/fgvc-comp-2023/dataset/labels/train'
train_depth_dir = '/mundus/mrahman528/thesis/Depth-Anything/depth_vis_g/'

test_img_dir = '/mundus/mrahman528/thesis/detr/dataset/val'
test_label_dir = '/mundus/mrahman528/thesis/fgvc-comp-2023/dataset/labels/val'
test_depth_dir = '/mundus/mrahman528/thesis/Depth-Anything/depth_vis_g_val/'
# random_transform = transforms.RandomApply([CustomTransform], p=0.9)
transform = transforms.Compose([
    transforms.Resize((640,640)),
    # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.05),
    # transforms.RandomResizedCrop(size=(224, 224), antialias=True),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    transforms.ToTensor(),
])

# callback to upload model weights
def log_model(trainer):
    last_weight_path = trainer.last
    print(last_weight_path)

# mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_dataset = CustomYOLOv8Dataset(train_img_dir, train_depth_dir, train_label_dir, img_size=640, transform=transform,custom_transform= SeaThru(mode="train",depth_offset=True,random_B_gamma = True,shuffle_B_gamma=True,depth_pixel=True),custom_transform_prob=0.5)
# test_dataset = CustomYOLOv8Dataset(test_img_dir, test_depth_dir, test_label_dir, img_size=640, transform=transform, custom_transform= SeaThru(mode="val",depth_offset=False,random_B_gamma = True,shuffle_B_gamma=True,depth_pixel=True))
# test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn = custom_collate_fn)

# train_data_loader_actual = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, collate_fn = custom_collate_fn)



class CustomTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialization code as before

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        train_dataset = CustomYOLOv8Dataset(train_img_dir, train_depth_dir, train_label_dir, img_size=640, transform=transform,custom_transform= SeaThru(mode="train",depth_offset=True,random_B_gamma = True,shuffle_B_gamma=True,depth_pixel=True),custom_transform_prob=0.5)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, collate_fn = custom_collate_fn)

        return train_data_loader


args = dict(model='yolov8m.pt', data='/mundus/mrahman528/thesis/github-thesis/Object-Detection-and-OOD-Detection/dataset/dataset.yaml', epochs=1)
trainer = CustomTrainer(overrides=args)
trainer.add_callback("on_train_epoch_end", log_model)
trainer.train()
trained_model = trainer.best
