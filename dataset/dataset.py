from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import torch
from torchvision.transforms.functional import to_tensor, resize
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T

def histogram_stretching(image: torch.Tensor):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    image_copy = image.copy()
    valid_mask = ~np.isnan(image_copy).any(axis=2)
    
    stretched_image = np.zeros_like(image_copy)
    
    for c in range(image_copy.shape[2]):
        channel = image_copy[:, :, c]
        valid_pixels = channel[valid_mask]
        
        if valid_pixels.size == 0:  # Check if there are no valid pixels
            continue  # Skip this channel if empty
        
        # Flatten valid_pixels for percentile calculation if necessary
        valid_pixels_flat = valid_pixels.flatten()
        
        lower_percentile = np.percentile(valid_pixels_flat, 1)
        upper_percentile = np.percentile(valid_pixels_flat, 99)
        
        # Clip and stretch the valid pixel values
        valid_pixels_clipped = np.clip(valid_pixels, lower_percentile, upper_percentile)
        valid_pixels_stretched = (valid_pixels_clipped - lower_percentile) / (upper_percentile - lower_percentile)
        
        # Update stretched_image with stretched valid pixels
        stretched_channel = np.zeros_like(channel)
        stretched_channel[valid_mask] = valid_pixels_stretched
        stretched_image[:, :, c] = stretched_channel

    return torch.from_numpy(stretched_image).float()
    
# def compute_J(I, z, B, beta, gamma):
#     return (I - B * (1 - torch.exp(-gamma * z))) * torch.exp(beta * z)

# def compute_I(J,beta,z,B,gamma):
#     return (J * torch.exp(-beta * z) + B * (1 - torch.exp(-gamma * z)))
def compute_I_without_J(I_orig,beta,z,z_mod,B,gamma):
    a = torch.exp(-beta * z)
    b = torch.exp(-beta * z_mod)
    I_mod = (I_orig * b + B * (a*(1 - torch.exp(-gamma * z_mod)) - b * (1 - torch.exp(1 - torch.exp(-gamma * z)))))/a
    return I_mod

class SeaThru:
    def __init__(self,mode='train',depth_offset=True,random_B_gamma = False,shuffle_B_gamma=True,depth_pixel=True):
        self.mode = mode
        self.depth_offset= depth_offset
        self.random_B_gamma = random_B_gamma,
        self.shuffle_B_gamma = shuffle_B_gamma,
        self.depth_pixel = depth_pixel
    
    def __call__(self, img_name):  # Expects a tuple of (image, depth_image)
        # img_name, img, depth_img = imgs  # Unpack the tuple
        
        transformed_img = seathru(img_name,self.mode,self.depth_offset,self.shuffle_B_gamma,self.depth_pixel )
        if transformed_img is None:
            # Handle the None case appropriately
            raise ValueError("Transformed image is None")
            
        return transformed_img  # Return the transformed image
    
    def __repr__(self):
        return self.__class__.__name__ + f'(transform_param={self.transform_param})'


import json
import glob
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Assuming restored_image is a PIL Image
to_tensor_transform = transforms.ToTensor()
def seathru(image_name,mode="train",depth_offset=True,random_B_gamma = False,shuffle_B_gamma=True,depth_pixel=True):
    # Define the paths
    # print(depth_offset, B_mean, gamma_mean, shuffle_B_gamma, depth_pixel)
    depth_off = 0
    random_B = 0
    random_gamma = 0
    if mode == 'train':
        image_dir = '/mundus/mrahman528/thesis/detr/dataset/train/'
        depth_dir = '/mundus/mrahman528/thesis/Depth-Anything/depth_vis_g/'
        json_file_path = '/mundus/mrahman528/thesis/sucre/parameters_train.json'  # Update this path
    else:
        image_dir = '/mundus/mrahman528/thesis/detr/dataset/val/'
        depth_dir = '/mundus/mrahman528/thesis/Depth-Anything/depth_vis_g_val/'
        json_file_path = '/mundus/mrahman528/thesis/sucre/parameters_val.json'

    # Load the JSON data containing parameters
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Construct file paths
    image_file = os.path.join(image_dir, image_name)
    depth_file = os.path.join(depth_dir, os.path.splitext(image_name)[0] + '_depth.png')

    # Read and process the image
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    # Read and process the depth image
    # depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
    normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    distance_map = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    distance_map = torch.tensor(distance_map, dtype=torch.float32).to(device)
    distance_map_gray = torch.mean(distance_map, dim=2).to(device)
    distance_map_gray_updated = distance_map_gray
    # Only select pixels where there is depth information
    v_valid, u_valid = torch.where(distance_map_gray > 0)
    image_valid = image[v_valid, u_valid].to(device)

    #generate random depth offset
    if depth_offset:
        depth_off = random.uniform(-50,100)
        # print(f'depth_offset - {depth_off}')

    #generate random B and gamma
    if random_B_gamma:
        random_B = random.uniform(1e-6,1)
        # random_B = random.randrange(0,3)
        # print(f"random_B - {random_B}")
        random_gamma = random.uniform(0.0022304835938825366, 1.4107012727967811)
        # random_gamma = random.randrange(1, 5)
        # print(f"random_gamma - {random_gamma}")
    #Change the corner of the depth map
    if depth_pixel:
        # print(distance_map_gray[0,0])
        distance_map_gray_updated[0,0] = distance_map_gray[0,0] + random.randrange(0,255)
        # print(distance_map_gray_updated[0,0])
        # print(distance_map_gray[0,0])
        distance_map_gray_updated[-1,-1] = distance_map_gray[-1,-1] - random.randrange(0,255)
        # print(distance_map_gray_updated[-1,-1])
    # else:
    #     distance_map_valid = distance_map_gray[v_valid, u_valid]
    
    distance_map_valid = distance_map_gray[v_valid, u_valid].to(device)
    distance_map_valid_updated = distance_map_gray_updated[v_valid, u_valid].to(device)
    betac, gammac, Bc = [], [], []
    if image_name in data:
        for channel in ['channel_0', 'channel_1', 'channel_2']:
            
            if shuffle_B_gamma:
                #get a random image
                random_image_name = random.choice(list(data.keys()))
                #do not change beta c
                betac.append(data[image_name][channel]['betac'])
                #change gamma and B
                gammac.append(data[random_image_name][channel]['gammac'])
                Bc.append(data[random_image_name][channel]['Bc'])
            else:
                betac.append(data[image_name][channel]['betac'])
                gammac.append(data[image_name][channel]['gammac'])
                Bc.append(data[image_name][channel]['Bc'])
    else:
        print(f"Data for {image_name} not found in the data structure.")
        return None
    # J_cp = cv2.imread('/mundus/mrahman528/thesis/sucre/output_gaussian_seathru_dataset/'+ image_name, cv2.IMREAD_UNCHANGED)
    # J_cp = torch.tensor(J_cp, dtype=torch.float32).to(device)
    # J_cp_valid = J_cp[v_valid,u_valid]
    
    # J_cp = torch.full(image.shape, torch.nan, dtype=torch.float32).to(device)
    # for channel in range(3):
    #     beta = torch.tensor(betac[channel]).to(device)
    #     gamma = torch.tensor(gammac[channel]).to(device)
    #     B = torch.tensor(Bc[channel]).to(device)
    #     J_cp[v_valid, u_valid, channel] = compute_J(
    #             image_valid[:, channel], distance_map_valid, B, beta, gamma
    #         )
    # J_cp_valid = J_cp[v_valid,u_valid]

    # J_cp_valid = J_cp_valid.to(device)

    # # J_cp = Image.fromarray(np.uint8(histogram_stretching(J_cp)*255))
    # I = torch.full(J_cp.shape, torch.nan, dtype=torch.float32).to(device)
    # for channel in range(3):
    #     I[v_valid, u_valid, channel] = compute_I(
    #         J_cp_valid[:, channel], betac[channel],distance_map_valid_updated + depth_offset, Bc[channel] + B_mean[channel], gammac[channel] + gamma_mean[channel]
    #     )

    I_mod = torch.full(image.shape, torch.nan, dtype=torch.float32).to(device)

    for channel in range(3):
        I_mod[v_valid,u_valid,channel] = compute_I_without_J(
            image_valid[:,channel], betac[channel] - 5e-3, distance_map_valid, distance_map_valid_updated + depth_off ,Bc[channel] + random_B,gammac[channel] + random_gamma)
        
    
    # Convert to PIL Image for return, assuming histogram_stretching is correctly implemented
    restored_image = Image.fromarray(np.uint8(histogram_stretching(I_mod) * 255))
    return to_tensor_transform(restored_image)


class CustomYOLOv8Dataset(Dataset):
    def __init__(self, img_dir, depth_dir, label_dir, img_size=640, transform=None,custom_transform=None, custom_transform_prob=0.4):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.label_dir = label_dir
        self.custom_transform_prob = custom_transform_prob
        self.img_size = img_size
        self.transform = transform
        self.custom_transform= custom_transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = self.load_all_labels()

    def load_all_labels(self):
        all_labels = []
        for img_name in self.img_names:
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
            labels = self.load_yolo_labels(label_path)
            bboxes = labels[:, 1:]  # Assuming the format is [class_id, x_center, y_center, width, height]
            cls = labels[:, 0:1]  # Class ids
            all_labels.append({"bboxes": bboxes, "cls": cls})
        return all_labels
    
    def __len__(self):
        return len(self.img_names)

    # def __getitem__(self, idx):
    #     img_name = self.img_names[idx]
    #     img_path = os.path.join(self.img_dir, img_name)
    #     depth_img_path = os.path.join(self.depth_dir, os.path.splitext(img_name)[0] + '_depth.png')

    #     img = Image.open(img_path).convert('RGB')
    #     depth_img = Image.open(depth_img_path).convert('RGB')

    #     # Apply transformations that use both image and depth image
    #     if self.transform:
    #         img = self.transform(img)  # Adjusted to pass depth_img as well
    #         # print("transformed Image size : ", img.shape , type(img))
    #     if self.custom_transform and random.random() < self.custom_transform_prob:
    #         img = self.custom_transform(img_name)
    #         # print("hello")

    #     label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
    #     labels = self.load_yolo_labels(label_path)
    #     # print("before returning: ", img.shape)
    #     assert isinstance(labels, dict), f"Labels must be a dictionary, got {type(labels)} instead."
    #     assert 'bboxes' in labels, "'bboxes' key missing in labels."
    #     assert 'cls' in labels, "'cls' key missing in labels."
    #     return img, labels  # Only return the transformed image and labels


    # def load_yolo_labels(self, label_path):
    #     """
    #     Load labels from a YOLO format label file.
    #     """
    #     labels = []
    #     if os.path.exists(label_path):
    #         with open(label_path, 'r') as file:
    #             for line in file:
    #                 class_id, x_center, y_center, width, height = map(float, line.split())
    #                 labels.append([class_id, x_center, y_center, width, height])
    #     return np.array(labels, dtype=np.float32)
    def load_yolo_labels(self, label_path):
        """
        Load labels from a YOLO format label file and return as a dictionary.
        """
        bboxes = []
        cls = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    bboxes.append([x_center, y_center, width, height])
                    cls.append(class_id)
        return {'bboxes': np.array(bboxes, dtype=np.float32), 'cls': np.array(cls, dtype=np.int64)}

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_img_path = os.path.join(self.depth_dir, os.path.splitext(img_name)[0] + '_depth.png')

        img = Image.open(img_path).convert('RGB')
        depth_img = Image.open(depth_img_path).convert('RGB')

        # Apply transformations that use both image and depth image
        if self.transform:
            img = self.transform(img)  # Adjusted to pass depth_img as well
        if self.custom_transform and random.random() < self.custom_transform_prob:
            img = self.custom_transform(img_name)

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        labels = self.load_yolo_labels(label_path)

        # Ensure labels are returned correctly
        assert isinstance(labels, dict), f"Labels must be a dictionary, got {type(labels)} instead."
        assert 'bboxes' in labels, "'bboxes' key missing in labels."
        assert 'cls' in labels, "'cls' key missing in labels."

        return img, labels



import torch
from torchvision.transforms.functional import to_tensor, resize
import numpy as np

def custom_collate_fn(batch):
    processed_images = []
    all_bboxes = []
    all_cls = []

    for item in batch:
        image, labels = item
        processed_images.append(to_tensor(resize(image, size=(640, 640))))

        # Validate and collect labels
        if isinstance(labels, dict) and 'bboxes' in labels and 'cls' in labels:
            all_bboxes.append(torch.tensor(labels['bboxes'], dtype=torch.float32))
            all_cls.append(torch.tensor(labels['cls'], dtype=torch.long))
        else:
            print(f"Warning: Incorrect label format encountered for item: {labels}")
            # Optionally, handle the case where labels are not as expected
            # For example, append empty tensors to maintain batch size
            all_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
            all_cls.append(torch.empty((0,), dtype=torch.long))

    images_batch = torch.stack(processed_images)
    bboxes_batch = torch.cat(all_bboxes) if all_bboxes else torch.empty((0, 4), dtype=torch.float32)
    cls_batch = torch.cat(all_cls) if all_cls else torch.empty((0,), dtype=torch.long)

    return images_batch, {"bboxes": bboxes_batch, "cls": cls_batch}




