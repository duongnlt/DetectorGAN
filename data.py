import os

import cv2
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset


class GANDataset(Dataset):
    def __init__(self, path_to_abnormal_images, path_to_annotation, path_to_clean_images):
        super(GANDataset, self).__init__()
        self.path_to_abnormal_images = path_to_abnormal_images
        self.path_to_clean_images = path_to_clean_images
        self.path_to_annotation = path_to_annotation

        self.abnormal_images =  [f for f in os.listdir(self.path_to_abnormal_images)]
        self.clean_images = [f for f in os.listdir(self.path_to_clean_images)]

        self.dataset_len = max(len(self.abnormal_images), len(self.clean_images))
    
    def __len__(self):
        return self.dataset_len


    def __getitem__(self, item):
        '''Return an item from dataset

        '''
        abnormal_img_path = os.path.join(self.path_to_abnormal_images, self.abnormal_images[item])
        abnormal_img = Image.open(abnormal_img_path).convert('RGB')
        abnormal_img = FT.to_tensor(abnormal_img)

        # bounding boxes
        bboxes = []
        path_to_annotation_img = os.path.join(self.path_to_annotation, self.abnormal_images[item].split('.')[0] + '.txt')
        with open(path_to_annotation_img, 'r') as f:
            for line in f:
                bboxes.append([float(x) for x in line.strip().split(' ')])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        abnormal_mask = get_mask(*bboxes[1:])

        normal_img_path = os.path.join(self.path_to_clean_images, self.clean_images[item])
        normal_img = Image.open(normal_img_path).convert('RGB')
        normal_img = FT.to_tensor(normal_img)

        
    


    def get_mask(self, cx, cy, w, h):
        mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        mask[:, torch.round((cy-h/2)*1024).int():torch.round((cy+h/2)*1024).int(),
                  torch.round((cx-w/2)*1024).int():torch.round((cx+w/2)*1024).int()] = 1.
        return mask
