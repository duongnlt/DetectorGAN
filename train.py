import torch
import cv2
import torchvision.transforms.functional as FT
from PIL import Image

image = Image.open('/home/duong/Downloads/00000001_000.png').convert('RGB')

image = FT.to_tensor(image)
# # image = image.permute(2, 0 ,1)
image = FT.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# cv2.imshow('image', image)
# cv2.waitKey(0)
print(image.shape)
print(image)