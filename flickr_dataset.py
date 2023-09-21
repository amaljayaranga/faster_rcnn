import torch
from torch.nn import functional as F
from torch.utils.data import  Dataset
import copy
import cv2
import albumentations as A 
import os
from pycocotools.coco import COCO

import albumentations as A 
from albumentations.pytorch import ToTensorV2

def get_detect_transforms(train=False):

    if train:
         transform = A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    else:
        transform = A.Compose([
            A.Resize(600, 600),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    return transform

def get_crop_transform():
    crop_transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
    ])
    return crop_transform


class FlickrDataset(Dataset):
    def __init__(self, split, dataset_path, detect_transform=None, crop_transform=None):
        super().__init__()
        self.split = split
        self.dataset_path = dataset_path
        self.detect_transform = detect_transform
        self.crop_transform = crop_transform
        self.coco = COCO(os.path.join(self.dataset_path, split, split+"_coco_annotations.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def load_image(self, id):
        image_name = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.dataset_path, self.split, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        original_image = self.load_image(id)
        detect_image = copy.deepcopy(original_image)
        crops_image = copy.deepcopy(original_image)
        target = self.load_target(id)
        target = copy.deepcopy(self.load_target(id))

        bboxes = [t['bbox']  for t in target]  # x,y,w,h
        category_ids = [t['category_id']  for t in target]

        if self.detect_transform is not None:
            transformed = self.detect_transform(image=detect_image, bboxes= bboxes, category_ids=category_ids)

            detect_image = transformed['image']
            boxes = transformed['bboxes']
      
        boxes_xyxy = [] # convert bboxes from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            boxes_xyxy.append([xmin, ymin, xmax, ymax])
        
        
        tar = {}
        tar['boxes'] = torch.tensor(boxes_xyxy, dtype=torch.float32)
        tar['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        tar['image_id'] = torch.tensor([t['image_id'] for t in target])

        logo_crops = {}

        if self.split != 'train':
            return detect_image/255, tar, logo_crops
    
        # create image crops
        for box, cat_id in zip(bboxes, category_ids):
            crop = crops_image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

            if self.crop_transform:
                transformed_crop = self.crop_transform(image=crop)['image']
                transformed_crop = transformed_crop/255

            
            if cat_id not in list(logo_crops.keys()):
                logo_crops[cat_id] = [transformed_crop]
            else:
                existing_crops = logo_crops.get(cat_id)
                existing_crops.append(transformed_crop)
                logo_crops[cat_id] = existing_crops


        return detect_image/255, tar, logo_crops

    
    def __len__(self):
        return len(self.ids)

dataset_path = './final_dataset'
if __name__ == '__main__':
    fd = FlickrDataset('train', dataset_path, detect_transform=get_detect_transforms(), crop_transform=get_crop_transform())
    fd.ex(2)