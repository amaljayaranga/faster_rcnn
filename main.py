import torch
import numpy as np
from torch.utils.data import DataLoader
import math
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import argparse

import os
from tqdm import tqdm 
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from flickr_dataset import FlickrDataset
from detection_model import DetectionModel
from train_model import train_model
from eval_model import eval_model


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


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN model')
  parser.add_argument('--dataset_path', dest='dataset_path',
                      help='the path of dataset',
                      default='final_dataset', type=str)
  
  parser.add_argument('--model_save_path', dest='model_save_path',
                      help='the path where model weights are saved',
                      default='weights', type=str)
  
  parser.add_argument('--tensorboard_logs_save_dir', dest='tensorboard_logs_save_dir',
                      help='the path where Tensor Board logs are saved',
                      default='runs', type=str)
  
  parser.add_argument('--weights_name', dest='weights_name',
                      help='name of model weight',
                      default='faster_rcnn', type=str)
  
  parser.add_argument('--train_anno', dest='train_anno',
                      help='name of coco train annotations',
                      default='train_coco_annotations.json', type=str)
  
  parser.add_argument('--batch_size', dest='batch_size',
                      help='batch size',
                      default=2, type=int)
  
  parser.add_argument('--lr', dest='lr',
                      help='learning rate',
                      default=0.001, type=float)
  
  parser.add_argument('--weight_decay', dest='weight_decay',
                      help='learning rate',
                      default=0.0005, type=float)
  
  parser.add_argument('--momentum', dest='momentum',
                      help='momentum',
                      default=0.9, type=float)
  
  parser.add_argument('--n_epochs', dest='n_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  
  parser.add_argument('--alpha', dest='alpha',
                      help='hyper parameter for adding Triplet Loss to main loss',
                      default=0.8, type=float)
  

  args = parser.parse_args()
  return args

      
if __name__ == '__main__':

    args = parse_args()
   

    last_run = 1
    model_weights_path = ''


    # create folder to save tensorboard runs and model weights
    if os.path.exists(args.tensorboard_logs_save_dir):
        last_run = int(os.listdir(args.tensorboard_logs_save_dir)[-1])+1
        os.makedirs(os.path.join(os.getcwd(),args.tensorboard_logs_save_dir, str(last_run)))
        model_weights_path = os.path.join(os.getcwd(),args.model_save_path, str(last_run))
        os.makedirs(model_weights_path)
    else: 
        os.makedirs(os.path.join(os.getcwd(),args.tensorboard_logs_save_dir))
        os.makedirs(os.path.join(os.getcwd(),args.tensorboard_logs_save_dir, str(last_run)))

        os.makedirs(os.path.join(os.getcwd(),args.model_save_path))
        model_weights_path = os.path.join(os.getcwd(),args.model_save_path, str(last_run))
        os.makedirs(model_weights_path)

    
    writer = SummaryWriter('runs/'+str(last_run))
    print(f'TensorBoard Logs are saved in runs/{str(last_run)}')

    dataset_path = os.path.join(os.getcwd(),args.dataset_path)


    train_dataset = FlickrDataset('train', dataset_path, detect_transform=get_detect_transforms(train=True), crop_transform=get_crop_transform())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)

    eval_dataset = FlickrDataset('eval', dataset_path, detect_transform=get_detect_transforms())
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # load COCO annotations
    coco = COCO(os.path.join(dataset_path, "train", args.train_anno))
    categories = coco.cats
   
    n_classes = len(categories.keys())
    classes = [i[1]['name'] for i in categories.items()]

    print(f'dataset has {n_classes} classes and classes are {classes}')

    # load model and use values as Faster RCNN paper suggests
    model = DetectionModel(num_classes=n_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)


    device = torch.device("cuda")
    model = model.to(device)

    best_iou = 0.0

    print('model started training')

    for epoch in range(args.n_epochs):

        train_model(model, train_loader, optimizer, device, epoch, writer, args.alpha) 

        avg_iou = eval_model(model, eval_loader, device, epoch, writer)

    #save the best and last model weights to disk 
    if avg_iou > best_iou:
        best_iou = avg_iou
        save_path =  os.path.join(model_weights_path,str(args.weights_name+'_best_'+str(last_run)+'.pth'))
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path} with avg IoU of {avg_iou:.4f}")

    save_path =  os.path.join(model_weights_path,str(args.weights_name+'_last_'+str(last_run)+'.pth'))
    torch.save(model.state_dict(), save_path)
    print(f"Last model saved to {save_path} with avg IoU of {avg_iou:.4f}")
   




  
        

  































sys.exit()



def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))



for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch) 

 ## visualize ###
# sample = fd[8]
# print(sample)
# img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
# print(type(img_int))

# plt.imshow(draw_bounding_boxes(
#     img_int, sample[1]['boxes'], [classes[i] for i in sample[1]['labels']], width=4, colors=[(10, 0, 255)], font_size=40,
# ).permute(1, 2, 0))
# plt.show()


model.eval()
torch.cuda.empty_cache()


dataset_path = './final_dataset'
test_dataset = FlickrDataset('eval', dataset_path, transform=get_transforms(False))

img, _ = test_dataset[5]
img_int = torch.tensor(img*255, dtype=torch.uint8)
with torch.no_grad():
    prediction = model([img.to(device)])
    pred = prediction[0]   
    print('prediction', prediction)
    
fig = plt.figure(figsize=(14, 10))
plt.imshow(draw_bounding_boxes(img_int,
    pred['boxes'][pred['scores'] > 0.1],
    [classes[i] for i in pred['labels'][pred['scores'] > 0.1].tolist()], width=4
).permute(1, 2, 0))
plt.show()

   

# def visualize(image, bboxes, category_ids, labels):

#     def visualize_bbox(img, bbox, class_name,thickness=2):

#                     BOX_COLOR = (255, 0, 0) # Red
#                     TEXT_COLOR = (255, 255, 255) # White

#                     x_min, y_min, w, h = bbox
#                     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
                
#                     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
                    
#                     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
#                     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#                     cv2.putText(
#                         img,
#                         text=class_name,
#                         org=(x_min, y_min - int(0.3 * text_height)),
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale=0.35, 
#                         color=TEXT_COLOR, 
#                         lineType=cv2.LINE_AA,
#                     )
#                     return img

#     np_image = image.numpy()
#     cv2_image = np.transpose(np_image, (1, 2, 0))
#     img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
#     for bbox_idx,(bbox, category_id) in enumerate(zip(bboxes, category_ids)):
#         class_name = labels[bbox_idx]
#         img = visualize_bbox(img, bbox, class_name)


#     plt.figure(figsize=(12, 12))
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()

# bboxes = sample[1]['boxes'].tolist()
# labels_id = [i.item() for i in sample[1]['labels']]
# labels = [classes[i] for i in sample[1]['labels']]

# print(bboxes,labels_id,  labels)

# visualize(img_int, bboxes, labels_id, labels)


