
import torch
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import argparse

from detection_model import DetectionModel

transform = A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.1),
            ToTensorV2()])


def visualize_all(augmented_images, preds, confidence):

    fig, axs = plt.subplots(2, int(len(augmented_images)/2), figsize=(14, 10), num='TTA Results') 

    for ax, img, pred in zip(axs.ravel(), augmented_images, preds):
        img_with_boxes = draw_bounding_boxes(img,
                                            pred['boxes'][pred['scores'] > confidence],
                                            [classes[i] for i in pred['labels'][pred['scores'] > confidence].tolist()],
                                            width=4).permute(1, 2, 0)
        ax.imshow(img_with_boxes)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def tta_inference(model, img_path, n_augmentations=6, confidence=0.2):

    image = cv2.imread(img_path)
    
    boxes_list = []
    scores_list = []
    labels_list = []

    augmented_images = []
    preds = []

    for _ in range(n_augmentations):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = transform(image=image)
        augmented_image = augmented["image"]
        img_int = torch.tensor(augmented_image, dtype=torch.uint8)
      
        augmented_image = augmented["image"].unsqueeze(0)/255
        augmented_image = augmented_image.to(device)

        pred = model(augmented_image)

        boxes_list.append(pred[0]['boxes'])
        scores_list.append(pred[0]['scores'])
        labels_list.append(pred[0]['labels'])

        augmented_images.append(img_int)
        preds.append(pred[0])

    visualize_all(augmented_images, preds, confidence)

    # take top 10
    scores_concat = torch.cat(scores_list)
    top_indices = scores_concat.argsort(descending=True)[:10]
    aggregated_predictions = {
        'boxes': torch.cat(boxes_list)[top_indices],
        'scores': scores_concat[top_indices],
        'labels': torch.cat(labels_list)[top_indices]
    }

    return aggregated_predictions

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Inference Model with Test Time Augmentation')
  parser.add_argument('--dataset_path', dest='dataset_path',
                      help='the path of dataset',
                      default='final_dataset', type=str)
  
  parser.add_argument('--model_save_path', dest='model_save_path',
                      help='the path where model weights are saved',
                      default='weights/1/faster_rcnn_best_1.pth', type=str)
  args = parser.parse_args()
  return args


if __name__ == '__main__':

    args = parse_args()

    dataset_path = os.path.join(os.getcwd(),args.dataset_path)
    model_path =  os.path.join(os.getcwd(),args.model_save_path)


    # Load categories from annotations
    coco = COCO(os.path.join(dataset_path, "train", "train_coco_annotations.json"))
    categories = coco.cats
    n_classes = len(categories.keys())
    classes = [i[1]['name'] for i in categories.items()]
    print("Classes", classes)


    # Load saved weights
    model = DetectionModel(num_classes=n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda")
    model.to(device)

    # Load test image 
    img_path = os.path.join(dataset_path, 'test', '344228483.jpg')
    aggregated_predictions = tta_inference(model, img_path)
    #print(aggregated_predictions)


    # good 133298345.jpg
    # good 179101388.jpg
    # good no logo 344228483.jpg



