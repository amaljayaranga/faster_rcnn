
from tqdm import tqdm 
import torch
import numpy as np
from torchvision.ops import box_iou, nms

def calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.5, nms_iou_threshold=0.5):

    TP = 0
    FP = 0
    FN = 0
    total_iou =  0.0

    # If there are no predictions and no ground truths, return
    if len(pred_boxes) == 0 and len(true_boxes) == 0:
        return TP, FP, FN, 0.0 
    
    # If there are no predictions, all ground truths are False Negatives
    if len(pred_boxes) == 0:
        return TP, FP, len(true_boxes), 0.0
    
    # If there are no ground truths, all predictions are False Positives
    if len(true_boxes) == 0:
        return TP, len(pred_boxes), FN, 0.0



    # Apply NMS based on classes on prediction boxes
    unique_classes = pred_labels.unique()
    final_boxes, final_labels, final_scores = [], [], []
    for cls in unique_classes:
        indices = (pred_labels == cls)
        cls_boxes = pred_boxes[indices]
        cls_scores = pred_scores[indices]
        
        kept_indices = nms(cls_boxes, cls_scores, nms_iou_threshold)
        final_boxes.append(cls_boxes[kept_indices])
        final_labels.append(pred_labels[indices][kept_indices])
        final_scores.append(cls_scores[kept_indices])

    pred_boxes = torch.cat(final_boxes, dim=0)
    pred_labels = torch.cat(final_labels, dim=0)
    pred_scores = torch.cat(final_scores, dim=0)

    ious = box_iou(pred_boxes, true_boxes)

    # For each predicted box, find the ground truth box with the highest IoU
    for i in range(ious.shape[0]):
        max_iou, idx = ious[i].max(0)
        
        # If the maximum IoU is above the threshold and labels match
        if max_iou >= iou_threshold and pred_labels[i] == true_labels[idx]:
            TP += 1
            total_iou += max_iou
        else:
            FP += 1

    # Calculate FN
    FN = len(true_boxes) - TP

    # Calculate average IoU
    avg_iou = total_iou / (TP + FP) if TP + FP > 0 else 0.0

    return TP, FP, FN, avg_iou


def eval_model(model, eval_loader, device, epoch, writer):

    model.to(device)
    model.eval()

    total_TP = []
    total_FP = []
    total_FN = []
    total_ious = []
 
    with torch.no_grad():
        for images, targets, _ in tqdm(eval_loader):

            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, target in zip(predictions, targets):

                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                pred_scores = pred['scores']

                true_boxes = target['boxes']
                true_labels = target['labels']

                TP, FP, FN, avg_iou = calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels)
              
                total_TP.append(TP) 
                total_FP.append(FP)
                total_FN.append(FN)
                avg_iou = avg_iou.cpu().numpy() if not isinstance(avg_iou, float) else avg_iou
                total_ious.append(avg_iou)


    epoch_iou = np.mean(total_ious)
    precision = sum(total_TP) / (sum(total_TP) + sum(total_FP)) if (sum(total_TP) + sum(total_FP)) != 0 else 0
    recall = sum(total_TP) / (sum(total_TP) + sum(total_FN)) if (sum(total_TP) + sum(total_FN)) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("Epoch {},  Evaluation IoU: {:.6f},  Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f} ".format(epoch, epoch_iou, precision, recall, f1))
    writer.add_scalar('Evaluation IoU', epoch_iou, epoch)
    writer.add_scalar('Precision', precision, epoch)
    writer.add_scalar('Recall', recall, epoch)
    writer.add_scalar('F1', f1, epoch)

    return  epoch_iou
