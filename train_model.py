
from tqdm import tqdm 
import torch
import numpy as np
import math
import pandas as pd
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import sys



# for metric calculation
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)



def train_model(model, train_loader, optimizer, device, epoch, writer, alpha):

    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    all_triplet_losses = []
    
    for images, targets, logo_crops in tqdm(train_loader):

        # for detection
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        # getting logo crops tensors and labels
        logo_labels = []
        crops = []
        for i in range(len(logo_crops)):
            labels = list(logo_crops[i].keys())
            for label in labels:
                crop = logo_crops[i][label]
                for crop_i in crop:
                    logo_labels.append(label)
                    crops.append(crop_i)

        
        logo_labels_t = torch.tensor(logo_labels, dtype=torch.int64) 
        crops_st = torch.stack(crops).to(device)
        
        # getting detection results and embeddings from model
        loss_dict, embeddings = model(images, targets, logo_crops=crops_st)
        embeddings = embeddings.cpu()

        # metric learning with Triplet Loss
        indices_tuple = mining_func(embeddings, logo_labels_t)
        triplet_loss = loss_func(embeddings, logo_labels_t, indices_tuple)
        all_triplet_losses.append(triplet_loss.item())
       

        detection_losses = sum(loss for loss in loss_dict.values())
  
        # append Triplet loss into main loss with a hyperparameter alpha
        losses = detection_losses + alpha*triplet_loss

        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        # stop training when loss become infinity 
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, Stopped Training")
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict)

    # adding losses to Tensorboard 
    total_loss =  np.mean(all_losses)
    loss_classifier = all_losses_dict['loss_classifier'].mean()
    loss_box_reg = all_losses_dict['loss_box_reg'].mean()
    loss_rpn_box_reg = all_losses_dict['loss_rpn_box_reg'].mean()
    loss_objectness = all_losses_dict['loss_objectness'].mean()
    loss_triplets = np.array(all_triplet_losses).mean()

    writer.add_scalar('Training Loss', total_loss, epoch)
    writer.add_scalar('Classifier Loss', loss_classifier, epoch)
    writer.add_scalar('Box Regression Loss', loss_box_reg, epoch)
    writer.add_scalar('RPN Boss Loss', loss_rpn_box_reg, epoch)
    writer.add_scalar('Objectness Loss', loss_objectness, epoch)
    writer.add_scalar('Triplet Loss', loss_triplets, epoch)


    print("Epoch {}, Training lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}, loss_triplets: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], total_loss,
        loss_classifier,
        loss_box_reg,
        loss_rpn_box_reg,
        loss_objectness,
        loss_triplets
    ))

