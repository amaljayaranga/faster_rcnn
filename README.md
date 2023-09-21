  # Faster RCNN Object Detection 📝  
  This Faster RCNN was trained on Flickr27 Logo dataset with added metric learning to improve class classifications.

  
  ## Data Preprocessing  | data_preprocessor.py 🚀
  The  original data of the Flickr27 was collected and converted into COCO data annotations for train and val folders.
  Query data were used to create test folder.

  ## Model Architecture | detection_model.py 🔥  
  The model architecture contains of three components.
  1. Backbone - Pretrained Resnet50 - To get feature vectors
  2. Anchor Generator - to generate different anchor boxes
  3. ROI Pooling - to create bboxes predictions

  The embeddings for logo crops also generated from the same backbone.
  The Triplet Loss is used to calculate metric distances and added to the main Loss of the model to improve model class classifications.
      
  ## Dataset Class | flickr_dataset.py ✨ 
  A custom dataset class inherited from torch.utils.data.Dataset.
  The dataset class has added feature to get logo crops according to annotations along with logo labels.
  Theses additional data is passed to the dataloader.

  ## Train Model | train_model.py
  A training loop of the model is defined here with augmentations.

  ## Eval Model | eval_model.py
  A evaluation loop of the model is defined and Iou metrics are calculated to compare the model performances.

  ## Main Code | main.py
  Make the main training and evaluation of the model.

  ## Inference | inference.py
  Make the inferences from the saved model weights and do Test Time Augmentations.