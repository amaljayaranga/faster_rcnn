import pandas as pd
import os
import sys
import cv2 
import json
import random
import shutil
from datetime import datetime
import argparse


class DataPreProcessor:
    def __init__(self, source_dir, training_annotation_file, query_annotation_file, source_images, dest_dir):
        super().__init__()
        self.source_dir = source_dir
        self.source_images = source_images
        self.dest_dir = dest_dir
        self.FOLDER_NAMES = ['train', 'eval']
        self.TEST_FOLDER = 'test'
        self.SPLIT_TRAIN_EVAL = [0.7, 0.3]
        self.coco_json_name = 'coco_annotations.json'
        self.SEED = 22
        self.source_imgs_dir = os.path.join(self.source_dir, self.source_images)
        self.train_annotation_path = os.path.join(self.source_dir, training_annotation_file)
        self.query_annotation_path = os.path.join(self.source_dir, query_annotation_file)

        self.unique_brands = []
        self.unique_images = []
        self.all_annos = {}
    
    def collect_all_annotations(self):

        if not os.path.exists(self.train_annotation_path):
            print('Training Annotation file does not exist')
            sys.exit()

        with open(self.train_annotation_path, 'r') as annotations:
            annos = annotations.readlines()

            all_brands = []
            all_images = []

            for anno in annos:
                img_name, brand, tr_num, x_min, y_min, x_max, y_max = anno.rstrip().split(' ')

                # arrange to COCO format [x_min, y_min, width, height] 
                x_min = int(x_min)
                y_min = int(y_min)
                width = int(x_max)-int(x_min)
                height = int(y_max)-int(y_min)
                anno_item = [brand, x_min, y_min, width, height]

                # remove wrong bboxes
                if int(x_max) > int(x_min) and int(y_max) > int(y_min):

                    if img_name not in list(self.all_annos.keys()):
                        self.all_annos[img_name] = [anno_item]
                    else:
                        current_list = self.all_annos.get(img_name)
                        if anno_item not in current_list:
                            current_list.extend([anno_item])
                            self.all_annos[img_name] = current_list

                    all_brands.append(brand)
                    all_images.append(img_name)

            self.unique_brands = list(set(all_brands))
            self.unique_images = list(set(all_images))
        

    def create_dataset_structure(self):

        # create dataset structure with train, eval
        for folder_name in self.FOLDER_NAMES:
            if os.path.exists(os.path.join(self.dest_dir,folder_name)):
                shutil.rmtree(os.path.join(self.dest_dir,folder_name))
            os.makedirs(os.path.join(self.dest_dir,folder_name)) 

    def split_annotations(self):

        random.seed(self.SEED)
        random.shuffle(self.unique_images)

        # split dataset into train, eval along with annotations

        train_indexes = int(len(self.unique_images)*self.SPLIT_TRAIN_EVAL[0])
      
        train_images = self.unique_images[0:train_indexes]

        eval_images = self.unique_images[train_indexes:]

    
        return train_images, eval_images
    

    def copy_images(self,images, folder):
        for image in images:
            shutil.copyfile(os.path.join(self.source_imgs_dir, image), os.path.join(self.dest_dir,folder,image))


    def generate_coco_json(self,images, folder_name):

        coco = {}

        # adding information
        coco['info'] = {
            'year' : 2023,
            'version' : 1,
            'description' : 'Flickr 27 Logos Dataset',
            'contributor' : 'Amal Jayaranga',
            'date' : str(datetime.now()),
        }


        # adding classes
        coco['categories'] = [{'id':brand_idx, 'name': brand} for brand_idx, brand in enumerate(self.unique_brands)]

        # adding images and annotations
        coco['images'] = []
        coco['annotations'] = []

        anno_id = 0
        
        for img_id, img_name in enumerate(images):
            img = cv2.imread(os.path.join(self.source_imgs_dir,img_name))
            coco['images'].append({'id' : img_id, 'file_name':img_name, 'height':img.shape[0], 'width':img.shape[1]})


            annotations_per_img = self.all_annos.get(img_name)
            for anno_per_img in annotations_per_img:
                category_id = self.unique_brands.index(anno_per_img[0])
                bbox = [anno_per_img[1], anno_per_img[2], anno_per_img[3], anno_per_img[4]]
                area = anno_per_img[3]*anno_per_img[4]
                coco['annotations'].append({'id':anno_id, 'image_id':img_id, 'category_id':category_id, 'bbox':bbox, 'area':area, 'iscrowd':0})
                anno_id += 1

         # writing coco json
        json_coco = json.dumps(coco, indent=4)
        with open(os.path.join(self.dest_dir, folder_name, folder_name+'_'+self.coco_json_name), "w") as outfile:
            outfile.write(json_coco)

        print('{} coco json file written successfully.'.format(folder_name))


    def create_test_dataset(self):

        if not os.path.exists(self.query_annotation_path):
            print('Query Annotation file does not exist')
            sys.exit()

        if os.path.exists(os.path.join(self.dest_dir,self.TEST_FOLDER)):
            shutil.rmtree(os.path.join(self.dest_dir,self.TEST_FOLDER))
        os.makedirs(os.path.join(self.dest_dir,self.TEST_FOLDER)) 

        with open(self.query_annotation_path, 'r') as annotations:
            annos = annotations.readlines()

            for anno in annos:
                img_name, brand = anno.split('\t')
                shutil.copyfile(os.path.join(self.source_imgs_dir, img_name), os.path.join(self.dest_dir,self.TEST_FOLDER,img_name))
        
        print('Test images are copied successfully ')



    def forward(self):
        self.collect_all_annotations()
        self.create_dataset_structure()

        train_images, eval_images= self.split_annotations()
    
        for images, folder in zip([train_images, eval_images],self.FOLDER_NAMES):
            self.copy_images(images, folder)
            self.generate_coco_json(images, folder)

        self.create_test_dataset()


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Pre Processing Data and Create Required Datatsets with COCO Format')
  parser.add_argument('--original_dataset_path', dest='original_dataset_path',
                      help='the path of original Flickr dataset',
                      default='flicker27_dataset', type=str)
  
  parser.add_argument('--train_anno_file', dest='train_anno_file',
                      help='Train annotations file',
                      default='training_set_annotation.txt', type=str)
  
  parser.add_argument('--query_anno_file', dest='query_anno_file',
                      help='Query annotations file',
                      default='query_set_annotation.txt', type=str)
  
  parser.add_argument('--original_images_folder', dest='original_images_folder',
                      help='the path of original Flickr dataset images',
                      default='images', type=str)
  
  parser.add_argument('--processed_dataset_folder', dest='processed_dataset_folder',
                      help='Final processed dataset folder',
                      default='final_dataset', type=str)
  args = parser.parse_args()
  return args
            


if __name__ == '__main__':

    args = parse_args()

    data_preprocessor = DataPreProcessor(args.original_dataset_path, 
                                         args.train_anno_file,
                                         args.query_anno_file,
                                         args.original_images_folder,
                                         args.processed_dataset_folder )
    data_preprocessor.forward()

            

        









    

    


    


    

 



    

    
      
   

    
   