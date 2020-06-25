from torch.utils.data import Dataset
import pandas as pd
import csv
import os
from PIL import Image
import numpy as np

import torch

class CarModelDatasetVOC(Dataset):
    def __init__(self, root, annotation, class_list, img_size, car_only=False, transforms=None):
        """
        Args:
            annotation (string): CSV file with training annotations
            class_list (string): CSV file with class list
        """
        
        self.root = root
        self.transforms = transforms
        self.img_size = img_size
        self.car_only = car_only
            
        annotations_list = pd.read_csv(annotation)
        self.classes = {}
        self.image_data = {}
        for index, row in annotations_list.iterrows():
            if row.isnull().values.any():
                break
            
            filename = row['filename']
            img_id = int(filename.split('.')[0])
            class_id = row['class_id']
            if class_id not in self.classes:
                self.classes[class_id] = ''
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            if img_id not in self.image_data:
                self.image_data[img_id] = [{'filename': filename}, [{'bbox': [xmin, ymin, xmax, ymax], 'class_id': class_id}]]
            else:
                self.image_data[img_id][1].append({'bbox': [xmin, ymin, xmax, ymax], 'class_id': class_id})
                
        self.image_names = list(self.image_data.keys())
        
        class_list_df = pd.read_csv(class_list)
        for index, row in class_list_df.iterrows():
            class_name = row['Class_Name']
            class_id = row['Id']
            if class_id in self.classes:
                self.classes[class_id] = class_name
                
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
        """
        img_id = self.image_names[index]
        # open the input image
        filename = self.image_data[img_id][0]['filename']
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        original_width, original_height = img.size
        
        # Resize image
        img = img.resize(self.img_size)
        img = np.array(img)
        
        annotations_list = self.image_data[img_id][1]

        # number of objects in the image
        num_objs = len(annotations_list)

        # Bounding boxes for objects
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        # Size of bbox (Rectangular)
        labels = []
        for i in range(num_objs):
            xmin = annotations_list[i]['bbox'][0]
            ymin = annotations_list[i]['bbox'][1]
            xmax = annotations_list[i]['bbox'][2]
            ymax = annotations_list[i]['bbox'][3]
            
            xmin = xmin / original_width * self.img_size[0]
            xmax = xmax / original_width * self.img_size[0]
            ymin = ymin / original_height * self.img_size[1]
            ymax = ymax / original_height * self.img_size[1]
        
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotations_list[i]['class_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img_id = torch.tensor([img_id])     
        # Labels (In my case, I only one class: target class or background)
        if self.car_only:
            labels = torch.ones((num_objs,), dtype=torch.int64)
        else:
            labels = torch.as_tensor(labels, dtype=torch.int64)
        #  suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        target["filename"] = filename
        # target["width"] = original_width
        # target["height"] = original_height
        target["image_path"] = os.path.join(self.root, filename)

        return img, target

    def __len__(self):

        return len(self.image_names)
        
    def name_to_label(self, name):
        
        return self.labels[name]

    def label_to_name(self, label):
        
        return self.classes[label]