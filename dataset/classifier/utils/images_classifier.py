import scipy.io as sio
import numpy as np
import os
import shutil
import cv2 as cv

parent_dir = os.path.dirname(os.getcwd())
cars_class_dir = os.path.join(parent_dir, 'car_by_class')
cars_class_classifier_dir = os.path.join(parent_dir, 'car_by_class_classifier')

def mat_opener():
    cars_annos = sio.loadmat(os.path.join(parent_dir, 'all_cars_annos.mat'))
    cars_annos = cars_annos['annotations']
    list_annos = [np.array(anno.tolist()) for anno in cars_annos]
    list_annos = list_annos[0]

    return list_annos

def crop_images():
    list_annos = mat_opener()

    for folder in os.listdir(cars_class_dir):
        for img in os.listdir(os.path.join(cars_class_dir, folder)):
            for anno in list_annos:
                img_name = anno[0][0].split('/')[-1]
                if img == img_name:
                    x_min = anno[1][0]
                    y_min = anno[2][0]
                    x_max = anno[3][0]
                    y_max = anno[4][0]
                    class_dir = os.path.join(cars_class_classifier_dir, folder)
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    img_cv = cv.imread(os.path.join(cars_class_dir, folder, img))
                    crop_img = img_cv[y_min: y_max, x_min: x_max]
                    cv.imwrite(os.path.join(class_dir, img), crop_img)
                    print(img)
                    break

crop_images()