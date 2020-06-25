import shutil
import os
import pandas as pd
from shutil import copyfile

def clean_dataset_by_brand(class_name):
    if any([brand in class_name.lower() for brand in ['acura', 'buick', 'daewoo', 'geo_metro', 'gmc', 'isuzu', 'lincoln', 'plymouth', 'ram', 'scion', 'spyker', 'eagle', 'fisker', 'hummer']]):
        
        return False

    return True

def clean_dataset_by_year(class_name):
    first_year = 1950
    if any([str(year) in class_name for year in range(first_year, first_year + 50)]):

        return False

    return True

def copy_images():
    parent_dir = os.path.dirname(os.getcwd())
    root_path = os.path.join(parent_dir, 'car_ims')
    dest_folder = os.path.join(parent_dir, 'car_ims_mod')

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    labels = pd.read_csv(os.path.join(parent_dir, 'car_ims_labels.csv'))
    class_names = pd.read_csv(os.path.join(parent_dir, "class_names_id.csv"))

    for index, row in labels.iterrows():
        filename = row['filename']
        class_id = row['class_id']
        class_name = class_names.loc[class_names['Id'] == class_id].values[0][0]
        print(filename, class_name)

        if clean_dataset_by_brand(class_name) and clean_dataset_by_year(class_name):
            copyfile(os.path.join(root_path, filename), os.path.join(dest_folder, filename))

def move_images_to_folder():
    parent_dir = os.path.dirname(os.getcwd())
    root_path = os.path.join(parent_dir, 'car_ims_mod')
    dest_folder = os.path.join(parent_dir, 'car_by_class')
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    class_names = pd.read_csv(os.path.join(parent_dir, 'class_names_id.csv'))

    for folder in ['train_modified', 'test_modified']:
        labels = pd.read_csv(os.path.join(parent_dir, folder + '_labels.csv'))
        images_folder = os.path.join(parent_dir, folder)
        for index, row in labels.iterrows():
            filename = row['filename']
            class_id = row['class_id']
            class_name = class_names.loc[class_names['Id'] == class_id].values[0][0]
            print(filename, class_name)
            folder_name = '_'.join(str(word).lower() for word in class_name.split(' '))
            class_folder = os.path.join(dest_folder, folder, folder_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            
            copyfile(os.path.join(images_folder, filename), os.path.join(class_folder, filename))


if __name__ == '__main__':
    move_images_to_folder()
