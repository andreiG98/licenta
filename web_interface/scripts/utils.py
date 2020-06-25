import os
import json
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
from sys import exit

from torchvision.ops import nms
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision

from models.scripts import construct_model_detection, construct_model_classification


def transform_image(image, img_size):
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    inference_transform = transforms.Compose([transforms.Resize(img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean_nums, std_nums)])

    return inference_transform(image).unsqueeze(0)

CATEGORY_NAMES = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_category_names(path):
    global CATEGORY_NAMES
    
    class_list_path = '/'.join(path.split('/')[:-2])
    class_names_json = os.path.join(class_list_path, 'class_names_new.json')
    
    with open(class_names_json) as json_file:
        CATEGORY_NAMES = json.load(json_file)

def load_model_detection(path):
    print('Loading model for car detection....')

    with open(os.path.join(path, 'config.json')) as json_file:
        config = json.load(json_file)
        
    best_path = os.path.join(path, 'best.pth')
    checkpoint = torch.load(best_path)
    model_state_dict = checkpoint['model']

    model_od = construct_model_detection(config)
    model_od.load_state_dict(model_state_dict)
    model_od.to(device)
    model_od.eval()

    return model_od, config['imgsize']
    
def load_model_classification(path):
    print('Loading model for car classification....')
    get_category_names(path) # Load category names

    with open(os.path.join(path, 'config.json')) as json_file:
        config = json.load(json_file)
        
    best_path = os.path.join(path, 'best.pth')
    checkpoint = torch.load(best_path)
    model_state_dict = checkpoint['model']

    num_classes = len(CATEGORY_NAMES)
    model_cl = construct_model_classification(config, num_classes)
    model_cl.load_state_dict(model_state_dict)
    model_cl.to(device)
    model_cl.eval()

    return model_cl, config['imgsize']

@torch.no_grad()
def get_prediction_detection(model_od, img_size_od, img_path, threshold=0.6):
    CATEGORY_NAMES_CAR = ['__background__', 'Car']
    
    image = Image.open(img_path) # Load the image
    original_width, original_height = image.size

    img_size = img_size_od
    print('detection at {}px'.format(img_size))
        
    image = transform_image(image, img_size=img_size)
    # Move to default device
    image = image.to(device)

    print('forward model detection')
    pred = model_od(image) # Pass the image to the model

    pred_class = [CATEGORY_NAMES_CAR[i] for i in list(pred[0]['labels'])] # Get the Prediction Class
    pred_boxes = [[box[0] / img_size[0] * original_width, box[1] / img_size[1] * original_height, box[2] / img_size[0] * original_width, box[3] / img_size[1] * original_height]
                    for box in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.

    pred_boxes = [pred_boxes[t] for t in pred_t]
    pred_class = [pred_class[t] for t in pred_t]
    pred_score = [pred_score[t] for t in pred_t]

    keep = nms(boxes=torch.FloatTensor(pred_boxes), scores=torch.FloatTensor(pred_score), iou_threshold=0.1)
    keep_boxes = [pred_boxes[i] for i in keep]
    keep_class = [pred_class[i] for i in keep]
    keep_score = [pred_score[i] for i in keep]

    pred_class = []

    return keep_boxes, keep_class, keep_score

def compute_heatmap(model, pred, pred_class_id, img_pil, img_original):
    # get the gradient of the output with respect to the parameters of the model
    pred[:, pred_class_id].backward(retain_graph=True)

    # pull the gradients out of the model
    gradients = model.get_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(img_pil).detach()

    # weight the channels by corresponding gradients
    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()

    img_np = np.array(img_original)
    heatmap = cv.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_np

    return superimposed_img

# @torch.no_grad()
def get_prediction_classification(model_cl, img_size_cl, img):
    img_size = img_size_cl
    print('classification at {}px'.format(img_size))
    
    img_original = img
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    image = transform_image(image, img_size=img_size)
    # Move to default device
    image = image.to(device)
    
    print('forward model classification')
    pred = model_cl(image) # Pass the image to the model
    
    probabilities = nn.Softmax(dim=1)(pred)
    scores, ids = torch.topk(probabilities, 5)

    scores, ids = scores.cpu(), ids.cpu()

    # heatmaps = [compute_heatmap(model_cl, pred, ids[0][idx], image, img_original) for idx in range(len(ids[0]))]
    heatmaps = [compute_heatmap(model_cl, pred, ids[0][0], image, img_original)]
    
    class_names = [CATEGORY_NAMES[str(label_id.item())] for label_id in ids[0]]
    scores = [str(round(float(score.item() * 100), 4)) for score in scores[0]]

    class_names_upper = []
    for class_name in class_names:
        class_names_upper.append(' '.join([word.upper() for word in class_name.split('_')]))
    

    classification_prediction = {}
    classification_prediction['class_names'] = class_names_upper
    classification_prediction['scores'] = scores
    classification_prediction['heatmaps'] = heatmaps

    return classification_prediction

def object_detection_api(model_od, model_cl, img_size_od, img_size_cl, img_path, predicted_path, threshold=0.6, rect_th=2, text_size=0.5, text_th=2):
    print('object detection api')
    boxes, pred_cls, pred_score = get_prediction_detection(model_od, img_size_od, img_path, threshold) # Get predictions

    img = cv.imread(img_path) # Read image with openCV

    folder_name = predicted_path.split('/')[-1]
    keep_class = []
    predicted_path_images = []

    os.makedirs(os.path.join(predicted_path, 'heatmaps'))

    font = cv.FONT_HERSHEY_SIMPLEX
    rectangle_bgr = (0, 0, 0)
    alpha = 0.6  # Transparency factor.
    original_overlay = img.copy()
    image_new = img.copy()

    filename = img_path.split('/')[-1]
    
    for i in range(len(boxes)):
        xmin = int(boxes[i][0])
        ymin = int(boxes[i][1])
        xmax = int(boxes[i][2])
        ymax = int(boxes[i][3])
        
        if i == 0:
            keep_class.append(('', -1))
        
        cv.rectangle(image_new, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        box_width, box_height = (xmax - xmin), (ymax - ymin)
        
    cv.imwrite(os.path.join(predicted_path, filename), image_new)
    predicted_path_images.append(os.path.join(folder_name, filename))

    for i in range(len(boxes)):
        xmin = int(boxes[i][0])
        ymin = int(boxes[i][1])
        xmax = int(boxes[i][2])
        ymax = int(boxes[i][3])
        
        img_cropped = img[ymin: ymax, xmin: xmax]

        classification_prediction = get_prediction_classification(model_cl, img_size_cl, img_cropped)
        cropped_img_name = str(i) + '_' + filename
        cv.imwrite(os.path.join(predicted_path, cropped_img_name), img_cropped)
        predicted_path_images.append(os.path.join(folder_name, cropped_img_name))

        heatmap_name = str(i) + '_heatmap_' + filename
        cv.imwrite(os.path.join(predicted_path, 'heatmaps', heatmap_name), classification_prediction['heatmaps'][0])
        predicted_path_images.append(os.path.join(folder_name, 'heatmaps', heatmap_name))

        keep_class.append((classification_prediction['class_names'], classification_prediction['scores']))
        keep_class.append((classification_prediction['class_names'], classification_prediction['scores']))

        # current_heatmap = []
        # for idx, heatmap in enumerate(classification_prediction['heatmaps']):
        #     heatmap_name = 'heatmap_' + str(idx) + '_' + cropped_img_name
        #     cv.imwrite(os.path.join(predicted_path, 'heatmaps', heatmap_name), heatmap)
        #     current_heatmap.append(os.path.join(folder_name, 'heatmaps', heatmap_name))
        # heatmaps_path.append(current_heatmap)
            
    return predicted_path_images, keep_class