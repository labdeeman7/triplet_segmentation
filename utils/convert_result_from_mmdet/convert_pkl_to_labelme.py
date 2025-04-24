## Handles conversions from pkl file mmdetection to label files at a threshold. 
from pycocotools import mask
import pickle
import numpy as np
import base64
from os.path import join
from utils.convert_to_coco.convert_dataset_to_instrument_class_coco import instance_mask_to_coco_contours_polygon_xy_or_labelme_format
from utils.general.dataset_variables import TripletSegmentationVariables

INSTRUMENT_ID_TO_INSTRUMENT_CLASS_DICT = TripletSegmentationVariables.categories['instrument']
VERB_ID_TO_INSTRUMENT_CLASS_DICT = TripletSegmentationVariables.categories['verb']
TARGET_ID_TO_INSTRUMENT_CLASS_DICT = TripletSegmentationVariables.categories['target']
TRIPLET_ID_TO_INSTRUMENT_CLASS_DICT = TripletSegmentationVariables.categories['triplet']


from utils.general.save_files import save_to_json
import shutil
import os


def convert_mmdet_rle_to_mask(size: np.ndarray, 
                              counts: bytes):

    input_value = {
        'size': size ,
        'counts': counts
    }

    instance_mask = mask.decode(input_value)
    instance_mask = instance_mask.astype('uint8')  
    contours = instance_mask_to_coco_contours_polygon_xy_or_labelme_format


def get_class_id_instance_id(input_list):
    id_counts = {}
    result = []

    for i, id in enumerate(input_list):
        if id in id_counts:
            id_counts[id] += 1
        else:
            id_counts[id] = 1
        result.append([id, id_counts[id]])

    return result



def copy_folder_contents(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the contents of the source folder
    for item in os.listdir(source_folder):
        # Get the full paths of the source and destination items
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)

        # If it's a file, copy it to the destination folder
        if os.path.isfile(source_item):
            shutil.copy2(source_item, destination_item)
        # If it's a folder, recursively call copy_folder_contents
        elif os.path.isdir(source_item):
            copy_folder_contents(source_item, destination_item)




def convert_predicted_segmentation_to_labelme_shapes_key(single_image_prediction, class_threshold = 0.5):
    # print(single_image_prediction)
    pred_instances = single_image_prediction['pred_instances']
    
    class_labels  = pred_instances['labels'].tolist()
    verb_labels  = pred_instances.get('verb_labels', np.array([])).tolist()
    target_labels  = pred_instances.get('target_labels', np.array([])).tolist()

    scores = pred_instances['scores']
    verb_scores  = pred_instances.get('verb_scores', [])
    target_scores =  pred_instances.get('target_scores', []) 

    
    # print(f'class_labels {class_labels}')
    
    shapes = [] 
    class_id_counts = {}
    for i, class_id in enumerate(class_labels):        
        if scores[i] > class_threshold:  
                
            #get the instance_id
            if class_id in class_id_counts:
                class_id_counts[class_id] +=1
            else: 
                class_id_counts[class_id] = 1    
            
            instance_id = class_id_counts[class_id]  
            

            if len(verb_scores):
                verb_score =  float(verb_scores[i])
                verb_id = verb_labels[i]
                verb_name = VERB_ID_TO_INSTRUMENT_CLASS_DICT[str(min(verb_id+1, 10) )]
            else:
                verb_score = None   
                verb_name = None 
            
            if len(target_scores):
                target_score =  float(target_scores[i])   
                target_id = target_labels[i]
                target_name = TARGET_ID_TO_INSTRUMENT_CLASS_DICT[str(min(target_id+1, 15) )]
            else:
                target_score = None    
                target_name = None     

            class_name = INSTRUMENT_ID_TO_INSTRUMENT_CLASS_DICT[str(class_id+1)]
            
            instance_masks_rle_size_counts_dict = pred_instances['masks'][i]
            
            instance_mask = mask.decode(instance_masks_rle_size_counts_dict)
            instance_mask = instance_mask.astype('uint8')  
            contour_points_for_an_instance = instance_mask_to_coco_contours_polygon_xy_or_labelme_format(instance_mask, 'labelme')      
            
            for contour_points in contour_points_for_an_instance:     
                
                     
                single_instance_contours_info_dict = {
                    "label": class_name,
                    "points": contour_points,
                    "group_id": int(instance_id),
                    "verb": verb_name,
                    "target": target_name,
                    "description": "",
                    "shape_type": "polygon", 
                    "flags": None, 
                    "score": float(scores[i]),
                    "verb_score":  verb_score,
                    "target_score": target_score     
                }
                            
                shapes.append(single_instance_contours_info_dict) 
               
    return shapes 

def convert_predicted_segmentation_to_labelme_shapes_key_when_predicted_directly(single_image_prediction, class_threshold = 0.5):
    pred_instances = single_image_prediction['pred_instances']
    ivt_class_labels  = pred_instances['labels'].tolist()
    scores = pred_instances['scores']
        
    shapes = [] 
    class_name_counts = {}
    for i, ivt_class_id in enumerate(ivt_class_labels):        
        if scores[i] > class_threshold:  
            
            triplet_name = TRIPLET_ID_TO_INSTRUMENT_CLASS_DICT[str(ivt_class_id+1)]
            class_name, verb_name, target_name =  triplet_name.split(',')
            
                
            #get the instance_id
            if class_name in class_name_counts:
                class_name_counts[class_name] +=1
            else: 
                class_name_counts[class_name] = 1    
            
            instance_id = class_name_counts[class_name]  
           
            instance_masks_rle_size_counts_dict = pred_instances['masks'][i]
            
            instance_mask = mask.decode(instance_masks_rle_size_counts_dict)
            instance_mask = instance_mask.astype('uint8')  
            contour_points_for_an_instance = instance_mask_to_coco_contours_polygon_xy_or_labelme_format(instance_mask, 'labelme')      
            
            for contour_points in contour_points_for_an_instance:     
                     
                single_instance_contours_info_dict = {
                    "label": class_name,
                    "points": contour_points,
                    "group_id": int(instance_id),
                    "verb": verb_name,
                    "target": target_name,
                    "description": "",
                    "shape_type": "polygon", 
                    "flags": None, 
                    "score": float(scores[i]),
                    "verb_score":  float(scores[i]),
                    "target_score": float(scores[i])     
                }
                            
                shapes.append(single_instance_contours_info_dict) 
               
    return shapes 


def convert_pickle_to_labelme_json(save_ann_dir,
                                    pickle_file_path,
                                    ivt_predicted_directly = False, 
                                    with_pretrained_model = False,
                                    class_threshold = 0.5):
    if with_pretrained_model:
        global INSTRUMENT_ID_TO_INSTRUMENT_CLASS_DICT
        INSTRUMENT_ID_TO_INSTRUMENT_CLASS_DICT = TripletSegmentationVariables.categories['instrument_direct_pred']
        
    if not os.path.exists(save_ann_dir):
        os.makedirs(save_ann_dir)
    
    print('started opening pickle')
    with open(pickle_file_path, 'rb') as file:
        # Load the object from the file
        mmdet_results = pickle.load(file)
    
    print('finished opening pickle')
    for single_image_prediction in mmdet_results:
        img_file_name = os.path.basename(single_image_prediction['img_path'])
        print(img_file_name)
        ann_file_name = img_file_name.split('.')[0] + '.json'
        if ivt_predicted_directly:
            shapes = convert_predicted_segmentation_to_labelme_shapes_key_when_predicted_directly(single_image_prediction, class_threshold) 
        else:
            shapes = convert_predicted_segmentation_to_labelme_shapes_key(single_image_prediction, class_threshold)        

        # make the corresponding_label_me dict 
        labelme_ann_dict = {
            "version": "1.0.0", 
            "flags": {}, 
            "shapes": shapes,
            "imagePath": img_file_name         
        }
        
        print(f'saved at {join(save_ann_dir, ann_file_name)}') 
        save_to_json(labelme_ann_dict, join(save_ann_dir, ann_file_name))    
    
        
