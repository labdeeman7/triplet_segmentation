
from os.path import join
import os
import json
import numpy as np
from utils.general.read_files import read_from_json 

from utils.convert_to_coco.convert_dataset_to_instrument_class_coco import get_bbox_info_from_coco_contour_xy
from utils.general.dataset_variables import TripletSegmentationVariables


def get_list_of_info_dict_for_metric_calculation_in_img(json_dict: dict):
    
    INSTRUMENT_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['instrument']
    INSTRUMENT_CLASS_TO_ID_DICT = {instrument_class: instrument_id for instrument_id, instrument_class in INSTRUMENT_ID_TO_CLASS_DICT.items()}
    TRIPLET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['triplet']
    TRIPLET_CLASS_TO_ID_DICT = {triplet_class: triplet_id for triplet_id, triplet_class in TRIPLET_ID_TO_CLASS_DICT.items()}
    IMGWIDTH = TripletSegmentationVariables.width
    IMGHEIGHT = TripletSegmentationVariables.height
    
    refactored_json_dict = {}
    for contour_info in json_dict['shapes']:
        class_name = contour_info['label']
        instance_id =  contour_info['group_id']
        contour_info = {
            "contour_pts" : contour_info['points'], 
            "score": contour_info.get('score', 1.0),
            "verb": contour_info['verb'],
            "verb_score": contour_info.get('verb_score', 1.0),
            "target": contour_info['target'],
            "target_score": contour_info.get('target_score', 1.0)
        }
        
        
        if((class_name, instance_id) in refactored_json_dict): 
            refactored_json_dict[(class_name, instance_id)].append(contour_info)    
        else:
            refactored_json_dict[(class_name, instance_id)] =  [contour_info] 
        
       
    list_of_info_dict_for_metric_calculation_in_img = [] 
    for (class_name, instance_id), contour_info_for_class_and_instance_id_list in refactored_json_dict.items():
        
        
        score = contour_info_for_class_and_instance_id_list[0]['score']
        verb_score = contour_info_for_class_and_instance_id_list[0]['verb_score']
        verb = contour_info_for_class_and_instance_id_list[0]['verb']
        target = contour_info_for_class_and_instance_id_list[0]['target']
        target_score = contour_info_for_class_and_instance_id_list[0]['target_score']
        
        triplet_name=f'{class_name},{verb},{target}'
        
        
        # convert to polygon_xy and use list this is an issue with labelme for getting bounding box information      
        contour_points_for_an_instance_polygon_xy= [] 
        for contour_info_for_class_and_instance_id in contour_info_for_class_and_instance_id_list:
            contour_pts =  np.array(contour_info_for_class_and_instance_id["contour_pts"], dtype=np.int32)
            contour_pts = contour_pts.flatten().tolist()
            
            contour_points_for_an_instance_polygon_xy.append(contour_pts)
        
        # get bboxes from contours
        x_min, y_min, width, height = get_bbox_info_from_coco_contour_xy(contour_points_for_an_instance_polygon_xy)
        
        if triplet_name not in TRIPLET_CLASS_TO_ID_DICT.keys():
            triplet_name = f'{class_name},null_verb,null_target'
           
        single_instance_info_dict_for_metric_calculation = {
            'triplet': str(int(TRIPLET_CLASS_TO_ID_DICT[triplet_name])-1), #zero init
            'instrument': [str(int(INSTRUMENT_CLASS_TO_ID_DICT[class_name])-1), score, x_min/IMGWIDTH,
                          y_min/IMGHEIGHT, width/IMGWIDTH, height/IMGHEIGHT],            
            'class_name': class_name,
            'instance_id': instance_id
        }  
            
        list_of_info_dict_for_metric_calculation_in_img.append(single_instance_info_dict_for_metric_calculation) 
    
    return list_of_info_dict_for_metric_calculation_in_img  


def run_ivt_metric_object_for_folder(ivt_metric_object,
                                    pred_ann_dir, 
                                    gt_ann_dir):  

    pred_ann_list = sorted(os.listdir(pred_ann_dir))

    for i, filename in enumerate(pred_ann_list):    
        pred_ann_path = join(pred_ann_dir, filename)
        gt_ann_path =  join(gt_ann_dir, filename)     
        print(f'currently on {i}, {filename}')  
        
        
        json_dict_pred =  read_from_json(pred_ann_path)
        list_of_info_dict_for_metric_calculation_in_img_pred = get_list_of_info_dict_for_metric_calculation_in_img(json_dict_pred) 
        
        json_dict_gt =  read_from_json(gt_ann_path)
        list_of_info_dict_for_metric_calculation_in_img_gt = get_list_of_info_dict_for_metric_calculation_in_img(json_dict_gt)  
        
        print(f'pred, {list_of_info_dict_for_metric_calculation_in_img_pred}')
        print(f'gt, {list_of_info_dict_for_metric_calculation_in_img_gt}')
        

        # print(list_of_info_dict_for_metric_calculation_in_img_gt)
        
        #detect.update(labels, predictions, format=format)
        ivt_metric_object.update(targets=[list_of_info_dict_for_metric_calculation_in_img_gt],
                    predictions=[list_of_info_dict_for_metric_calculation_in_img_pred], 
                    format="dict")
        