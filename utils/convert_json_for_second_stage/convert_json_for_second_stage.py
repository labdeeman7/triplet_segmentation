import json
import numpy as np
import cv2
import os
from collections import defaultdict
from os.path import join

def convert_json_ann_to_second_stage_ann_for_single_json_file(json_ann_path, 
                                                              img_dir,
                                                              output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the JSON file
    with open(json_ann_path, 'r') as f:
        data = json.load(f)

    # Load the image dimensions
    img_name = f'{os.path.basename(json_ann_path).split('.')[0]}.png' 
    img = cv2.imread(join(img_dir,img_name ))
    # print(img.shape)
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Group contours by class name and group id
    grouped_shapes = defaultdict(list)
    for shape in data['shapes']:
        key = (shape['label'], shape['group_id'])
        contour_info = {
            'points': shape['points'],
            'verb': shape['verb'],
            'target': shape['target']
        }
        grouped_shapes[key].append(contour_info)

    # Generate binary masks for each instance
    for (label, group_id), contour_infos in grouped_shapes.items():
        
        # Create a blank binary mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # get verb and target, they all the same anyway.  
        verb_name = contour_infos[0]['verb']  
        target_name =  contour_infos[0]['target']
        
        for contour_info in contour_infos:
            points = contour_info['points']    
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)

        # Construct the filename
        base_image_name = os.path.basename(json_ann_path).split('.')[0]
        filename = f"{base_image_name},{label},{group_id},{verb_name},{target_name}.png"

        # Save the binary mask
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, mask)

    print(f"Binary masks have been saved to {output_folder}.")


def convert_json_ann_to_second_stage_ann_for_a_dataset_split(dataset_split_json_ann_dir, 
                                                             dataset_split_img_dir,
                                                             output_folder):
    json_ann_paths  = [join(dataset_split_json_ann_dir, json_ann_name) 
                       for json_ann_name in os.listdir(dataset_split_json_ann_dir)   ] 
    
    for json_ann_path in json_ann_paths:
        convert_json_ann_to_second_stage_ann_for_single_json_file(json_ann_path, 
                                                              dataset_split_img_dir,
                                                              output_folder)
        
    