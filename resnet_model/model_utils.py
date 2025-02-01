from utils.general.dataset_variables import TripletSegmentationVariables
import numpy as np
from utils.general.save_files import save_to_json
from utils.general.read_files import read_from_json

def get_dataset_label_ids(config, _SurgicalDataset, ):   
    if hasattr(config, "label_id_json_file_path"):
        label_ids = read_from_json(config.label_id_json_file_path)
    else:
        print(f'Getting dataset train label ids {config.task_name}')      
        train_dataset_for_counting = _SurgicalDataset(config, config.train_image_dir, config.train_ann_dir, None, train_mode=False) #For just counting class and samples available. 
        label_ids = []
        if train_dataset_for_counting.class_name == 'SurgicalMultitaskDataset':
            _, _, _, _, verb_id, target_id, _ = train_dataset_for_counting[i]
            label_ids.append([verb_id, target_id])
        
        for i in range(len(train_dataset_for_counting)):
                _, _, _, _, task_id, _ = train_dataset_for_counting[i]  # Extract task_id
                label_ids.append(task_id)

    
        print(f'length of dataset train is {len(label_ids)}')
        save_to_json(label_ids, f'{config.dataset_path}/label_ids/label_ids_{config.task_name}_train_v3.json')
        raise ValueError('stop here')
        
  
    return label_ids


def get_verbtarget_to_verb_and_target_matrix(): 
    '''
    Generate a num_verbsxnum_verbtargets and a num_targetsxnum_verbtargets
    '''
    # Define the verbtarget and verb dictionary
    verbtargets = TripletSegmentationVariables.categories['verbtarget']
    verbs = TripletSegmentationVariables.categories['verb']
    targets =  TripletSegmentationVariables.categories['target']


    # Create a mapping of verb names to indices
    verb_to_index = {verb: int(idx) - 1 for idx, verb in verbs.items()}
    target_to_index = {target: int(idx) - 1 for idx, target in targets.items()}

    num_verbs = len(verbs)
    num_targets = len(targets)
    num_verbtargets = len(verbtargets)

    # Initialize a 10x56 matrix with zeros and 15x56 with zeros.
    verbtarget_to_verb_matrix = np.zeros((num_verbs, num_verbtargets), dtype=np.int32)
    verbtarget_to_target_matrix = np.zeros((num_targets, num_verbtargets), dtype=np.int32)

    # Populate the matrix
    for vt_idx, vt in verbtargets.items():
        vt_idx = int(vt_idx) - 1  # Convert to 0-based index
        verb, target = vt.split(',')  # Extract the verb and target    
        verb_idx = verb_to_index[verb]  # Get the verb index
        target_idx = target_to_index[target]  # Get the taget index
        
        verbtarget_to_verb_matrix[verb_idx, vt_idx] = 1 # essentially in a row, we are only adding 1 if it exists. 
        verbtarget_to_target_matrix[target_idx, vt_idx] = 1

    return verbtarget_to_verb_matrix, verbtarget_to_target_matrix