from collections import OrderedDict

class TripletSegmentationVariables(object):    
    
    dataset_size_v2 = 21443
    dataset_size_v3 = 30955
    num_instuments = 6
    num_verbs = 10
    num_targets = 15
    num_triplets = 100
    num_verbtargets = 56
    
    width = 854
    height = 480
        
    categories = {
    'instrument': OrderedDict({
        '1': 'grasper',
        '2': 'bipolar',
        '3': 'hook',
        '4': 'scissors',
        '5': 'clipper',
        '6': 'irrigator'
    }),
    
    # 'instrument_direct_pred': OrderedDict({
    #     '1': 'grasper',
    #     '2': 'hook',
    #     '3': 'irrigator',
    #     '4': 'clipper',
    #     '5': 'bipolar',
    #     '6': 'scissors',
    #     '7': 'snare'
    # }),
       
    'verb': OrderedDict({
        '1': 'grasp',
        '2': 'retract',
        '3': 'dissect',
        '4': 'coagulate',
        '5': 'clip',
        '6': 'cut',
        '7': 'aspirate',
        '8': 'irrigate',
        '9': 'pack',
        '10': 'null_verb'
    }),
    
    'target': OrderedDict({
        '1': 'gallbladder',
        '2': 'cystic_plate',
        '3': 'cystic_duct',
        '4': 'cystic_artery',
        '5': 'cystic_pedicle',
        '6': 'blood_vessel',
        '7': 'fluid',
        '8': 'abdominal_wall_cavity',
        '9': 'liver',
        '10': 'adhesion',
        '11': 'omentum',
        '12': 'peritoneum',
        '13': 'gut',
        '14': 'specimen_bag',
        '15': 'null_target'
    }),
    
    'triplet': OrderedDict({
            '1': 'grasper,dissect,cystic_plate',
            '2': 'grasper,dissect,gallbladder',
            '3': 'grasper,dissect,omentum',
            '4': 'grasper,grasp,cystic_artery',
            '5': 'grasper,grasp,cystic_duct',
            '6': 'grasper,grasp,cystic_pedicle',
            '7': 'grasper,grasp,cystic_plate',
            '8': 'grasper,grasp,gallbladder',
            '9': 'grasper,grasp,gut',
            '10': 'grasper,grasp,liver',
            '11': 'grasper,grasp,omentum',
            '12': 'grasper,grasp,peritoneum',
            '13': 'grasper,grasp,specimen_bag',
            '14': 'grasper,pack,gallbladder',
            '15': 'grasper,retract,cystic_duct',
            '16': 'grasper,retract,cystic_pedicle',
            '17': 'grasper,retract,cystic_plate',
            '18': 'grasper,retract,gallbladder',
            '19': 'grasper,retract,gut',
            '20': 'grasper,retract,liver',
            '21': 'grasper,retract,omentum',
            '22': 'grasper,retract,peritoneum',
            '23': 'bipolar,coagulate,abdominal_wall_cavity',
            '24': 'bipolar,coagulate,blood_vessel',
            '25': 'bipolar,coagulate,cystic_artery',
            '26': 'bipolar,coagulate,cystic_duct',
            '27': 'bipolar,coagulate,cystic_pedicle',
            '28': 'bipolar,coagulate,cystic_plate',
            '29': 'bipolar,coagulate,gallbladder',
            '30': 'bipolar,coagulate,liver',
            '31': 'bipolar,coagulate,omentum',
            '32': 'bipolar,coagulate,peritoneum',
            '33': 'bipolar,dissect,adhesion',
            '34': 'bipolar,dissect,cystic_artery',
            '35': 'bipolar,dissect,cystic_duct',
            '36': 'bipolar,dissect,cystic_plate',
            '37': 'bipolar,dissect,gallbladder',
            '38': 'bipolar,dissect,omentum',
            '39': 'bipolar,grasp,cystic_plate',
            '40': 'bipolar,grasp,liver',
            '41': 'bipolar,grasp,specimen_bag',
            '42': 'bipolar,retract,cystic_duct',
            '43': 'bipolar,retract,cystic_pedicle',
            '44': 'bipolar,retract,gallbladder',
            '45': 'bipolar,retract,liver',
            '46': 'bipolar,retract,omentum',
            '47': 'hook,coagulate,blood_vessel',
            '48': 'hook,coagulate,cystic_artery',
            '49': 'hook,coagulate,cystic_duct',
            '50': 'hook,coagulate,cystic_pedicle',
            '51': 'hook,coagulate,cystic_plate',
            '52': 'hook,coagulate,gallbladder',
            '53': 'hook,coagulate,liver',
            '54': 'hook,coagulate,omentum',
            '55': 'hook,cut,blood_vessel',
            '56': 'hook,cut,peritoneum',
            '57': 'hook,dissect,blood_vessel',
            '58': 'hook,dissect,cystic_artery',
            '59': 'hook,dissect,cystic_duct',
            '60': 'hook,dissect,cystic_plate',
            '61': 'hook,dissect,gallbladder',
            '62': 'hook,dissect,omentum',
            '63': 'hook,dissect,peritoneum',
            '64': 'hook,retract,gallbladder',
            '65': 'hook,retract,liver',
            '66': 'scissors,coagulate,omentum',
            '67': 'scissors,cut,adhesion',
            '68': 'scissors,cut,blood_vessel',
            '69': 'scissors,cut,cystic_artery',
            '70': 'scissors,cut,cystic_duct',
            '71': 'scissors,cut,cystic_plate',
            '72': 'scissors,cut,liver',
            '73': 'scissors,cut,omentum',
            '74': 'scissors,cut,peritoneum',
            '75': 'scissors,dissect,cystic_plate',
            '76': 'scissors,dissect,gallbladder',
            '77': 'scissors,dissect,omentum',
            '78': 'clipper,clip,blood_vessel',
            '79': 'clipper,clip,cystic_artery',
            '80': 'clipper,clip,cystic_duct',
            '81': 'clipper,clip,cystic_pedicle',
            '82': 'clipper,clip,cystic_plate',
            '83': 'irrigator,aspirate,fluid',
            '84': 'irrigator,dissect,cystic_duct',
            '85': 'irrigator,dissect,cystic_pedicle',
            '86': 'irrigator,dissect,cystic_plate',
            '87': 'irrigator,dissect,gallbladder',
            '88': 'irrigator,dissect,omentum',
            '89': 'irrigator,irrigate,abdominal_wall_cavity',
            '90': 'irrigator,irrigate,cystic_pedicle',
            '91': 'irrigator,irrigate,liver',
            '92': 'irrigator,retract,gallbladder',
            '93': 'irrigator,retract,liver',
            '94': 'irrigator,retract,omentum',
            '95': 'grasper,null_verb,null_target',
            '96': 'bipolar,null_verb,null_target',
            '97': 'hook,null_verb,null_target',
            '98': 'scissors,null_verb,null_target',
            '99': 'clipper,null_verb,null_target',
            '100': 'irrigator,null_verb,null_target'
        }),
    
    'verbtarget': OrderedDict({
            '1': 'dissect,cystic_plate',
            '2': 'dissect,gallbladder',
            '3': 'dissect,omentum',
            '4': 'grasp,cystic_artery',
            '5': 'grasp,cystic_duct',
            '6': 'grasp,cystic_pedicle',
            '7': 'grasp,cystic_plate',
            '8': 'grasp,gallbladder',
            '9': 'grasp,gut',
            '10': 'grasp,liver',
            '11': 'grasp,omentum',
            '12': 'grasp,peritoneum',
            '13': 'grasp,specimen_bag',
            '14': 'pack,gallbladder',
            '15': 'retract,cystic_duct',
            '16': 'retract,cystic_pedicle',
            '17': 'retract,cystic_plate',
            '18': 'retract,gallbladder',
            '19': 'retract,gut',
            '20': 'retract,liver',
            '21': 'retract,omentum',
            '22': 'retract,peritoneum',
            '23': 'coagulate,abdominal_wall_cavity',
            '24': 'coagulate,blood_vessel',
            '25': 'coagulate,cystic_artery',
            '26': 'coagulate,cystic_duct',
            '27': 'coagulate,cystic_pedicle',
            '28': 'coagulate,cystic_plate',
            '29': 'coagulate,gallbladder',
            '30': 'coagulate,liver',
            '31': 'coagulate,omentum',
            '32': 'coagulate,peritoneum',
            '33': 'dissect,adhesion',
            '34': 'dissect,cystic_artery',
            '35': 'dissect,cystic_duct',
            '36': 'cut,blood_vessel',
            '37': 'cut,peritoneum',
            '38': 'dissect,blood_vessel',
            '39': 'dissect,peritoneum',
            '40': 'cut,adhesion',
            '41': 'cut,cystic_artery',
            '42': 'cut,cystic_duct',
            '43': 'cut,cystic_plate',
            '44': 'cut,liver',
            '45': 'cut,omentum',
            '46': 'clip,blood_vessel',
            '47': 'clip,cystic_artery',
            '48': 'clip,cystic_duct',
            '49': 'clip,cystic_pedicle',
            '50': 'clip,cystic_plate',
            '51': 'aspirate,fluid',
            '52': 'dissect,cystic_pedicle',
            '53': 'irrigate,abdominal_wall_cavity',
            '54': 'irrigate,cystic_pedicle',
            '55': 'irrigate,liver',
            '56': 'null_verb,null_target'})
    
    }
    
    instrument_to_verb_classes = {
                                0: [0, 1, 2, 8, 9],
                                1: [0, 1, 2, 3, 9],
                                2: [1, 2, 3, 5, 9],
                                3: [2, 3, 5, 9],
                                4: [4, 9],
                                5: [1, 2, 6, 7, 9]
                                }
    
    instrument_to_target_classes = {
                                0: [0, 1, 2, 3, 4, 8, 10, 11, 12, 13, 14],
                                1: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14],
                                2: [0, 1, 2, 3, 4, 5, 8, 10, 11, 14],
                                3: [0, 1, 2, 3, 5, 8, 9, 10, 11, 14],
                                4: [1, 2, 3, 4, 5, 14],
                                5: [0, 1, 2, 4, 6, 7, 8, 10, 14]
                                }
    
    instrument_to_verbtarget_classes = {
                                    0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,55],
                                    1: [0,1,2,6,9,12,14,15,17,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,55],
                                    2: [0,1,2,17,19,23,24,25,26,27,28,29,30,33,34,35,36,37,38,55],
                                    3: [0, 1, 2, 30, 35, 36, 39, 40, 41, 42, 43, 44, 55],
                                    4: [45, 46, 47, 48, 49, 55],
                                    5: [0, 1, 2, 17, 19, 20, 34, 50, 51, 52, 53, 54, 55]
                                    }
        
    
    instrument_to_triplet_classes = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 94],
        1: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 95], 
        2: [59, 60, 61, 63, 64, 46, 47, 48, 49, 50, 51, 52, 53, 57, 58, 54, 55, 56, 62, 96],
        3: [74, 75, 76, 65, 67, 73, 66, 68, 69, 70, 71, 72, 97], 
        4: [77, 78, 79, 80, 81, 98],
        5: [85, 86, 87, 91, 92, 93, 83, 82, 84, 88, 89, 90, 99]}
    
     
    
    instrument_colors = {
    'grasper': (255, 0, 0),        # red
    'hook': (0, 128, 0),           # green
    'irrigator': (255, 255, 0),    # yellow
    'clipper': (128, 0, 128),      # purple
    'bipolar': (255, 165, 0),      # orange
    'scissors': (0, 255, 255),     # cyan    
    }
    
    instrument_info =  {
    '0': {'name': 'grasper', 'isthing': 1, 'color': (255, 0, 0)},
    '1': {'name': 'bipolar', 'isthing': 1, 'color': (255, 165, 0)},
    '2': {'name': 'hook', 'isthing': 1, 'color': (0, 128, 0)},
    '3': {'name': 'scissors', 'isthing': 1,  'color': (0, 255, 255) },
    '4': {'name': 'clipper', 'isthing': 1,  'color': (128, 0, 128)},
    '5': {'name': 'irrigator', 'isthing': 1, 'color': (255, 255, 0)},
    }
    
    
    
    seq_to_split_dict_tiny = {
        'VID04_t50_sparse': 'train',
        'VID49_t50_sparse': 'test',
        'VID111_t50_sparse': 'val',
        }
    
    seq_to_split_dict_full = {
        'VID14_t50_full': 'test',
        'VID15_t50_full': 'test',
        'VID22_t50_full': 'test',
        'VID49_t50_sparse': 'test',
        'VID50_t50_sparse': 'test',
        'VID51_t50_sparse': 'test',
        'VID65_t50_sparse': 'test',
        'VID66_t50_sparse': 'test',
        
        'VID29_t50_full': 'val',
        'VID110_t50_sparse': 'val',
        'VID111_t50_sparse': 'val',
        
        'VID23_t50_full': 'train',
        'VID01_t50_full': 'train',
        'VID02_t50_sparse': 'train',
        'VID04_t50_sparse': 'train',
        'VID05_t50_sparse': 'train',
        'VID06_t50_sparse': 'train',
        'VID08_t50_sparse': 'train',
        'VID103_t50_sparse': 'train',
        'VID10_t50_sparse': 'train',
        'VID12_t50_full': 'train',
        'VID13_t50_sparse': 'train',
        'VID18_t50_full': 'train',
        'VID25_t50_full': 'train',
        'VID26_t50_full': 'train',
        'VID27_t50_full': 'train',
        'VID31_t50_sparse': 'train',
        'VID32_t50_sparse': 'train',
        'VID35_t50_full': 'train',
        'VID36_t50_sparse': 'train',
        'VID40_t50_sparse': 'train',
        'VID42_t50_sparse': 'train',
        'VID43_t50_full': 'train',
        'VID47_t50_sparse': 'train',
        'VID48_t50_full': 'train',
        
        'VID52_t50_full': 'train',
        'VID56_t50_sparse': 'train',
        'VID57_t50_sparse': 'train',
        'VID60_t50_sparse': 'train',
        'VID62_t50_sparse': 'train',
        'VID68_t50_sparse': 'train',
        'VID70_t50_sparse': 'train',
        'VID73_t50_sparse': 'train',
        'VID74_t50_sparse': 'train',
        'VID75_t50_sparse': 'train',
        'VID78_t50_sparse': 'train',
        'VID79_t50_sparse': 'train',
        'VID80_t50_sparse': 'train',
        'VID92_t50_sparse': 'train',
        'VID96_t50_sparse': 'train',
        }
    
    seq_to_split_dict_v3_gt_plus_prototype = {
        'VID14': 'test',
        'VID15': 'test',
        'VID22': 'test',
        'VID49': 'test',
        'VID50': 'test',
        'VID51': 'test',
        'VID65': 'test',
        'VID66': 'test',
        'VID29': 'val',
        'VID110': 'val',
        'VID111': 'val',
        'VID23': 'train',
        'VID01': 'train',
        'VID02': 'train',
        'VID04': 'train',
        'VID05': 'train',
        'VID06': 'train',
        'VID08': 'train',
        'VID103': 'train',
        'VID10': 'train',
        'VID12': 'train',
        'VID13': 'train',
        'VID18': 'train',
        'VID25': 'train',
        'VID26': 'train',
        'VID27': 'train',
        'VID31': 'train',
        'VID32': 'train',
        'VID35': 'train',
        'VID36': 'train',
        'VID40': 'train',
        'VID42': 'train',
        'VID43': 'train',
        'VID47': 'train',
        'VID48': 'train',
        'VID52': 'train',
        'VID56': 'train',
        'VID57': 'train',
        'VID60': 'train',
        'VID62': 'train',
        'VID68': 'train',
        'VID70': 'train',
        'VID73': 'train',
        'VID74': 'train',
        'VID75': 'train',
        'VID78': 'train',
        'VID79': 'train',
        'VID80': 'train',
        'VID92': 'train',
        'VID96': 'train'
        
        }


class CholecSeg8kVariables(object):
    dataset_size = 8080
    conversion_from_cholecseg8k_to_tissue_segmentation = {
        50: {'orig_name': 'black_background', 'new_name': 'background', 'id': 0, 'color': (0, 0, 0)},
        255: {'orig_name': 'None', 'new_name': 'background', 'id': 0, 'color': (0, 0, 0)},
        11: {'orig_name': 'abdominal_wall', 'new_name': 'abdominal_wall', 'id': 1, 'color': (128, 0, 0)},
        21: {'orig_name': 'liver', 'new_name': 'liver', 'id': 2, 'color': (0, 128, 0)},
        13: {'orig_name': 'gastrointestinal_tract', 'new_name': 'gastrointestinal_tract', 'id': 3, 'color': (0, 0, 128)},
        12: {'orig_name': 'fat', 'new_name': 'fat', 'id': 4, 'color': (128, 128, 0)},
        23: {'orig_name': 'connective_tissue', 'new_name': 'connective_tissue', 'id': 5, 'color': (128, 0, 128)},
        24: {'orig_name': 'blood', 'new_name': 'blood', 'id': 6, 'color': (0, 128, 128)},
        25: {'orig_name': 'cystic_duct', 'new_name': 'cystic_duct', 'id': 7, 'color': (255, 255, 0)},
        22: {'orig_name': 'gallbladder', 'new_name': 'gallbladder', 'id': 8, 'color': (255, 0, 255)},
        33: {'orig_name': 'hepatic_vein', 'new_name': 'hepatic_vein', 'id': 9, 'color': (0, 255, 255)},
        5: {'orig_name': 'liver_ligament', 'new_name': 'liver_ligament', 'id': 10, 'color': (220, 40, 80)},
        31: {'orig_name': 'grasper', 'new_name': 'instrument', 'id': 11, 'color': (80, 120, 220)},
        32: {'orig_name': 'L-hook Electrocautery', 'new_name': 'instrument', 'id': 11, 'color': (80, 120, 220)},
    }
    
    seq_to_split_dict = {
        'VID09': 'test',
        'VID20': 'test',
        'VID24': 'test',
        'VID55': 'test',
        
        'VID01': 'train',
        'VID12': 'train',
        'VID18': 'train',
        'VID25': 'train',
        'VID26': 'train',
        'VID27': 'train',
        'VID35': 'train',
        'VID43': 'train',
        'VID48': 'train',
        'VID52': 'train',
        
        'VID17': 'val',
        'VID28': 'val',
        'VID37': 'val',
        
        }


