class TripletSegmentationVariables(object):        
    categories = {
    'instrument': {
        '1': 'grasper',
        '2': 'bipolar',
        '3': 'hook',
        '4': 'scissors',
        '5': 'clipper',
        '6': 'irrigator'
    },
       
    'verb': {
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
    },
    
    'target': {
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
    },
    
    'triplet': {
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
        }
    }
    
    dataset_size = 21443
    
    width = 854
    height = 480
    
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
    
    

    
    