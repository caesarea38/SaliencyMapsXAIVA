# ------------------
# DATASET ROOT PATHS
# ------------------

cub_root = './cub200'
cifar10_root = './cifar10'
cifar100_root = './cifar100'

# -----------------
# DATASET CLASS NAMES
# -----------------

explanation_methods = ['CAM', 'GradCAM', 'ScoreCAM', 'SmoothScoreCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM']
class_names = {
    'cifar10': {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'},
    'cifar100': {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'},
    'cub200': {
        1: 'Laysan_Albatross', 
        2: 'Sooty_Albatross', 
        4: 'Crested_Auklet',
        6: 'Parakeet_Auklet',
        9: 'Red_winged_Blackbird',
        10: 'Rusty_Blackbird',
        11: 'Yellow_headed_Blackbird',
        12: 'Bobolink',
        14: 'Lazuli_Bunting',
        15: 'Painted_Bunting',
        16: 'Cardinal',
        17: 'Spotted_Catbird',
        18: 'Gray_Catbird',
        19: 'Yellow_breasted_Chat',
        20: 'Eastern_Towhee',
        21: 'Chuck_wills_Widow',
        23: 'Red_faced_Cormorant',
        24: 'Pelagic_Cormorant',
        25: 'Bronzed_Cowbird',
        26: 'Shiny_Cowbird',
        27: 'Brown_Creeper',
        29: 'Fish_Crow',
        31: 'Mangrove_Cuckoo',
        38: 'Least_Flycatcher',
        39: 'Olive_sided_Flycatcher',
        40: 'Scissor_tailed_Flycatcher',
        41: 'Vermilion_Flycatcher',
        43: 'Frigatebird',
        44: 'Northern_Fulmar',
        45: 'Gadwall',
        46: 'American_Goldfinch',
        47: 'European_Goldfinch',
        49: 'Eared_Grebe',
        51: 'Pied_billed_Grebe',
        53: 'Blue_Grosbeak',
        54: 'Evening_Grosbeak',
        55: 'Pine_Grosbeak',
        56: 'Rose_breasted_Grosbeak',
        57: 'Pigeon_Guillemot',
        58: 'California_Gull',
        59: 'Glaucous_winged_Gull',
        60: 'Heermanns_Gull',
        61: 'Herring_Gull',
        62: 'Ivory_Gull',
        63: 'Ring_billed_Gull',
        64: 'Slaty_backed_Gull',
        66: 'Annas_Hummingbird',
        67: 'Ruby_throated_Hummingbird',
        68: 'Rufous_Hummingbird',
        69: 'Green_Violetear',
        70: 'Long_tailed_Jaeger',
        72: 'Blue_Jay',
        73: 'Florida_Jay',
        74: 'Green_Jay',
        75: 'Dark_eyed_Junco',
        76: 'Tropical_Kingbird',
        77: 'Gray_Kingbird',
        79: 'Green_Kingfisher',
        80: 'Pied_Kingfisher',
        81: 'Ringed_Kingfisher',
        84: 'Horned_Lark',
        86: 'Mallard',
        87: 'Western_Meadowlark',
        88: 'Hooded_Merganser',
        89: 'Red_breasted_Merganser',
        91: 'Nighthawk',
        92: 'Clarks_Nutcracker',
        93: 'White_breasted_Nuthatch',
        96: 'Orchard_Oriole',
        98: 'Ovenbird',
        99: 'Brown_Pelican',
        103: 'American_Pipit',
        104: 'Whip_poor_Will',
        105: 'Horned_Puffin',
        106: 'Common_Raven',
        107: 'White_necked_Raven',
        108: 'American_Redstart',
        109: 'Geococcyx',
        110: 'Loggerhead_Shrike',
        112: 'Bairds_Sparrow',
        114: 'Brewers_Sparrow',
        115: 'Chipping_Sparrow',
        116: 'Clay_colored_Sparrow',
        117: 'House_Sparrow',
        119: 'Fox_Sparrow',
        121: 'Harriss_Sparrow',
        122: 'Henslows_Sparrow',
        123: 'Le_Contes_Sparrow',
        124: 'Lincolns_Sparrow',
        125: 'Nelsons_Sparrow',
        126: 'Savannah_Sparrow',
        127: 'Seaside_Sparrow',
        128: 'Song_Sparrow',
        130: 'Vesper_Sparrow',
        131: 'White_crowned_Sparrow',
        132: 'White_throated_Sparrow',
        133: 'Cape_Glossy_Starling',
        135: 'Barn_Swallow',
        136: 'Cliff_Swallow',
        138: 'Scarlet_Tanager',
        139: 'Summer_Tanager',
        140: 'Artic_Tern',
        141: 'Black_Tern',
        142: 'Caspian_Tern',
        143: 'Common_Tern',
        144: 'Elegant_Tern',
        145: 'Forsters_Tern',
        147: 'Green_tailed_Towhee',
        148: 'Brown_Thrasher',
        149: 'Sage_Thrasher',
        150: 'Black_capped_Vireo',
        151: 'Blue_headed_Vireo',
        152: 'Philadelphia_Vireo',
        153: 'Red_eyed_Vireo',
        154: 'Warbling_Vireo',
        156: 'Yellow_throated_Vireo',
        157: 'Bay_breasted_Warbler',
        158: 'Black_and_white_Warbler',
        159: 'Black_throated_Blue_Warbler',
        160: 'Blue_winged_Warbler',
        161: 'Canada_Warbler',
        163: 'Cerulean_Warbler',
        166: 'Hooded_Warbler',
        167: 'Kentucky_Warbler',
        168: 'Magnolia_Warbler',
        169: 'Mourning_Warbler',
        170: 'Myrtle_Warbler',
        171: 'Nashville_Warbler',
        172: 'Orange_crowned_Warbler',
        173: 'Palm_Warbler',
        174: 'Pine_Warbler',
        175: 'Prairie_Warbler',
        176: 'Prothonotary_Warbler',
        177: 'Swainsons_Warbler',
        178: 'Tennessee_Warbler',
        180: 'Worm_eating_Warbler',
        181: 'Yellow_Warbler',
        183: 'Louisiana_Waterthrush',
        187: 'Pileated_Woodpecker',
        188: 'Red_bellied_Woodpecker',
        189: 'Red_cockaded_Woodpecker',
        190: 'Red_headed_Woodpecker',
        191: 'Downy_Woodpecker',
        192: 'Bewicks_Wren',
        193: 'Cactus_Wren',
        194: 'Carolina_Wren',
        195: 'House_Wren',
        197: 'Rock_Wren',
        198: 'Winter_Wren',
        199: 'Common_Yellowthroat',
        0: 'Black_footed_Albatross',
        3: 'Groove_billed_Ani',
        5: 'Least_Auklet',
        7: 'Rhinoceros_Auklet',
        8: 'Brewers_Blackbird',
        13: 'Indigo_Bunting',
        22: 'Brandts_Cormorant',
        28: 'American_Crow',
        30: 'Black_billed_Cuckoo',
        32: 'Yellow_billed_Cuckoo',
        33: 'Gray_crowned_Rosy_Finch',
        34: 'Purple_Finch',
        35: 'Northern_Flicker',
        36: 'Acadian_Flycatcher',
        37: 'Great_Crested_Flycatcher',
        42: 'Yellow_bellied_Flycatcher',
        48: 'Boat_tailed_Grackle',
        50: 'Horned_Grebe',
        52: 'Western_Grebe',
        65: 'Western_Gull',
        71: 'Pomarine_Jaeger',
        78: 'Belted_Kingfisher',
        82: 'White_breasted_Kingfisher',
        83: 'Red_legged_Kittiwake',
        85: 'Pacific_Loon',
        90: 'Mockingbird',
        94: 'Baltimore_Oriole',
        95: 'Hooded_Oriole',
        97: 'Scotts_Oriole',
        100: 'White_Pelican',
        101: 'Western_Wood_Pewee',
        102: 'Sayornis',
        111: 'Great_Grey_Shrike',
        113: 'Black_throated_Sparrow',
        118: 'Field_Sparrow',
        120: 'Grasshopper_Sparrow',
        129: 'Tree_Sparrow',
        134: 'Bank_Swallow',
        137: 'Tree_Swallow',
        146: 'Least_Tern',
        155: 'White_eyed_Vireo',
        162: 'Cape_May_Warbler',
        164: 'Chestnut_sided_Warbler',
        165: 'Golden_winged_Warbler',
        179: 'Wilsons_Warbler',
        182: 'Northern_Waterthrush',
        184: 'Bohemian_Waxwing',
        185: 'Cedar_Waxwing',
        186: 'American_Three_toed_Woodpecker',
        196: 'Marsh_Wren'
    }
}