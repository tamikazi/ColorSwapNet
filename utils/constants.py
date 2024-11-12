import numpy as np
import torch

# ROOT_DATASET = "/home/shaakira.gadiwan/project/data/ADEChallengeData2016"
# ROOT_DATASET = "C:/Users/tahmi/Documents/MENG2023/ENEL645/ADEChallengeData2016_processed/ADEChallengeData2016"
ROOT_DATASET = "/home/tahmid.kazi/project_data/ADEChallengeData2016"

# CKPT_DIR_PATH_SEG = "/home/shaakira.gadiwan/project/best_models"
# CKPT_DIR_PATH_SEG = "C:/Users/tahmi/Documents/MENG2023/ENEL645/models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_PER_GPU = 2
NUM_WORKERS = 2

NUM_ITER_PER_EPOCH_SEG = 5000
NUM_EPOCHS_SEG = 20
TOTAL_NUM_ITER_SEG = NUM_ITER_PER_EPOCH_SEG * NUM_EPOCHS_SEG

IMG_SIZES_SEG = (300, 375, 450, 525, 575)
IMG_MAX_SIZE_SEG = 900
PADDING_SEG = 8  # 8 when dilated, 32 when not dilated
SEGM_DOWNSAMPLING_RATE_SEG = 8  # 8 when dilated, 32 when not dilated

IMG_SIZES_GAN = (300, 375, 450, 525, 575)
IMG_MAX_SIZE_GAN = 900
PADDING_GAN = 8  # 8 when dilated, 32 when not dilated
SEGM_DOWNSAMPLING_RATE_GAN = 8  # 8 when dilated, 32 when not dilated

OPTIMIZER_PARAMETERS_SEG = {
        "LEARNING_RATE": 0.02,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4
    }

NUM_CLASSES = 2
FC_DIM_SEG = 2048  # 512 for resnet18, for rest it is 2048

#MODEL_ENCODER_WEIGHTS_PATH_SEG = path to the model weights
#MODEL_DECODER_WEIGHTS_PATH_SEG = 

#MODEL_ENCODER_WEIGHTS_PATH_GAN = path to the model weights
#MODEL_DECODER_WEIGHTS_PATH_GAN = 

LIST_SCENES = ['airplane_cabin', 'airport_terminal', 'airport_ticket_counter', 'alcove', 'amusement_arcade', 'anechoic_chamber', 
               'apse_indoor', 'aquarium', 'aquatic_theater', 'arcade', 'archive', 'arrival_gate_indoor', 'art_gallery', 'art_studio', 
               'artists_loft', 'assembly_line', 'athletic_field_indoor', 'atrium_home', 'atrium_public', 'attic', 'auditorium', 
               'auto_factory', 'auto_mechanics_indoor', 'auto_showroom', 'backseat', 'backstage', 'backstairs', 'backstairs_indoor', 
               'badminton_court_indoor', 'baggage_claim', 'bakery', 'balcony_interior', 'ball_pit', 'ballroom', 'bank_indoor', 'bank_vault', 
               'banquet_hall', 'baptistry_indoor', 'bar', 'barbershop', 'basement', 'basketball_court_indoor', 'bathhouse', 'bathroom', 
               'batting_cage_indoor', 'bazaar_indoor', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_hall', 'billiard_room', 'biology_laboratory', 
               'bleachers_indoor', 'bomb_shelter_indoor', 'bookbindery', 'bookstore', 'booth_indoor', 'bow_window_indoor', 'bowling_alley', 
               'boxing_ring', 'boxing_ring_indoor', 'breakroom', 'brewery_indoor', 'brickyard_indoor', 'bridge_indoor', 'broadcasting_room', 
               'building_lobby', 'burial_chamber', 'bus_interior', 'bus_station_indoor', 'butchers_shop', 'cabin_indoor', 'cafeteria', 'camera_store', 
               'campus', 'candy_store', 'canteen', 'car_dealership', 'car_interior', 'cardroom', 'cargo_container_interior', 'carousel_indoor', 
               'carport_indoor', 'casino_indoor', 'catacomb', 'cathedral_indoor', 'cavern_indoor', 'changing_room', 'chapel', 'checkout_counter', 
               'chemistry_lab', 'chicken_coop_indoor', 'chicken_farm_indoor', 'childs_room', 'choir_loft_interior', 'church_indoor', 'cinema_indoor', 
               'circus_tent_indoor', 'classroom', 'clean_room', 'clock_tower_indoor', 'cloister_indoor', 'closet', 'clothing_store', 'cocktail_lounge', 
               'coffee_shop', 'command_center', 'computer_lab', 'computer_room', 'concert_hall', 'conference_room', 'control_room', 'control_tower_indoor', 
               'convenience_store', 'convenience_store_indoor', 'convention_center', 'corridor', 'courthouse', 'courthouse_indoor', 'courtroom', 
               'covered_bridge_indoor', 'crawl_space', 'crematorium', 'crypt', 'cybercafe', 'dairy_indoor', 'dance_studio', 'darkroom', 'day_care_center', 
               'delicatessen', 'dentists_office', 'department_store', 'departure_lounge', 'designers_office', 'diner_indoor', 'dinette_home', 'dining_car', 
               'dining_hall', 'dining_room', 'discotheque', 'doorway_indoor', 'dorm_room', 'dormitory', 'dress_shop', 'dressing_room', 'driving_range_indoor', 
               'drugstore', 'editing_room', 'elevator_interior', 'elevator_lobby', 'elevator_shaft', 'emergency_room', 'engine_room', 'entrance_hall', 'escalator_indoor', 
               'examination_room', 'exhibition_hall', 'factory_indoor', 'fastfood_restaurant', 'ferryboat_indoor', 'firing_range_indoor', 'fishmarket_indoor', 
               'fitting_room', 'fitting_room_interior', 'florist_shop_indoor', 'food_court', 'freight_elevator', 'frontseat', 'funeral_home', 'furnace_room', 
               'gambling_hall', 'game_room', 'garage_indoor', 'gazebo_interior', 'general_store_indoor', 'geodesic_dome_indoor', 'gift_shop', 'great_hall', 
               'greenhouse_indoor', 'guardroom', 'gun_deck_indoor', 'gym_indoor', 'gymnasium_indoor', 'hair_salon', 'hallway', 'handball_court', 'hangar_indoor', 
               'home_office', 'home_theater', 'hospital_room', 'hot_tub_indoor', 'hotel_breakfast_area', 'hotel_room', 'hunting_lodge_indoor', 'ice_skating_rink_indoor', 
               'indoor_procenium', 'indoor_round', 'inn_indoor', 'jacuzzi_indoor', 'jail_indoor', 'jail_indoor', 'kennel_indoor', 'kindergarden_classroom', 'kiosk_indoor', 
               'kitchen', 'kitchenette', 'laboratory', 'labyrinth_indoor', 'laundromat', 'lavatory', 'lecture_room', 'legislative_chamber', 'library', 'library_indoor', 
               'lido_deck_indoor', 'liquor_store_indoor', 'living_room', 'lobby', 'locker_room', 'lookout_station_indoor', 'market_indoor', 'market_indoor', 
               'massage_room', 'mess_hall', 'mezzanine', 'mini_golf_course_indoor', 'monastery_indoor', 'mosque_indoor', 'movie_theater_indoor', 'museum_indoor', 
               'music_studio', 'newsroom', 'nursery', 'observatory_indoor', 'office', 'office_cubicles', 'operating_room', 'pantry', 'parking_garage_indoor', 
               'pharmacy', 'pilothouse_indoor', 'playroom', 'podium_indoor', 'poolroom_home', 'reading_room', 'reception', 'recreation_room', 'restaurant', 
               'restroom_indoor', 'road_indoor', 'room', 'sauna', 'school_indoor', 'server_room', 'sewing_room', 'shoe_shop', 'shop', 'shopping_mall', 
               'shopping_mall_indoor', 'shower', 'shower_room', 'squash_court', 'stage_indoor', 'staircase', 'storage_room', 'subway_interior', 'subway_station', 
               'supermarket', 'swimming_pool_indoor', 'synagogue_indoor', 'tearoom', 'television_studio', 'tennis_court_indoor', 'theater_indoor', 'throne_room', 
               'ticket_booth', 'ticket_window_indoor', 'train_interior', 'train_station', 'utility_room', 'veterinarians_office', 'volleyball_court_indoor', 'waiting_room', 
               'warehouse_indoor', 'washroom', 'widows_walk_interior', 'wine_cellar', 'workshop_indoor', 'wrestling_ring_indoor', 'yoga_studio']