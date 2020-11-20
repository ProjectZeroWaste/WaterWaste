import sys, os, json
import argparse
import logging
from tqdm import tqdm

from detector import Detector
#from dataset_config import DetectorCFG
from dataset_config_ds2_storm import DetectorCFG
import viz_utils
import img_utils
from create_gif import gif_creator
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


EXPANSION = 3 # Default

CONFIDENCE=0.7 #standard
#CONFIDENCE=0.45

def main(args):
    cropped_path = os.path.join(args.output_dir,'cropped')
    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)
    bbox_path = os.path.join(args.output_dir, 'bbox')
    if not os.path.exists(bbox_path):
        os.mkdir(bbox_path)
    detection_results = []

    labelmap = DetectorCFG.labelmap
    tf_detector = Detector(DetectorCFG.model_path)

    # Log and list output filenames some checks for file collisions 
    crop_output_image_list = {}
    bbox_output_image_list = {}

    image_list = img_utils.list_imgs_subdir(args.image_dir)

    #image_list = np.choose(image_list, 90)
    
    #for batch, names in tqdm(iterator):
    for img_names in tqdm(image_list):
        image = img_utils.read_image(img_names)
        image.load()

        result = tf_detector.predict(image, img_names)
        detection_results.append(result)

        """
        # Create Crop Images
        print("predictions complete.")
        print("Starting Cropping Images.")
        img_cropped_list = img_utils.crop_image(result['detections'], image, expansion=EXPANSION)
        for ix, img_cropped in enumerate(img_cropped_list):
            img_crop_path, crop_output_image_list = img_utils.set_output_filename(img_names, 
                                                            crop_index=ix, 
                                                            output_image_list=crop_output_image_list, 
                                                            output_dir=cropped_path)
            img_cropped.save(img_crop_path)
        """
        print("Creating BBOX for Original Image.")
        # Create BBOX
        viz_utils.render_detection_bounding_boxes(result['detections'], image, 
                                                  label_map=labelmap,
                                                  confidence_threshold=CONFIDENCE)
        bbox_img_path, bbox_output_image_list = img_utils.set_output_filename(img_names, 
                                                        output_image_list=bbox_output_image_list, 
                                                        output_dir=bbox_path)
        image.save(bbox_img_path)

    print("done.")

    output_detector_file_results = os.path.join(args.output_dir, 'detector_results.json')
    with open(output_detector_file_results, 'w') as f:
        json.dump(detection_results, f, indent=1)
    print('Output file saved at {}'.format(output_detector_file_results))

    gif_creator('ssd_r50_fpn_ds2s_SynReal_111720_step25k_Valv0')

if __name__ == '__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--image_dir', help='dir for images')
    #args = parser.parse_args()

    from argparse import Namespace
    args = Namespace(
        image_dir = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Validation_v0/high_res_validation_set/',    #.JPG - img_util
        output_dir='/home/redne/WaterWaste/test_images/ssd_r50_fpn_ds2s_SynReal_111720_step25k_Valv0/'
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)    


    sys.exit(main(args))