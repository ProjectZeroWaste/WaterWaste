import sys, os, json
import argparse
import logging
from tqdm import tqdm

from detector import Detector
from dataset_config import DetectorCFG
import viz_utils
import img_utils


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


#EXPANSION = 3 # Default
EXPANSION = 200

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
    #for batch, names in tqdm(iterator):
    for img_names in tqdm(image_list):
        image = img_utils.read_image(img_names)
        image.load()

        result = tf_detector.predict(image, img_names)
        detection_results.append(result)

        
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
        
        print("Creating BBOX for Original Image.")
        # Create BBOX
        viz_utils.render_detection_bounding_boxes(result['detections'], image, 
                                                  label_map=labelmap,
                                                  confidence_threshold=0.7)
        bbox_img_path, bbox_output_image_list = img_utils.set_output_filename(img_names, 
                                                        output_image_list=bbox_output_image_list, 
                                                        output_dir=bbox_path)
        image.save(bbox_img_path)

    print("done.")

    output_detector_file_results = os.path.join(args.output_dir, 'detector_results.json')
    with open(output_detector_file_results, 'w') as f:
        json.dump(detection_results, f, indent=1)
    print('Output file saved at {}'.format(output_detector_file_results))



if __name__ == '__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--image_dir', help='dir for images')
    #args = parser.parse_args()

    from argparse import Namespace
    args = Namespace(
        #image_dir =  '/home/redne/ZeroWaste3D/Detection/tf/detect_and_crop/utils/dev/images/original/',
        #image_dir = '/home/redne/WaterWaste/test_images/s/original_02/',
        #output_dir='/home/redne/WaterWaste/test_images/output_02/'

        image_dir = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Validation_v0/high_res_validation_set/',
        output_dir='/home/redne/WaterWaste/test_images/high_res_val_set_01/'
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)    


    sys.exit(main(args))