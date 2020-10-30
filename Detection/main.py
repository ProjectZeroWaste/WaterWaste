import sys, os, json
import argparse
import logging
from tqdm import tqdm

from detector import Detector
import viz_utils
from img_utils import list_imgs_subdir, read_image, CropBoxes, crop_image, set_output_filename


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pb_model = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/tfodapi/ds0v5/frcnn_acoco_ds0v5_trash_102620_step3k/frozen_inference_graph.pb'
labels_map = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/tfodapi/ds0v5/frcnn_acoco_ds0v5_trash_102620_step3k/labelmap.pbtxt'

category = 'Trash'
nb_of_classes = 1
batch_size = 1

def main(args):
    #img_path = os.path.join(args.dataset, 'original')
    cropped_path = os.path.join(args.output_dir,'cropped')
    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)
    bbox_path = os.path.join(args.output_dir, 'bbox')
    if not os.path.exists(bbox_path):
        os.mkdir(bbox_path)
    detection_results = []

    labelmap = {'1': 'Trash'}

    #iterator = BatchIterator(img_path, batch_size, continue_path = cropped_path)
    #iterator = BatchIterator(img_path, batch_size)
    #detector = ObjectDetection(pb_model, labels_map, nb_of_classes)

    tf_detector = Detector(pb_model)
    #get_category = GetCategoryBox(category, img_path)
    #crop_boxes = CropBoxes(img_path, cropped_path)

    # Dictionary mapping output file names to a collision-avoidance count.
    # #https://github.com/microsoft/CameraTraps/blob/b17e2977c1ee52761ba0e54123e8363495bcae45/detection/run_tf_detector.py#L338
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    crop_output_image_list = {}
    bbox_output_image_list = {}

    image_list = list_imgs_subdir(args.image_dir)
    #for batch, names in tqdm(iterator):
    for img_names in tqdm(image_list):
        image = read_image(img_names)
        image.load()

        result = tf_detector.predict(image)
        detection_results.append(result)

        
        # Create Crop Images
        img_cropped_list = crop_image(result['detections'], image, expansion=3)
        for ix, img_cropped in enumerate(img_cropped_list):
            img_crop_path, crop_output_image_list = set_output_filename(img_names, 
                                                            crop_index=ix, 
                                                            output_image_list=crop_output_image_list, 
                                                            output_dir=cropped_path)
            img_cropped.save(img_crop_path)
        
        #main_boxes, images = get_category(boxes, names)
        #crop_boxes(main_boxes, images)

        # Create BBOX
        viz_utils.render_detection_bounding_boxes(result['detections'], image, 
                                                  label_map=labelmap,
                                                  confidence_threshold=0.7)
        bbox_img_path, bbox_output_image_list = set_output_filename(img_names, 
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
    #parser.add_argument('--dataset', help='train, test or valid')
    #args = parser.parse_args()

    from argparse import Namespace
    args = Namespace(
        #image_dir =  '/home/redne/ZeroWaste3D/Detection/tf/detect_and_crop/utils/dev/images/original/',
        image_dir = '/home/redne/WaterWaste/test_images/s/original_02/',
        output_dir='/home/redne/WaterWaste/test_images/output_01/'
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)    


    sys.exit(main(args))