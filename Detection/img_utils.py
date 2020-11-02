import os
import glob

# may need in the future to validate more image extensions than `.jpg`
# img

def list_imgs_subdir(img_dir):
    """Recursively List all images from all subdirectories"""
    return glob.glob(os.path.join(img_dir, '**', '*.jpg*'), recursive=True)

def list_imgs_onedir(img_dir):
    """(no recusive) List all images from ONE subdirectories"""
    return glob.glob(os.path.join(img_dir, '*.jpg*'))


import numpy as np
from PIL import Image


def read_image(img_path):
    image = Image.open(img_path)
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image



def set_output_filename(image_name, output_image_list, output_dir, crop_index=-1):
    #https://github.com/microsoft/CameraTraps/blob/b17e2977c1ee52761ba0e54123e8363495bcae45/detection/run_tf_detector.py#L338
    
    image_name = os.path.basename(image_name).lower()
    name, ext = os.path.splitext(image_name)
    if crop_index >= 0:
        name += '_crop{:0>2d}'.format(crop_index)
    image_name = '{}{}{}'.format(name, '_detections', '.jpg')
    if image_name in output_image_list:
        n_collisions = output_image_list[image_name]
        image_name = '{:0>4d}'.format(n_collisions) + '_' + image_name
        output_image_list[image_name] += 1
    else:
        output_image_list[image_name] = 0
    image_name = os.path.join(output_dir, image_name)
    return image_name, output_image_list


def crop_image(detections, image, confidence_threshold=0.8, expansion=0):
    """
    #https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector.py#L338
    Crops detections above *confidence_threshold* from the PIL image *image*,
    returning a list of PIL images.
    *detections* should be a list of dictionaries with keys 'conf' and 'bbox';
    see bbox format description below.  Normalized, [x,y,w,h], upper-left-origin.
    *expansion* specifies a number of pixels to include on each side of the box.
    """

    ret_images = []

    for detection in detections:

        score = float(detection['conf'])

        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

            # Convert to pixels so we can use the PIL crop() function
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if expansion > 0:
                left -= expansion
                right += expansion
                top -= expansion
                bottom += expansion

            # PIL's crop() does surprising things if you provide values outside of
            # the image, clip inputs
            left = max(left,0); right = max(right,0)
            top = max(top,0); bottom = max(bottom,0)

            left = min(left,im_width-1); right = min(right,im_width-1)
            top = min(top,im_height-1); bottom = min(bottom,im_height-1)

            ret_images.append(image.crop((left, top, right, bottom)))

        # ...if this detection is above threshold

    # ...for each detection

    return ret_images