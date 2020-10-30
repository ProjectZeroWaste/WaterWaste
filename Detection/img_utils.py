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
