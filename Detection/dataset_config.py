
def load_label_map(LABEL_PATH, NUM_CLASSES):
    from object_detection.utils import label_map_util

    try:
        label_map = label_map_util.load_labelmap(LABEL_PATH)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
    except:
        import traceback
        traceback.print_exc()
    return category_index

def load_label_map2(label_map):
    label_id_name = dict()
    label_name_id = dict()

    for label in label_map:        
        label_id_name[label['id']] = label['name']
        label_name_id[label['name']] = label['id']

    return label_id_name, label_name_id

DEFAULT_DETECTOR_LABELMAP = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'Trash'}
]

class DetectorCFG:
    model_path = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/tfodapi/ds0v5/frcnn_acoco_ds0v5_trash_102620_step3k/frozen_inference_graph.pb'
    labelmap = {'1': 'Trash'}
    labelmap_num = 1
    detection_confidence = 0.1
    detector_box_confidence = 0.1
    label_id_name, label_name_id = load_label_map2(DEFAULT_DETECTOR_LABELMAP)