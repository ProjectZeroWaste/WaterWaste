
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
    {'id': 1, 'name': 'H_beveragebottle'},
    {'id': 2, 'name': 'D_lid'},
    {'id': 3, 'name': 'S_cup'},
    {'id': 4, 'name': 'P_foodcontainer'},
    {'id': 5, 'name': 'P_beveragecontainer'},
    {'id': 6, 'name': 'D_foodcontainer'},
    {'id': 7, 'name': 'H_facemask'},
    {'id': 8, 'name': 'M_aerosol'},
    {'id': 9, 'name': 'H_otherbottle'},
    {'id': 10, 'name': 'P_cup'},
    {'id': 11, 'name': 'M_beveragecan'}
]

#ssd_r50_fpn_ds2_storm_111520_step18k
#frcnn_inceptionv2_ds2_storm_111620_step30k
#frcnn_inceptionv2_ds2_storm_SynReal_111620_step9k
#frcnn_inceptionv2_ds2_storm_SynReal_111620_step16k
#frcnn_inceptionv2_ds2_storm_SynReal_111620_x2_shards_step28k

#frcnn_ds2s_Syn70Rea30_111720_step14k frcnn_ds2x_s70r30_111720_14k
#frcnn_ds2s_Syn30Rea70_111720_step20k frcnn_ds2s_s30r70_111720_20k
#frcnn_ds2s_Syn50Rea50_111720_step16k frcnn_ds2s_s50r50_111720_16k
#frcnn_ds2s_Syn30Rea70_111720_step31k frcnn_ds2s_s30r70_111720_31k
#frcnn_ds2s_Syn70Rea30_111720_step35k
#frcnn_ds2s_Syn0Real100_111720_step20k
# ssd_r50_fpn_ds2s_SynReal_111720_step25k
class DetectorCFG:
    model_path = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/tfodapi/ds2/ssd_r50_fpn_ds2s_SynReal_111720_step25k/frozen_inference_graph.pb'
    labelmap = {'1':'H_beveragebottle', '2':'D_lid', '3':'S_cup', '4':'P_foodcontainer','5':'P_beveragecontainer','6':'D_foodcontainer','7':'H_facemask', '8':'M_aerosol', '9':'H_otherbottle', '10': 'P_cup', '11':'M_beveragecan'}
    labelmap_num = 11
    detection_confidence = 0.1
    detector_box_confidence = 0.1
    label_id_name, label_name_id = load_label_map2(DEFAULT_DETECTOR_LABELMAP)

#Good ones
##frcnn_r50_ds2_storm_111620_step10k    frcnn_r50_ds2s_111620_step10k