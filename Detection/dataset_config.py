class DetectorCFG:
    model_path = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/tfodapi/ds0v5/frcnn_acoco_ds0v5_trash_102620_step3k/frozen_inference_graph.pb'
    labelmap = {'1': 'Trash'}
    labelmap_num = 1
    detection_confidence = 0.1
    detector_box_confidence = 0.1