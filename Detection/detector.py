import os
import numpy as np
import tensorflow as tf

from helpers import truncate_float


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_model(tf_model_path):
    print('Tensorflow model graph loaded into memory...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(tf_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('Tensorflow detection session started.')

    return detection_graph


class Detector:

    def __init__(self, tf_model_path):

        # self.download_models()
        self.tfodapi_init(tf_model_path)
        print("Model loaded. Ready for inference")

    def tfodapi_init(self, tf_model_path):
        # load frozen tensorflow detection model and initialize
        # the tensorflow graph
        detection_graph = load_model(tf_model_path)
        self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        #self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def set_bounding_box(self, bbox):
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]

        bbox_coord = [bbox[1], bbox[0], bbox_w, bbox_h]

        for ix, bx in enumerate(bbox_coord):
            bbox_coord[ix] = truncate_float(float(bx), precision=4)

        return bbox_coord

    def process_boxes(self, boxes, classes, scores, min_score=0.1):
        """Processes detected box coordinates and removes if under the detector confidence score `min_score`"""
        predictions = zip(boxes, scores, classes)

        detection_results = []
        for box, score, label in predictions:
            if score > min_score:
                detections = {
                    'category': str(int(label)),
                    'conf': truncate_float(float(score), precision=3),
                    'bbox': self.set_bounding_box(box)
                }
                detection_results.append(detections)
                
        return detection_results

    def predict(self, image, img_names):
        batch = 0 # only using batch size of one
        np_im = np.asarray(image, np.uint8)
        (boxes, scores, classes) = self.sess.run([self.boxes, self.scores, self.classes],
                                                 feed_dict={self.image_tensor: np.expand_dims(np_im, axis=0)})

        detection_results = self.process_boxes(boxes[batch], classes[batch], scores[batch])

        # log_prediction_json image_detection_results
        return {'file_name': img_names,
                'detections':detection_results}