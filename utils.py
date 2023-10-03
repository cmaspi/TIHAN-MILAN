import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from glob import glob
import sys
from PIL import Image, ImageDraw


model_path = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'



def load_detection_graph(model_path):
    # Create a new TensorFlow graph for object detection
    detection_graph = tf.Graph()

    # Set the newly created graph as the default
    with detection_graph.as_default():
        # Load the pre-trained model's graph definition
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            # Read and parse the serialized graph from the model file
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            # Import the parsed graph definition into the current graph
            tf.import_graph_def(od_graph_def, name='')

    # Return the loaded detection graph
    return detection_graph


def select_filtered_boxes(boxes,
                          classes,
                          scores,
                          score_threshold=0,
                          target_class=10):
    # Squeeze the arrays to remove unnecessary dimensions
    squeezed_scores = np.squeeze(scores)
    squeezed_classes = np.squeeze(classes)
    squeezed_boxes = np.squeeze(boxes)
    
    # Create a boolean mask to select boxes based on class and score criteria
    selection_mask = np.logical_and(squeezed_classes == target_class,
                                     squeezed_scores > score_threshold)
    
    # Use the mask to select the relevant boxes
    selected_boxes = squeezed_boxes[selection_mask]
    
    return selected_boxes


class TLClassifier(object):
    def __init__(self):
        self.detection_graph = load_detection_graph(model_path)
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image, 0.1)
        self.traffic_light_box = None
        self.classified_index = 0

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sel_boxes = select_filtered_boxes(boxes=boxes, classes=classes, scores=scores,
                                 score_threshold=score_threshold, target_class=10)

        return sel_boxes


def crop_roi_image(image_np, sel_box):
    imh, imw, _ = image_np.shape
    left, right, top, bottom = (sel_box[1]*imw, sel_box[3]*imw,
                                sel_box[0]*imh, sel_box[2]*imh)
    cropped_image = image_np[int(top):int(bottom),
                             int(left):int(right), :]
    return cropped_image



def make_bounding_boxes(image,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        color='red',
                        thickness=4):
    draw = ImageDraw.Draw(image)
    imw, imh = image.size
    left, right, top, bottom = xmin*imw, xmax*imw, ymin*imh, ymax*imh
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness, fill=color)

