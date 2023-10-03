import tensorflow as tf
import sys
from PIL import Image, ImageDraw
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


class TLClassifier(object):
    def __init__(self, model_path):

        self.graph = tf.Graph()

        # Set the newly created graph as the default
        with self.graph.as_default():
            # Load the pre-trained model's graph definition
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                # Read and parse the serialized graph from the model file
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                # Import the parsed graph definition into the current graph
                tf.import_graph_def(od_graph_def, name='')
        
        self.image = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        img = np.zeros((100, 100, 3))
        self.detect_multi_object(img, 0.1)
        self.traffic_light_box = None
        self.classified_index = 0

    def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        image_np_expanded = np.expand_dims(image_np, axis=0)

        output = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], 
                feed_dict={self.image_tensor: image_np_expanded})
        
        squeezed_scores = np.squeeze(output[1])
        squeezed_classes = np.squeeze(output[2])
        squeezed_boxes = np.squeeze(output[0])
    
        selection_mask = np.logical_and(squeezed_classes == 10, squeezed_scores > score_threshold)
    
        return squeezed_boxes[selection_mask]


def crop_roi_image(img, box):
    h, w, _ = img.shape
    l = int(box[1]*w)
    r = int(box[3]*w)
    t = int(box[0]*h)
    b = int(box[2]*h)
    cropped = img[t:b, l:r, :]
    return cropped



def make_bounding_boxes(img, y, x, Y, X)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    l = x*w
    r = X*w
    t = y*h
    b = Y*h
    draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)],width=5, fill='black')

