#!/usr/bin/env python3

import tensorflow as tf


class Yolo:

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Yolo class constructor"""
        self.model = tf.keras.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
