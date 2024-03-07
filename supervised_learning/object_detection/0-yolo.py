#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """_summary_"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """_summary_

        Args:
            model_path (_type_): _description_
            classes_path (_type_): _description_
            class_t (_type_): _description_
            nms_t (_type_): _description_
            anchors (_type_): _description_
        """
        self.model = load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """
        Load the class names from a file
        """
        with open(classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [name.strip() for name in class_names]
        return class_names
