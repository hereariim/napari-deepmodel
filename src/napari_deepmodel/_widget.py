"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget


from enum import Enum
from subprocess import call
import numpy
import napari
from napari.types import ImageData
import cv2
from magicgui import magic_factory, magicgui
from skimage import data
import sys

import numpy
import tensorflow as tf
from tensorflow import keras
from focal_loss import BinaryFocalLoss
from tensorflow.keras import backend as K
import numpy as np
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from skimage.io import imread
from napari_plugin_engine import napari_hook_implementation
from enum import Enum
from napari.utils.notifications import show_info

class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


def do_image_segmentation(
    layer: ImageData
    ) -> ImageData:
    
    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)   
        size_ = image.shape
        img = image[:,:,:3]
        X[0] = resize(img, (256, 256), mode='constant', preserve_range=True)
        return X,size_

    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour ??viter la division par 0 
    
    image_reshaped,size_ = redimension(layer)

    model_new = tf.keras.models.load_model("C:\\Users\\Metuarea Herearii\\napari-deepmodel\\src\\napari_deepmodel\\best_model_FL_BCE_0_5.h5",custom_objects={'dice_coefficient': dice_coefficient})
    prediction = model_new.predict(image_reshaped)
    preds_test_t = (prediction > 0.2).astype(np.uint8)
    temp=np.squeeze(preds_test_t[0,:,:,0])*255
    
    return cv2.resize(temp, dsize=(size_[1],size_[0]))

@magic_factory(call_button="Run")
def do_model_segmentation(
    layer: ImageData, radio_option=1
    ) -> ImageData:
    show_info('Succes !')
    return do_image_segmentation(layer)