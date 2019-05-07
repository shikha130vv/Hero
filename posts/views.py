from django.shortcuts import render, redirect 
import pandas as pd
from .forms import *
from django.http import HttpResponse
from PIL import Image
import numpy as np

from keras.models import load_model

import tensorflow as tf
from keras import backend as K
import os
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response

from rest_framework.generics import ListAPIView, RetrieveAPIView, CreateAPIView
from rest_framework.permissions import IsAdminUser

from .serializer import *
from django.conf import settings

from .keras_yolo import yolo_head, yolo_boxes_to_corners
global graph
graph = tf.get_default_graph()

class ImageCreate(CreateAPIView):
    serializer_class = ImageSerializer

    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            #image_string = serializer.fields["image"].file.read()
            #img = Image.open(image_string)
            #arr = np.asarray(img)

            serializer.save()
            image_path = serializer.data.get('image')[1:]
            
            image_abs_path = os.path.join(settings.BASE_DIR, image_path)
            #cropped_image = get_cropped_image(image_path)
            img_class = get_image_class(image_path)
            return HttpResponse('<img width=300  src="' + image_path + '"><br>' + img_class)
            return HttpResponse('<img src="' + "not ok" + '"><br>' )



def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def get_model_data():
    
    workpath = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workpath, 'KhanModel')
    yolo_model_path = os.path.join(workpath, 'yolo.h5')
    class_path = os.path.join(workpath, 'coco_classes.txt')
    anchor_path = os.path.join(workpath, 'yolo_anchors.txt')

    FRmodel = load_model(model_path, custom_objects={"triplet_loss":triplet_loss})  

    yolo_model = None#load_model(yolo_model_path)

    class_names = read_classes(class_path)

    anchors = read_anchors(anchor_path)

    csv_path = os.path.join(workpath, 'model_encoding.csv')

    if 1==1:
        df = pd.read_csv(csv_path)
        df["fn_encoding"] = df["str_encoding"].map(lambda x:np.array(x.split("^")).reshape(1,-1).astype(np.float32))
    else:
        df = load_data(FRmodel)
        df["str_encoding"] = df["fn_encoding"].map(lambda x: "^".join(list(x.flatten().astype(str))))
        df[["class_name","str_encoding"]].to_csv(cvs_path, index=False)
    return FRmodel, yolo_model, class_names, anchors , df


FRmodel, yolo_model, class_names, anchors, df = get_model_data()
yolo_org_image_shape = (720., 1280.)


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = box_confidence*box_class_probs
    box_classes = tf.arg_max(box_confidence*box_class_probs, dimension=-1)
    box_class_scores = tf.reduce_max(box_confidence*box_class_probs, axis=-1)
    filtering_mask = box_class_scores > threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    nms_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, \
                                               max_output_size=max_boxes, iou_threshold=iou_threshold)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[0], yolo_outputs[1],yolo_outputs[2],yolo_outputs[3]
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    return scores, boxes, classes


def process_boxes(image, out_scores, out_boxes, out_classes, class_names):
    thickness = (image.size[0] + image.size[1]) // 300
    cropped_img = image
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        if predicted_class == "person":
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            cropped_img = np.array(image)[top+1:bottom-1, left+1:right-1]
            cropped_img = Image.fromarray(cropped_img)
    return cropped_img  

def preprocess_yolo_image(img_path, org_image_size, model_image_size):
    image = Image.open(img_path)
    image = image.resize(tuple(reversed(org_image_size)), Image.BICUBIC)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def get_cropped_image(path):
	with graph.as_default():
		sess = K.get_session()
		image, image_data = preprocess_yolo_image(path, (720, 1280), (608, 608))

		yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
		scores, boxes, classes = yolo_eval(yolo_outputs, yolo_org_image_shape)

		out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase(): 0})

		cropped_img = process_boxes(image, out_scores, out_boxes, out_classes, class_names)

	return cropped_img


def get_encoding(img_file, FRmodel):
    pic = Image.open(img_file)
    img = pic.resize((96,96), Image.ANTIALIAS)
    img = np.array(img)
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    with graph.as_default():
    	encoding = FRmodel.predict_on_batch(x_train)
    return encoding

def get_image_class(img_file): 
    global FRmodel

    encoding = get_encoding(img_file, FRmodel)

    df["dist"] = df["fn_encoding"].apply(lambda x: np.linalg.norm(encoding- x))
    idx = df.sort_values("dist").index.values[0]
    idx1 = df.sort_values("dist").index.values[1]
    pred_class = df.loc[idx, "class_name"]
    pred_dist = df.loc[idx, "dist"]
    tail_dist = df.loc[idx1, "dist"]
    pred_class1 = df.loc[idx, "class_name"]
    return pred_class
    #return "Prediction:" + pred_class + " Distance:" + str(pred_dist) + " Next Prediction:" + pred_class1 + " Distance:" + str(tail_dist)


