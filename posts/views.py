from django.shortcuts import render, redirect 
import pandas as pd
from .forms import *
from django.http import HttpResponse
from PIL import Image
import numpy as np

from keras.models import load_model
import tensorflow as tf
import os

def post_image_view(request): 
  
    if request.method == 'POST': 
        form = PostForm(request.POST, request.FILES) 
  
        if form.is_valid(): 
            objModel = form.save() 
            img_class = get_image_class(objModel.cover)
            return redirect('success/' + img_class) 
    else: 
        form = PostForm() 
    return render(request, 'post_image_form.html', {'form' : form}) 
  
  
def success(request, img_class): 
    return HttpResponse('Hero: ' + img_class) 
  
 
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

   
def load_data(FRmodel):
    workpath = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(workpath, 'Q1_TrainingData')
    classes = os.listdir(train_data_path)
    classes = list(set(classes) - {".DS_Store"})
    arr_df = []
    for i in range(len(classes)):
        class_path = classes[i]
        file_list = os.listdir(train_data_path + "/" + class_path)
        df = pd.DataFrame({"file":file_list})
        df["class"] = i
        df["class_name"] = classes[i]
        arr_df.append(df)

    train_data = pd.concat(arr_df)
    train_data = train_data.reset_index()
    train_data["fn_encoding"] = train_data.apply(lambda x: get_encoding(train_data_path + "/" + x["class_name"] + "/" + x["file"], FRmodel), axis=1)
    return train_data


def get_encoding(img_file, FRmodel):
    pic = Image.open(img_file)
    img = pic.resize((96,96), Image.ANTIALIAS)
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    encoding = FRmodel.predict_on_batch(x_train)
    return encoding

def get_image_class(img_file): 
    workpath = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workpath, 'KhanModel')
    FRmodel = load_model(model_path, custom_objects={"triplet_loss":triplet_loss})

    encoding = get_encoding(img_file, FRmodel)
    df = load_data(FRmodel)
    if 1==2:
         workpath = os.path.dirname(os.path.abspath(__file__))
         cvs_path = os.path.join(workpath, 'model_encoding.xls')
         df = pd.read_excel(cvs_path)
         df["fn_encoding"] = df["str_encoding"].map(lambda x:np.array(x.split("^")).reshape(1,-1).astype(np.float32))

    df["dist"] = df["fn_encoding"].apply(lambda x: np.linalg.norm(encoding- x))
    idx = df.sort_values("dist").head(1).index
    idx1 = df.sort_values("dist")[1:].head(1).index
    pred_class = df.loc[idx, "class_name"].values[0]
    pred_dist = df.loc[idx, "dist"].values[0]
    tail_dist = df.loc[idx1, "dist"].values[0]
    return pred_class + str(pred_dist) + str(idx) + str(tail_dist)




