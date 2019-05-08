from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd


def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss


def get_model_data():
    graph = tf.get_default_graph()

    workpath = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workpath, 'KhanModel')
    
    FRmodel = load_model(model_path, custom_objects={"triplet_loss":triplet_loss})  

    csv_path = os.path.join(workpath, 'model_encoding.csv')

    if 1==1:
        df = pd.read_csv(csv_path)
        df["fn_encoding"] = df["str_encoding"].map(lambda x:np.array(x.split("^")).reshape(1,-1).astype(np.float32))
    else:
        df = load_data(FRmodel)
        df["str_encoding"] = df["fn_encoding"].map(lambda x: "^".join(list(x.flatten().astype(str))))
        df[["class_name","str_encoding"]].to_csv(cvs_path, index=False)
    return graph, FRmodel, df

graph, FRmodel, df = get_model_data()

def get_encoding(img_file, FRmodel):
    global graph
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

