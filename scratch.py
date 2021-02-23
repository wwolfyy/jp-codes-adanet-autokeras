# %% import packages
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import os
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.models import load_model

# %% define path
target_ATP = "KGBfub1f0"
dataset_id = "bNsCFOCOFOC"
output_folder = "/home/lstm/Desktop/AutoKeras_output/"+target_ATP+dataset_id+"/"

# %% load model
modellist = pd.DataFrame(os.listdir(output_folder))
filt = modellist[0].str.startswith(target_ATP)
modellist = modellist.loc[ filt, : ].sort_values(by=0).reset_index(drop=True)

for i in modellist.iterrows():
    # tf.saved_model.load(output_folder+i[1][0]+"/best_model")
    print(i)
    loaded_model = load_model(output_folder+i[1][0]+"/best_model/", custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.summary()


# %%
