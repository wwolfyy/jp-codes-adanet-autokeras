# %% import packages
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import os
from sklearn.metrics import r2_score, accuracy_score

# %% define parameters & data

# training parameters
epochs = 1000
max_trials = 500

# define data
validation_size = 32
max_rows = 1000
target_col = "target"
target_ATP = "KOSPIb1f1"
dataset_id = "bNsCFCCXFCC"
data_folder = "20210219_" + dataset_id + "/"

root_folder = "/home/lstm/Insync/jaesangpark@gmail.com/Google Drive/Data Exchange/autoML_train_test_data/"
output_folder = "/home/lstm/Desktop/AutoKeras_output/"+target_ATP+dataset_id+"/"

datalist = pd.DataFrame(os.listdir(root_folder+data_folder))
filter_trainvalid = datalist[0].str.contains(target_ATP) & datalist[0].str.contains("trainvalid")
filter_test = datalist[0].str.contains(target_ATP) & datalist[0].str.contains("test")
datalist_trainvalid = datalist.loc[ filter_trainvalid, : ].sort_values(by=0).reset_index(drop=True)
datalist_test = datalist.loc[ filter_test, : ].sort_values(by=0).reset_index(drop=True)

# check name congruency
for i,j in zip( datalist_trainvalid.iterrows(), datalist_test.iterrows() ):
    data_name = i[1][0][0:20]
    if not data_name in j[1][0]:
        raise ValueError('data name mismatch between trainvalid set and test set')


# %% loop through each dataset
metrics = pd.DataFrame({"corr":[np.nan], "r2":[np.nan], "acc":[np.nan]}, index=[""])
for i, j in zip(datalist_trainvalid.iterrows(), datalist_test.iterrows()):

    # read in data
    tvset_name = i[1][0]
    testset_name = j[1][0]
    trainset = pd.read_csv(root_folder + data_folder + tvset_name, \
                           parse_dates=['timestamp'], index_col="timestamp").iloc[:-validation_size]
    validset = pd.read_csv(root_folder + data_folder + tvset_name, \
                           parse_dates=['timestamp'], index_col="timestamp").iloc[-validation_size:]
    testset = pd.read_csv(root_folder + data_folder + testset_name, \
                          parse_dates=['timestamp'], index_col="timestamp")

    # convert data type
    trainset["DoW"] = trainset["DoW"].astype('category')
    validset["DoW"] = validset["DoW"].astype('category')
    testset["DoW"] = testset["DoW"].astype('category')

    # cut off if too long (train & test sets)
    if len(trainset) > max_rows:
        trainset = trainset.tail(max_rows)
    if len(testset) > max_rows:
        testset = testset.tail(max_rows)

    # define dataset reference
    data_ref = tvset_name.replace('trainvalid_', '').replace('.csv', '')

    # print data
    print("\n"+data_ref)

    print("\n train set: " + str(len(trainset)) + " rows")
    print(trainset.iloc[:,0:5].head())
    print(trainset.iloc[:,0:5].tail())

    print("\n validation: " + str(len(validset)) + " rows")
    print(validset.iloc[:,0:5].head())
    print(validset.iloc[:,0:5].tail())

    print("\n test set: " + str(len(testset)) + " rows")
    print(testset.iloc[:,0:5].head())
    print(testset.iloc[:,0:5].tail())

    # separate feature / target & split validation set
    x_train = trainset
    y_train = x_train.pop(target_col)

    x_valid = validset
    y_valid = x_valid.pop(target_col)

    x_test = testset
    y_test = x_test.pop(target_col)

    # initialize structured data regressor.
    reg = ak.StructuredDataRegressor(
        directory=output_folder,
        project_name=data_ref,
        overwrite=True,
        max_trials=max_trials # try n different models.
    )

    # fit model
    reg.fit(
        x_train,
        y_train,
        validation_data = (x_valid, y_valid),
        epochs=epochs,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)]
    )

    # predict with the best model & plot
    pred = pd.DataFrame(reg.predict(x_test), index=y_test.index).iloc[:,0]

    # evaluate the best model with testing data.
    print(reg.evaluate(x_test, y_test))

    # plot
    axis_array = y_test.index.to_frame()['timestamp']
    divider_idx = len(axis_array)
    plot_engine = 'plotly' # or 'matplotlib'

    import sys
    sys.path.append("/home/lstm/Github/jp-codes-python/autoML_py36")
    import jp_utils
    jp_utils.stem_plot_compare(  # FIXME
        axis_array=axis_array,
        divider_idx=divider_idx,
        array_a=y_test,
        name_a='target',
        array_b=pred,
        name_b='pred',
        title_prefix=data_ref,
        window_size=5,
        fig_height=7,
        sharex=True,  # bool
        alert=False,  # alert if no test set
        plot_engine=plot_engine  # matplotlib or plotly
    )

    # collect metrics in dataframe and write out to CSV
    corr_ = round(y_test.corr(pred), 2)
    r2_ = round(r2_score(y_test, pred), 2)
    acc_ = round(accuracy_score(np.sign(y_test), np.sign(pred)), 2)
    metrics = metrics.append(
        pd.DataFrame({"corr":[corr_], "r2":[r2_], "acc":[acc_]}, index=[data_ref])
    )
    metrics.dropna(inplace=True)
    metrics.to_csv(output_folder+"metrics_"+target_ATP+dataset_id+".csv")

# %% exort model
# model = reg.export_model()
# model.summary()
# # numpy array in object (mixed type) is not supported.
# # you need convert it to unicode or float first.
# model.predict(x_train)
# model.save(output_folder + data_ref, save_format="tf")


# %% load model
# from tensorflow.keras.models import load_model
# ​
# Custom_Objects = ak.CUSTOM_OBJECTS
# ​
# # If you have custom metrics add each metric accordingly
# # Custom_Objects["f1_score"] = f1_score
# ​
# loaded_model = load_model("Models/First_Run", custom_objects=Custom_Objects)
# ​
# print(loaded_model.evaluate(test_file_path, "survived"))

# %% create docker image
