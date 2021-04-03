# %% import packages
# from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import adanet
import os
from sklearn.metrics import r2_score, accuracy_score

# define train/evaluate params
max_steps = 3000

# %% define data
validation_size = 32
max_rows = 1000
target_col = "target"
target_ATP = "KOSPIb1f0"
dataset_id = "bNsCFCCOFOC"
data_folder = "20210204_bNsCFCCOFOC/"

root_folder = "/home/lstm/Insync/jaesangpark@gmail.com/Google Drive/Data Exchange/autoML_train_test_data/"
output_folder = "/home/lstm/Desktop/adanet_output/" + target_ATP + dataset_id + "/"

# root_folder = r"C:\Users\jp\Google Drive\Data Exchange\autoML_train_test_data\\"
# output_folder = r"C:\Users\jp\Downloads\\"+target_ATP+dataset_id+"\\"

datalist = pd.DataFrame(os.listdir(root_folder + data_folder))
filter_trainvalid = datalist[0].str.contains(target_ATP) & datalist[0].str.contains("trainvalid")
filter_test = datalist[0].str.contains(target_ATP) & datalist[0].str.contains("test")
datalist_trainvalid = datalist.loc[filter_trainvalid, :].sort_values(by=0).reset_index(drop=True)
datalist_test = datalist.loc[filter_test, :].sort_values(by=0).reset_index(drop=True)

# check name congruency
for i, j in zip(datalist_trainvalid.iterrows(), datalist_test.iterrows()):
    data_name = i[1][0][0:20]
    if not data_name in j[1][0]:
        raise ValueError('data name mismatch between trainvalid set and test set')

# %% loop through each dataset
metrics = pd.DataFrame({"corr": [np.nan], "r2": [np.nan], "acc": [np.nan]}, index=[""])
for i, j in zip(datalist_trainvalid.iterrows(), datalist_test.iterrows()):

    # read in data
    tvset_name = i[1][0]  # tvset_name = datalist_trainvalid[0][0]
    testset_name = j[1][0]  # testset_name = datalist_test[0][0]
    trainset = pd.read_csv(root_folder + data_folder + tvset_name, \
                           parse_dates=['timestamp'], index_col="timestamp").iloc[:-validation_size]
    validset = pd.read_csv(root_folder + data_folder + tvset_name, \
                           parse_dates=['timestamp'], index_col="timestamp").iloc[-validation_size:]
    testset = pd.read_csv(root_folder + data_folder + testset_name, \
                          parse_dates=['timestamp'], index_col="timestamp")

    # define sample weight & attach to features
    weight_factor = 0.994
    ww_train = list(range(0, len(trainset)))
    wv_train = ww_train
    for wi, w in enumerate(ww_train):
        wv_train[wi] = 1 - (((1 - weight_factor) * (weight_factor ** ww_train[wi])) / (1 - weight_factor))
    # plt.plot(wv_train)
    # plt.show()

    wv_valid = [wv_train[-1]] * len(validset)  # weights for validation set
    wv_test = [wv_train[-1]] * len(testset)  # weights for validation set

    trainset['weight'] = wv_train
    validset['weight'] = wv_valid
    testset['weight'] = wv_test

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
    print("\n" + data_ref)

    print("\n train set: " + str(len(trainset)) + " rows")
    print(trainset.iloc[:, 0:5].head())
    print(trainset.iloc[:, 0:5].tail())

    print("\n validation: " + str(len(validset)) + " rows")
    print(validset.iloc[:, 0:5].head())
    print(validset.iloc[:, 0:5].tail())

    print("\n test set: " + str(len(testset)) + " rows")
    print(testset.iloc[:, 0:5].head())
    print(testset.iloc[:, 0:5].tail())

    # separate feature / target & split validation set
    x_train = trainset
    y_train = x_train.pop(target_col)

    x_valid = validset
    y_valid = x_valid.pop(target_col)

    x_test = testset
    y_test = x_test.pop(target_col)

    # define input function
    def input_fn(features, labels):
        """input function for train, eval, & predict"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(1)
        return dataset

    def test_input_fn():
        """input function for train, eval, & predict"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(x_test))).batch(1)
        return dataset

    # define feature columns
    feature_columns = []

    numeric_cols = x_train.keys()[(x_train.keys() != 'DoW') & (x_train.keys() != 'weight')]
    for key in numeric_cols:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # x_train.drop(columns=['DoW'], inplace=True)
    # x_valid.drop(columns=['DoW'], inplace=True)

    def one_hot_cat_column(feature_name, vocabulary):
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
        )

    cat_cols = x_train.keys()[x_train.keys() == 'DoW']
    for feature_name in cat_cols:
        # one-hot encode categorical features.
        vocabulary = x_train[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

    # define regression head
    head = tf.estimator.RegressionHead(
        weight_column='weight'
        # loss_fn='mean_squared_error'
    )

    # initiate adanet evaluator
    # eval_input_func = input_fn(x_valid, y_valid)
    evaluator = adanet.Evaluator(
        input_fn=lambda: input_fn(x_valid, y_valid),
        # metric_name='adanet_loss',
        # objective='minimize',
        steps=max_steps
    )

    # initiate adanet estimator
    model = adanet.AutoEnsembleEstimator(
        head=head,
        candidate_pool={
            "linear":
                tf.estimator.LinearEstimator(
                    head=head,
                    feature_columns=feature_columns,
                    config=None
                    # optimizer=...
                ),
            "dnn":
                tf.estimator.DNNEstimator(
                    head=head,
                    hidden_units=[64, 64, 64],
                    feature_columns=feature_columns,
                    # optimizer='Adagrad',
                    # activation_fn=tf.nn.relu,
                    dropout=0.2,
                    config=None,
                    warm_start_from=None,
                    batch_norm=True
                )
            # "boostedtree":
            #     tf.estimator.BoostedTreesEstimator(
            #         head=head,
            #         feature_columns=feature_columns,
            #         n_batches_per_layer = 1
            #         # n_trees=100,
            #         # max_depth=6,
            #         # learning_rate=0.1,
            #         # l1_regularization=0.0,
            #         # l2_regularization=0.0,
            #         # tree_complexity=0.0,
            #         # min_node_weight=0.0,
            #         # config=None,
            #         # center_bias=False,
            #         # pruning_mode='none',
            #         # quantile_sketch_epsilon=0.01
            #     )
        },
        max_iteration_steps=max_steps,
        force_grow=False,
        evaluator=evaluator,
        model_dir=output_folder + data_ref + "/"
    )

    # # define callback (early stop)
    # early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
    #     estimator=model, metric_name="loss", max_steps_without_decrease=max_steps_without_decrease)

    # train model
    model.train(
        input_fn=lambda: input_fn(x_train, y_train),
        # hooks=[early_stopping_hook],
        steps=max_steps
    )

    model.evaluate(input_fn=lambda: input_fn(x_valid, y_valid), steps=None)

    # predict with test set
    pred =

    # plot
    axis_array = y_test.index.to_frame()['timestamp']
    divider_idx = len(axis_array)
    plot_engine = 'plotly'  # or 'matplotlib'

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
        pd.DataFrame({"corr": [corr_], "r2": [r2_], "acc": [acc_]}, index=[data_ref])
    )
    metrics.dropna(inplace=True)
    metrics.to_csv(output_folder + "metrics.csv")

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
