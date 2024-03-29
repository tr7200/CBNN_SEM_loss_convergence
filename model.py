"""

2021 ASA SDSS paper code

Code structure follows DAG structure, 
2 V-structures nested together with immoralities

"""

import os
import math
import random
import logging

import numpy as np 
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler


def main(independent_vars: str=None,
         dependent_var: str=None):
    """Training loop for main result - Figure 1"""
    independent_variables = pd.read_csv(, sep=',', header=0).astype('float32')
    firm_performance = pd.read_csv(, sep=',', header=0)
    
    kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=12345)

    val_losses_df = pd.DataFrame()
    losses_df = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kfold.split(independent_variables)):
        X_train, X_test = independent_variables.iloc[train_index,], independent_variables.iloc[test_index,]
        y_train, y_test = firm_performance.iloc[train_index,], firm_performance.iloc[test_index,]

        X_train = MinMaxScaler(copy=False).fit_transform(X_train)
        X_train = pd.DataFrame(data=X_train)
        SM = np.array(X_train.iloc[:,:9])
        BD = np.array(X_train.iloc[:,21:])
        AS = np.array(X_train.iloc[:,9:21])

        sm_input = Input(shape=(9,))
        bd_input = Input(shape=(39,))
        as_input = Input(shape=(12,))

        krnl_dvrgnc_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (323 * 1.0)


        # outer chevron
        sm_bd_combined = concatenate([sm_input, bd_input])
        sm_bd_combined_out = tfp.layers.DenseFlipout(48, activation='relu', kernel_divergence_fn=krnl_dvrgnc_fn)(sm_bd_combined)

        as_and_sm_bd_combined = concatenate([sm_bd_combined_out, as_input])
        as_and_sm_bd_combined_out = tfp.layers.DenseFlipout(3, activation='relu', kernel_divergence_fn=krnl_dvrgnc_fn)(as_and_sm_bd_combined)
        V_struct_1_out  = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=25 + t[..., :1],
                                                        validate_args=True,
                                                        allow_nan_stats=False,
                                                        scale=2 + tf.math.softplus(0.01 * t[..., 1:])))(as_and_sm_bd_combined_out)

        V_struct_1 = Model([sm_input, bd_input, as_input], V_struct_1_out)

        V_struct_1_kl = sum(V_struct_1.losses)

        negloglik_V_struct_1 = lambda y, p_y: -p_y.log_prob(y) + V_struct_1_kl

        V_struct_1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                           loss=negloglik_V_struct_1)


        # inner chevron
        V1_out = V_struct_1([sm_input, bd_input, as_input])
        V1_SM_BD_combined = concatenate([V1_out, bd_input, sm_input])
        V1_SM_BD_combined_out1 = tfp.layers.DenseFlipout(37, activation='relu', kernel_divergence_fn=krnl_dvrgnc_fn)(V1_SM_BD_combined)
        V1_SM_BD_combined_out2 = tfp.layers.DenseFlipout(10, activation='relu', kernel_divergence_fn=krnl_dvrgnc_fn)(V1_SM_BD_combined_out1)
        V2_out = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=25 + t[..., :1], 
                                               validate_args=True, 
                                               allow_nan_stats=False,
                                               scale=2 + tf.math.softplus(0.01 * t[..., 1:])))(V1_SM_BD_combined_out2)

        V_struct_2 = Model([sm_input, bd_input, as_input], V2_out)

        V_struct_2_kl = sum(V_struct_2.losses)

        negloglik_V_struct_2 = lambda y, p_y: -p_y.log_prob(y) + V_struct_2_kl 

        losses = lambda y, p_y: negloglik_V_struct_1(y, p_y) + negloglik_V_struct_2(y, p_y)

        V_struct_2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
                           loss=losses)



        result = V_struct_2.fit([SM, BD, AS],
                                 y_train, 
                                 epochs=50,
                                 batch_size=4,
                                 verbose=1,
                                 validation_split=0.05)

        return result
        
        
        
if __name__ == '__main__':
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    SEED = 1387
    random.seed(SEED)
    np.random.seed(SEED)

    INDPENDENT_VARS = 'independent_variables.csv'
    DEPENDENT_VAR = 'firm_perf_total.csv'

    tf.reset_default_graph()
    
    result = main(independent_vars=INDEPENDENT_VARS,
                  dependent_var=DEPENDENT_VAR)
    
    # ... plots
