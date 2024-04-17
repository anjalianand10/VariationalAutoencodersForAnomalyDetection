import numpy as np
import tensorflow as tf
from datetime import datetime

def sample(args):
    #
    # Reparameterization trick by sampling from an isotropic unit Gaussian.
    #
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)

def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def is_anomaly(timestamp, anomaly_ranges):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    for start, end in anomaly_ranges:
        start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")
        end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S.%f")
        if start <= timestamp <= end:
            return True
    return False