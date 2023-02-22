# this is an example script to write test data 

import os
import subprocess
from typing import List, Union

import numpy as np
import tfrecord

from config import *

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# write mnist data as tfrecords by multiprocessing 
def _convert_to_1d_list(x: Union[np.ndarray, List]):
    if isinstance(x, np.ndarray):
        return x.flatten().tolist()
    elif isinstance(x, list):
        return np.array(x).flatten().tolist()
    else:
        return [x]


def convert_numpy_to_tfrecod(
    feature_array: np.ndarray, 
    label_array: np.ndarray, 
    file_index: int, 
    file_pattern: str, 
    index_pattern: str
):
    tfrecord_path = file_pattern.format(file_index)
    index_path = index_pattern.format(file_index)
    writer = tfrecord.TFRecordWriter(tfrecord_path)
    
    for f,l in zip(feature_array, label_array): 
        # write tfrecord files by tfrecord library 
        row_record = {"features": (_convert_to_1d_list(f), "int"),
                     "label": (_convert_to_1d_list(l), "int")}
        writer.write(row_record)

    writer.close()
    subprocess.call(["python3", "-m", "tfrecord.tools.tfrecord2idx", tfrecord_path, index_path.format(file_index)])


def make_fake_data():
    np.random.seed(1)
    features = np.random.randint(low=1100, high=3000, size=(SAMPLE_SIZE, SEQUENCE_LEN-2))
    features = np.hstack([np.ones(shape=(SAMPLE_SIZE,1), dtype=np.int64)*101, features, np.ones(shape=(SAMPLE_SIZE,1), dtype=np.int64)*102])
    labels = np.random.choice([0,1,2,3,4], size=(SAMPLE_SIZE))
    return features, labels



# write the datasets
fake_features, fake_labels = make_fake_data()
tfrecord_pattern = os.path.join(DATA_DIR, "data_partition_{}.tfrecord")
index_pattern = os.path.join(DATA_DIR, "data_index_{}.index")

convert_numpy_to_tfrecod(
    fake_features, 
    fake_labels, 
    0, 
    tfrecord_pattern, 
    index_pattern
)


