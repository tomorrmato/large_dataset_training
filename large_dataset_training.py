import glob
import os
import time

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import torch
from nvidia.dali import pipeline_def
from nvidia.dali.fn import readers
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam

from config import *

########################    DALI pipeline    ########################

@pipeline_def
def make_dali_dataloader(shard_id, num_shards):
    out = readers.tfrecord(
            path=sorted(glob.glob(f'{DATA_DIR}/*.tfrecord')),
            index_path=sorted(glob.glob(f'{DATA_DIR}/*.index')),
            features={
                "features": tfrec.VarLenFeature(tfrec.int64, 0),
                "label": tfrec.VarLenFeature(tfrec.int64, 0)},
            name='tfrecord_reader',
            num_shards=num_shards,
            shard_id=shard_id,
        )

    return (fn.reshape(out["features"], [SEQUENCE_LEN]), fn.reshape(out["label"], [-1,1]))


########################  pytorch lightning module with DALI dataloader    ########################

class TestMode(LightningModule):

    def __init__(self):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    def forward(self, **input):        
        return self.bert(**input)

    def process_batch(self, batch):
        x, y = batch
        return x,y 

    def training_step(self, batch, batch_idx):
        x, y = batch 
        output = self(**{"input_ids": x, "labels":y})
        return output.loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        device_id = self.local_rank # if you dali pipeline on CPU, this should be changed to None!
        shard_id = self.global_rank
        num_shards = self.trainer.world_size 
        
        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] for k in self.output_map]

        # in DDP training, init dataloader for each shard 
        _pipeline = make_dali_dataloader(
            batch_size=BATCH_SIZE, num_threads=1, 
            shard_id=shard_id, num_shards=num_shards, 
            device_id=device_id)

        dali_data_loader = LightningWrapper(
            _pipeline,
            output_map=["features", "label"], 
            reader_name='tfrecord_reader', 
            auto_reset=True, 
            last_batch_policy=LastBatchPolicy.PARTIAL
        )

        return dali_data_loader


# call pytorch lightning trainer 
model = TestMode()
trainer = Trainer(
    gpus=torch.cuda.device_count(), 
    strategy="ddp",
    min_epochs=1,
    max_epochs=1,
)
trainer.fit(model)
