import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
from model import Model
from constants import *
from glob import glob
from utils import *
from tqdm import tqdm

# dataset processing
DATA_TRAIN = glob(DIR_TFRECORDS+'/*.tfrecords')
dataset_train = tf.data.TFRecordDataset(DATA_TRAIN)
dataset_train = dataset_train.map(parse_record)
dataset_train = dataset_train.shuffle(buffer_size=1024)
dataset_train = dataset_train.batch(BATCH_SIZE)

# define model
with tf.device(DEVICE):
    model = Model()
    print('Model created')
    #model.train(dataset_train)
    #model.load_pretrained_weights('weights_YOLO')
    checkpoint = tfe.Checkpoint(model=model, optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print('Checkpoint loaded')
    
# train
with tf.device(DEVICE):
    model.optimizer = tf.train.AdamOptimizer(1e-6)
    for i in range(1):
        loss = model.train(dataset_train)
        print('Epoch:{} | Loss={}'.format(i, loss))
        
        if i % 10 == 0:
            # save checkpoint
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
            print('checkpoint saved.')


