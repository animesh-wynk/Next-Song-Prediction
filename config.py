# ! pip install python-snappy
# ! pip install pyarrow

import os, datetime, pytz, sys
from pprint import pprint
import tensorflow as tf

def q(msg=""):
    print(f">{msg}<")
    sys.exit()

def get_timestamp():
    '''
    format of time (str) returned: YYYY_MM_DD_HHMMSS
    '''
    # it will get the time zone of the specified location 
    IST = pytz.timezone("Asia/Kolkata")   
    timestamp = str(datetime.datetime.now(IST))[:19].replace(":", "").replace("-", "_").replace(" ", "_")
    return timestamp

REDIRECT_STD_OUT_TO_TXT_FILE = True 
USE_TIME_BUCKETS = True
USE_ENCODER_DECODER_ARCH = True

LR_DECAY_RATE = 0.50 # Set None if LR decay is not to be employed
LR_DECAY_EVERY_N_BATCH = 120_000

# DATA
DATA_BASE_PATH = "s3://wynk-ml-workspace/projects/_temp/rnn_recommendation/dataset_5/day=2021-06-01to2021-07-04/"

TRAIN_DATA_DIR_PATH = "train/"
VAL_DATA_DIR_PATH = "val_test/"
TEST_DATA_DIR_PATH = "val_test/"

# TRAIN DATA
TRAIN_DATA_PATH = (DATA_BASE_PATH + TRAIN_DATA_DIR_PATH + 
                         "_09_train_data_split_and_pad_seq_csv/part-00000-8de395ed-92ba-414a-abe5-76bbaf6de444-c000.csv")   

# TRAIN_DATA_PATH = "s3://wynk-ml-workspace/projects/_temp/rnn_recommendation/with_metadata/day=2021-05-16to2021-05-22/debug/part-00000-2c072854-c833-48db-bdd6-52cf1c229755-c000.csv"


TRAIN_SONGS_INFO_PATH = (DATA_BASE_PATH + TRAIN_DATA_DIR_PATH + 
                         "_10_train_data_song2info/part-00000-068ee48a-dc3e-4c95-a239-bcaf20b81820-c000.snappy.parquet")   
TRAIN_USERS_INFO_PATH = (DATA_BASE_PATH + TRAIN_DATA_DIR_PATH + 
                         "_06_train_data_user_emb_ids/part-00000-fd09ed94-da59-4b23-b562-a7476dd5014e-c000.snappy.parquet")

# VAL DATA (data from same day)
VAL_DATA_PATH = (DATA_BASE_PATH + VAL_DATA_DIR_PATH + 
                   "_04_val_data_path/part-00000-9c8dc6db-53da-458f-a1aa-1869a9938a10-c000.csv")

# TEST DATA (data from same day)
TEST_DATA_PATH = (DATA_BASE_PATH + TEST_DATA_DIR_PATH + 
                   "_04_test_data_path/part-00000-7232f3ba-b3fd-4a8f-8162-1ec634f18fdf-c000.csv") 

# OTHER DATA PATHS
QUALITATIVE_TEST_DATA_PATH = "qualitative_test_data/"
PICKLES_DIR_PATH = "pickles/"

SONG2INFO_PICKLE_PATH = os.path.join(PICKLES_DIR_PATH, "song2info.pickle")
BUILD_VOCAB_DICT_PATH = os.path.join(PICKLES_DIR_PATH, "build_vocab.pickle")

QUALITATIVE_TEST_DATA_PATH  = QUALITATIVE_TEST_DATA_PATH + "song_id_dataset_for_qualitative_assessment_of_rnn_model.csv"


### MULTI-GPU CONFIG
MAX_REPLICAS_DESIRED = 8
PHYSICAL_DEVICES = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(PHYSICAL_DEVICES[:MAX_REPLICAS_DESIRED], "GPU")

# Define distributed tf strategy
# STRATEGY = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])

'''
communication_options = tf.distribute.experimental.CommunicationOptions(
                        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
STRATEGY = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
'''

STRATEGY = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# STRATEGY = tf.distribute.MultiWorkerMirroredStrategy(['/gpu:0', '/gpu:1']) # does not work 

NUM_REPLICAS = STRATEGY.num_replicas_in_sync

# Define BATCH_SIZE_PER_REPLICA and BATCH_SIZE
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS # GLOBAL_BATCH_SIZE
# BATCH_SIZE = 64                      # batch size


### MODEL ARCHITECTURE CONFIG
LSTM_DIM = 2048#1024                      # output size of the rnn layer
SONG_EMB_DIM = 64
TIME_BUCKET_EMB_DIM = 8
USER_EMB_DIM = 64
TIME_BUCKET_VOCAB_SIZE = 12 + 1 # 228 buckets of 5 mins + padding
MAX_LEN = 10                         # maximum length of the input sequece to be fed inside the model while training 
MAX_TEST_SEQ_LEN = 60

SONG_PAD_TOKEN = "<pad>"
# SONG_UNK_TOKEN = "<unk>"

SONG_PAD_INDEX = 0
# SONG_UNK_INDEX = 1

### OTHER CONFIG
POPULAR_SONGS_PERCENTAGE = 5 # top POPULAR_SONGS_PERCENTAGE% songs sorted on frequency, to make the list of popular songs   

BATCH_NUM_START = 0

START_EPOCH = 0
END_EPOCH = 2                                                              # number of epochs for training the model

METRICS_EVALUATION_AND_SAVE_MODEL_EVERY_N_BATCH = 25_000                 # metrics evaluation on the test-set and saving model's weights happens after every METRICS_EVALUATION_EVERY_N_BATCH batches
SHOW_LOSS_EVERY_N_BATCH = 5_000                                          # training loss is printed after every SHOW_LOSS_EVERY_N_BATCH batches
WRITE_SUMMARY = True                                                    # whether to write summary on tensorboard or not 

NUM_TEST_SAMPLES_QUANTITATIVE = 10_000                                   # number of testing examples to be used for evaluating all the metrics 

NAME = f"exp23_withtimebuckets_rnn_7_day_data_{get_timestamp()}_{LSTM_DIM}_{SONG_EMB_DIM}_{MAX_LEN}"

SUMMARY_DIR = os.path.join("summary", "summary_"+NAME )
METRICS_SUMMARY_DIR = os.path.join("metrics", "metrics_summary_"+ NAME)
MODEL_DIR = os.path.join("models", "models_"+NAME)
LOG_DIR = os.path.join("logs", "logs_"+NAME)

LOG_DIR = os.path.join("logs", "logs_"+NAME)
QUALITATIVE_RESULTS_DIR = os.path.join("qualitative_results", "qualitative_results_" + NAME)

dirs_to_make_list = [LOG_DIR, QUALITATIVE_RESULTS_DIR, PICKLES_DIR_PATH]
for d in dirs_to_make_list:
    if not os.path.exists(d):
        os.makedirs(d) 
    
LOAD_MODEL = False
LOAD_MODEL_PATH = "models/models_withtimebuckets_rnn_7_day_data_2021_06_29_200418_1024_64_10/model_02_124999"

### RECOMMENDATIONS CONFIG
K = 10
NUM_RECOMMENDATION_TIMESTEPS = 10
QUALITATIVE_RESULTS_ON_HANDPICKED_SONGS = True
PRINT_QUALITATIVE_RESULTS = not True
WRITE_QUALITATIVE_RESULTS = True
