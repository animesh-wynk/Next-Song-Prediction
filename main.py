import os, shutil, pdb, time, sys
import numpy as np
import tensorflow as tf

from dataset import wynk_sessions_dataset
from model import rnn_reco_model
from metrics import compute_and_store_metrics
from config import *

def get_time_elapsed_in_h_m_s(time_elapsed_in_seconds):
    m, s = divmod(time_elapsed_in_seconds, 60)
    h, m = divmod(m, 60)
    return f"{str(int(h)).zfill(2)} h {str(int(m)).zfill(2)} m {str(int(s)).zfill(2)} s"

if REDIRECT_STD_OUT_TO_TXT_FILE:
    # Redirecting stdout and stderr to txt files
    sys.stdout = open(f"{os.path.join(LOG_DIR, 'stdout')}.txt", "w")
    sys.stderr = open(f"{os.path.join(LOG_DIR, 'stderr')}.txt", "w")

print(f"tf.__version__: {tf.__version__}")

strategy = STRATEGY
print(f"Number of devices: {strategy.num_replicas_in_sync}")

### Initialize dataset class object
dataset = wynk_sessions_dataset()

if WRITE_SUMMARY:
    # SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
    # the directory is removed, if it already exists
    if os.path.exists(SUMMARY_DIR):
        shutil.rmtree(SUMMARY_DIR)

    # os.makedirs(SUMMARY_DIR)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, "train"))
    test_summary_writer  = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, "test"))
    train_summary_counter = 0
else:
    test_summary_writer = None

### Define model and opt  
with strategy.scope():
    model = rnn_reco_model(dataset.vocab_size)
    
    if LOAD_MODEL:
        model.build(input_shape=(None, MAX_LEN))
        print(model.summary())
        model.load_weights(LOAD_MODEL_PATH)
        print(f"Model loaded: {LOAD_MODEL_PATH}")    
    
    optimizer = tf.keras.optimizers.Adam() 

    def compute_loss(model, y_batch, lstm):
        last_layer = model.layers[-1].weights
        
        # Notice that per_example_loss will have an entry per GPU
        # so in this case there'll be 2 -- i.e. the loss for each replica
        per_example_loss = tf.nn.sampled_softmax_loss(
                            weights=tf.transpose(last_layer[0]),
                            biases=last_layer[1],
                            labels=tf.expand_dims(y_batch, -1),
                            inputs=lstm,
                            num_sampled = 15,
                            num_classes = dataset.vocab_size,
                            num_true = 1,
                            remove_accidental_hits=True,
                            name = "sampled_softmax_loss"
                        )        # (bs, ) # here reduction is None
        
        # per_example_loss seems to be arrays for loss per example, for all the replicas
        # so if we have 2 GPUs, and the batch_size_per_replica = 4,
        # then per_example_loss will have xxx arrays of losses with xxx scalars in each of them 
        #tf.print('\nper_example_loss: ', per_example_loss)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE) 

    
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
#     tf.print('in distributed_train_step > ', per_replica_losses)#per_replica_losses.values)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_step(inputs):
    
    if USE_TIME_BUCKETS:
        song_emb_id_x_batch, song_emb_id_y_batch, time_bucket_emb_id_x_batch, time_bucket_emb_id_y_batch = inputs
    else:
        song_emb_id_x_batch, song_emb_id_y_batch = inputs
        time_bucket_emb_id_x_batch = None
    
    with tf.GradientTape() as tape:
        lstm = model(song_emb_inp=song_emb_id_x_batch,
                     time_bucket_emb_inp=time_bucket_emb_id_x_batch,
                     training=True)
        
        loss = compute_loss(model, song_emb_id_y_batch, lstm)   
        # loss seems to be the average loss for all the data points in a global batch   
        #tf.print('\nloss: ', loss)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #train_accuracy.update_state(labels, predictions)
    return loss    

train_gen_output_types = (tf.dtypes.int64, tf.dtypes.int64)
train_gen_output_shapes = ((None, MAX_LEN), (None,))
if USE_TIME_BUCKETS:
    train_gen_output_types += (tf.dtypes.uint8, tf.dtypes.uint8)
    train_gen_output_shapes+= ((None, MAX_LEN), (None,))
    
print("- - - TRAIN - - - ")  
best_metrics_dict = {'best_sps': -1,
                'best_recall': -1,
                'best_item_coverage': -1}

batch_num = BATCH_NUM_START
### Training loop
for e in range(START_EPOCH, END_EPOCH):
    print(f"EPOCH: {str(e+1).zfill(len(str(END_EPOCH)))}/{END_EPOCH}")
        
    # Initialize python data generator
    train_gen = dataset.preprocessed_data_generator
    
    # Convert python generator into tf data generator    
    train_gen = tf.data.Dataset.from_generator(
                                train_gen,
                                output_types=train_gen_output_types,
                                output_shapes=train_gen_output_shapes 
                                )    
    # Prefetch data
    train_gen = train_gen.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Make tf data generator distributable
    train_dist_dataset = strategy.experimental_distribute_dataset(train_gen)

    total_loss = 0
    tick = time.time()
    for batch_idx, batch in enumerate(train_dist_dataset): 
        
        """
        # LR decay
        if (batch_num!=0) and (batch_num%50_000==0):
            old_lr = optimizer.learning_rate.numpy()
            optimizer.learning_rate = optimizer.learning_rate.numpy()*0.95
            print(f"lr changed from {old_lr} to {optimizer.learning_rate.numpy()}")
        """ 
            
        if not REDIRECT_STD_OUT_TO_TXT_FILE:print(f"{str(batch_idx).zfill(6)}", end="\r")

        loss_value = distributed_train_step(batch)    
        total_loss += loss_value
        
        if WRITE_SUMMARY:
            with train_summary_writer.as_default():
                tf.summary.scalar("train/sampled-softmax loss", loss_value.numpy(), step = train_summary_counter)     
                train_summary_counter += 1
                
        if (batch_idx+1)%SHOW_LOSS_EVERY_N_BATCH==0:
            print(f"loss at batch_idx: {str(batch_idx).zfill(8)} is {str(round(total_loss.numpy()/SHOW_LOSS_EVERY_N_BATCH, 5)).zfill(5)} at {str(round((SHOW_LOSS_EVERY_N_BATCH)/(time.time() - tick), 3)).zfill(3)} batches/sec")             
            total_loss = 0
            tick = time.time()
        
        if (batch_idx+1)%METRICS_EVALUATION_AND_SAVE_MODEL_EVERY_N_BATCH == 0:            
            eval_tick = time.time()
            print("- - - EVALUATING METRICS  - - - ")
            count_dict = {"epoch": e,
                          "total_batches": batch_num}
                                                                     
            SAVE_MODEL, best_metrics_dict = compute_and_store_metrics(model, dataset, count_dict, best_metrics_dict, test_summary_writer)

            if SAVE_MODEL:
                model_save_name = f"{MODEL_DIR}/model_{str(e).zfill(2)}_{str(batch_idx).zfill(6)}"
                model.save_weights(model_save_name)
                print(f"model saved at >{model_save_name}<!") 
            time_elapsed_in_eval_step = time.time() - eval_tick
            print(f"time elapsed in evaluation step: {get_time_elapsed_in_h_m_s(time_elapsed_in_eval_step)}")         
            
        batch_num+=1
        
if REDIRECT_STD_OUT_TO_TXT_FILE:        
    sys.stdout.close()
    sys.stderr.close()

q("training over")
