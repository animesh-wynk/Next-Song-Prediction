import os, shutil, pdb, time
import tensorflow as tf
print('tf.__version__: ', tf.__version__)

from dataset import wynk_sessions_dataset
from model import rnn_reco_model
# from metrics import show_and_get_metrics
from config import *

strategy = STRATEGY
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

### Initialize dataset class object
dataset = wynk_sessions_dataset(TRAIN_DATA_PATH, TRAIN_SONGS_INFO_PATH)

if WRITE_SUMMARY:
    # SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
    # the directory is removed, if it already exists
    if os.path.exists(SUMMARY_DIR):
        shutil.rmtree(SUMMARY_DIR)

    # os.makedirs(SUMMARY_DIR)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, 'train'))
    test_summary_writer  = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, 'test'))
    train_summary_counter = 0
    
### Define model and opt  
with strategy.scope():
    model = rnn_reco_model(dataset.vocab_size, SONG_EMB_DIM, LSTM_DIM)
    
    if LOAD_MODEL:
        model.build(input_shape=(None, MAX_LEN))
        print(model.summary())
        model.load_weights(LOAD_MODEL_PATH)
        print(f'model loaded: {LOAD_MODEL_PATH}')    
    
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
                            name = 'sampled_softmax_loss'
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
    x_batch, y_batch = inputs
    with tf.GradientTape() as tape:
        lstm = model(x_batch, training=True)
        
        loss = compute_loss(model, y_batch, lstm)   
        # loss seems to be the average loss for all the data points in a global batch   
        #tf.print('\nloss: ', loss)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #train_accuracy.update_state(labels, predictions)
    return loss    
    
best_metrics_dict = {'best_sps': -1,
                'best_recall': -1,
                'best_item_coverage': -1}


    
### Training Loop
for e in range(START_EPOCH, EPOCHS):
    print(f'EPOCH: {str(e+1).zfill(len(str(EPOCHS)))}/{EPOCHS}')
    
    print('- - - TRAIN - - - ')  
    
    # Initialize python data generator
    train_gen = dataset.preprocessed_data_generator
    
    # Convert python generator into tf data generator    
    train_gen = tf.data.Dataset.from_generator(
                                train_gen,
                                output_types=(tf.dtypes.int64, tf.dtypes.int64),
                                output_shapes=((BATCH_SIZE, MAX_LEN), (BATCH_SIZE,)) 
                                )
    
    print("\nTRAINING")
    
    # Try prefetch!
    #train_gen = train_gen.prefetch(tf.data.AUTOTUNE)
    train_gen = train_gen.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Make the tf data generator distributable
    train_dist_dataset = strategy.experimental_distribute_dataset(train_gen)

    # Training
    total_loss = 0
    tick = time.time()
    for batch_idx, batch in enumerate(train_dist_dataset):        
        loss_value = distributed_train_step(batch)
        
        q('yahanpr!!!')
        
        total_loss += loss_value
#         print(f'--------------------  {batch_idx}  --------------------')
        
        batch_num_show = 50
        if (batch_idx+1)%batch_num_show==0:
            avg_loss = total_loss.numpy()/(batch_idx+1)
            print(f'loss at batch_idx: {str(batch_idx+1).zfill(8)} is {str(round(avg_loss, 5)).zfill(5)} at {str(round((batch_num_show)/(time.time() - tick), 3)).zfill(3)} batches/sec' )                
            tick = time.time()
        
        if batch_idx == 50000:
            q()
        
        if WRITE_SUMMARY:
            with train_summary_writer.as_default():
                tf.summary.scalar('train/sampled-softmax loss', batch_idx+5, step=train_summary_counter)                
                train_summary_counter += 1
        
        
    q() 



q()

if WRITE_SUMMARY:
    # SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
    # the directory is removed, if it already exists
    if os.path.exists(SUMMARY_DIR):
        shutil.rmtree(SUMMARY_DIR)

    # os.makedirs(SUMMARY_DIR)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, 'train'))
    test_summary_writer  = tf.summary.create_file_writer(os.path.join(SUMMARY_DIR, 'test'))
    train_summary_counter = 0


model = rnn_recommendation_system_model(dataset.NUM_ITEMS, EMB_DIM, LSTM_DIM)

if LOAD_MODEL:
    model.build(input_shape=(None, MAX_LEN))
    print(model.summary())
    model.load_weights(LOAD_MODEL_PATH)
    print(f'model loaded: {LOAD_MODEL_PATH}')
    
optimizer = tf.keras.optimizers.Adam()   
loss_value_batches = tf.keras.metrics.Mean(name='mean', dtype=None)
 
best_metrics_dict = {'best_sps': -1,
                'best_recall': -1,
                'best_item_coverage': -1}

# time_profile = {'preprocessing': 0,
#                'forward': 0,
#                'loss_func': 0,
#                'grads': 0,
#                'optimizer': 0}

@tf.function
def get_loss_and_grads(x_batch, y_batch):

    with tf.GradientTape() as tape:
        
        #forward_tick = time.time()
        lstm = model(x_batch, training=True) # (bs, 50), (bs, 646677)        
        #time_profile['forward'] += time.time() - forward_tick        

        # weights (num_items, lstm_dim) 
        # biases (num_items,)
        # labels (bs, 1)
        # inputs (bs, lstm_dim), where 16 is the bs 
        
        #loss_func_tick = time.time()
        last_layer = model.layers[-1].weights
        loss_value = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(last_layer[0]),
            biases=last_layer[1],
            labels=tf.expand_dims(y_batch, -1),
            inputs=lstm,
            num_sampled = 15,
            num_classes = dataset.NUM_ITEMS,
            num_true = 1,
            remove_accidental_hits=True,
            name = 'sampled_softmax_loss'
        )        # (bs, )
        #time_profile['loss_func'] += time.time() - loss_func_tick
        
        #grads_tick = time.time()
        grads = tape.gradient(loss_value, model.trainable_variables)
        #time_profile['grads'] += time.time() - grads_tick
        
        #optimizer_tick = time.time()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #time_profile['optimizer'] += time.time() - optimizer_tick
        
        return loss_value

batch_num = BATCH_NUM_START
for e in range(START_EPOCH, EPOCHS):
    print(f'EPOCH: {str(e+1).zfill(len(str(EPOCHS)))}/{EPOCHS}')
    
    print('- - - TRAIN - - - ')  
    train_gen = dataset.preprocessed_data_generator(e)
#     train_gen = dataset.create_generator_v3()
    
    tick = time.time()
    total_evaluation_n_saving_time = 0
    
    #preprocessing_tick = time.time()    
    for batch_id, (x_batch, y_batch) in enumerate(train_gen): # (BATCH_SIZE, MAX_LEN) and (BATCH_SIZE,)
        print(batch_id, end='\r')   
        
        if (batch_num!=0) and (batch_num%50_000==0):
            old_lr = optimizer.learning_rate.numpy()
            optimizer.learning_rate = optimizer.learning_rate.numpy()*0.95
            print(f'lr changed from {old_lr} to {optimizer.learning_rate.numpy()}')

        #time_profile['preprocessing'] += time.time() - preprocessing_tick
        
        loss_value = get_loss_and_grads(x_batch, y_batch)                
        loss_value = tf.math.reduce_mean(loss_value).numpy()
        loss_value_batches.update_state(loss_value)        

        '''
        if batch_id==100:
            print('batch_id: ', batch_id)            
            pprint(time_profile)
            total_time = 0
            for k in time_profile:
                total_time += time_profile[k]            
            print('total_time: ', total_time)
            print('>>>', time.time() - tick)
            q()                 
        preprocessing_tick = time.time()    
        continue
        print('should no get printed')
        '''
        if WRITE_SUMMARY:
            with train_summary_writer.as_default():
                tf.summary.scalar('train/sampled-softmax loss'             , loss_value, step=train_summary_counter)                
                train_summary_counter += 1
                    
        if (batch_id+1)%SHOW_LOSS_EVERY_N_BATCH == 0:
            print(f'batch_id: {batch_id+1} | loss: {round(float(loss_value_batches.result().numpy()), 5)} | {round((batch_id+1)/(time.time() - tick - total_evaluation_n_saving_time), 3)} batches/sec')

        if (batch_id+1)%METRICS_EVALUATION_AND_SAVE_MODEL_EVERY_N_BATCH == 0:
            evaluation_n_saving_time_start = time.time()
            
            print('- - - EVALUATING METRICS  - - - ')
            # Evaluating metrics for test set
            count_dict = {'epoch': e,
                          'total_batches': batch_num}
            
            SAVE_MODEL, best_metrics_dict = show_and_get_metrics(model, dataset, count_dict, best_metrics_dict, test_summary_writer)

            if SAVE_MODEL:
                model_save_name = '{}/model_{}_{}'.format(MODEL_DIR, str(e).zfill(2), str(batch_id).zfill(6))
                model.save_weights(model_save_name)
                print(f'model saved at >{model_save_name}<!') 
        
            total_evaluation_n_saving_time += time.time() - evaluation_n_saving_time_start         
        batch_num+=1