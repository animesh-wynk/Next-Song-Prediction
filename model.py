import tensorflow as tf
from tensorflow.keras.layers import Layer
import pdb
from config import *

class customLinear(Layer):
    def __init__(self, in_units, out_units, name=None):
        super(customLinear, self).__init__(name=name)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(in_units, out_units), dtype="float32"),
            trainable=True, 
            name=f"{name}/weights")
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(out_units,), dtype="float32"), 
            trainable=True, 
            name=f"{name}/bias")        

        self.w_regularizer = tf.keras.regularizers.l2(1e-5) 

    def get_regularizer_losses(self):
        '''
        calculates and returns the regularizer losses for weights and/or baises
        '''
        # print('\ncalculating regularizer losses...\n')
        
        # w_regularizer_loss = self.w_regularizer(self.w)
        # b_regularizer_loss = self.b_regularizer(self.b)
        #self.add_loss([w_regularizer_loss, b_regularizer_loss])
        # above written code would not work. can not understand why.

        # self.add_loss([lambda: self.w_regularizer(self.w), lambda: self.b_regularizer(self.b)]) 
        # This method (add_loss) can be used inside a subclassed layer or model's call function, 
        # in which case losses should be a Tensor or list of Tensors.
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

        return [self.w_regularizer(self.w)]

    def call(self, inputs):        
        return tf.matmul(inputs, self.w) + self.b    
      
class rnn_reco_model(tf.keras.Model):
    def __init__(self, vocab_size):        
        super(rnn_reco_model, self).__init__()        
        self.song_emb = tf.keras.layers.Embedding(vocab_size, 
                                                  SONG_EMB_DIM, 
                                                  mask_zero=True, 
                                                  name="song_embedding_layer")               
        
        if USE_TIME_BUCKETS:
            self.time_bucket_emb = tf.keras.layers.Embedding(TIME_BUCKET_VOCAB_SIZE, 
                                                             TIME_BUCKET_EMB_DIM, 
                                                             mask_zero=True, 
                                                             name="time_bucket_embedding_layer")
        
        self.bn1 = tf.keras.layers.BatchNormalization(name='batchnorm_inputs')
        
        self.lstm = tf.keras.layers.LSTM(LSTM_DIM, 
                                         return_state=True,
                                         return_sequences=True,
                                         recurrent_initializer='glorot_normal',
                                         name='lstm_layer'
                                         )
            
        self.bn2 = tf.keras.layers.BatchNormalization(name='batchnorm_lstm')
                
        self.attn = tf.keras.layers.Attention(use_scale=False, causal=True, name='attention')
        
        LSTM_DIM_HALF = int(LSTM_DIM/2)
        self.reduction_layer = tf.keras.layers.Dense(units=LSTM_DIM_HALF, 
                                                     activation="relu", 
                                                     name="activation_size_reduction_layer")
        
        LSTM_DIM_QUARTER = int(LSTM_DIM_HALF/2)
        self.reduction_layer2 = tf.keras.layers.Dense(units=LSTM_DIM_QUARTER, 
                                                     activation="relu", 
                                                     name="activation_size_reduction_layer2")
        
        self.dense = customLinear(in_units=LSTM_DIM_QUARTER, out_units=vocab_size, name=f"{self.name}/last_layer")
        self.dense.build((LSTM_DIM_QUARTER, ))
          
    
    def call(self, song_emb_inp, time_bucket_emb_inp, initial_state=None, training=True):   
#         print('\n-----')
#         print("in call")
#         print("song_emb_inp: ", song_emb_inp)
        song_emb = self.song_emb(song_emb_inp)                           # (bs, MAX_LEN, SONG_EMB_DIM)
#         print("song_emb_inp.shape: ", song_emb_inp.shape)
        song_emb_mask = self.song_emb.compute_mask(song_emb_inp)         # (bs, MAX_LEN)

        lstm_inp = song_emb                                              # (bs, MAX_LEN, SONG_EMB_DIM)
        lstm_inp_mask = song_emb_mask                                    # (bs, MAX_LEN)
                
        if (time_bucket_emb_inp is not None) and USE_TIME_BUCKETS:
            time_bucket_emb = self.time_bucket_emb(time_bucket_emb_inp)  # (bs, MAX_LEN, TIME_BUCKET_EMB_DIM)            
            lstm_inp = tf.concat([lstm_inp, time_bucket_emb], axis = -1) # (bs, MAX_LEN, SONG_EMB_DIM+TIME_BUCKET_EMB_DIM)
            
#         print("lstm_inp.shape: ", lstm_inp.shape)
        lstm_inp = self.bn1(lstm_inp)
#         print("lstm_inp.shape: ", lstm_inp.shape)        
        
        lstm, state_h, state_c = self.lstm(lstm_inp, 
                                           initial_state=initial_state)
#         print("lstm.shape: ", lstm.shape)
        
        lstm = self.bn2(lstm)
#         print("lstm.shape: ", lstm.shape)
                
        # Self attention so key=value in inputs
        attn = self.attn(inputs=[lstm, lstm], 
                        mask=[lstm_inp_mask, lstm_inp_mask])
#         print("attn.shape: ", attn.shape) 
        
        reduction_layer_out = self.reduction_layer(attn)
        
        reduction_layer_out2 = self.reduction_layer2(reduction_layer_out)
#         print("reduction_layer_out.shape: ", reduction_layer_out.shape)

#         output = tf.keras.layers.Dense(1000, name='output')(attn)
#         print("output.shape: ", output.shape)
#         q("attn shape")

        
        # lstm, state_h, state_c = self.lstm(lstm_inp, mask=song_emb_mask, initial_state=initial_state) 
        # lstm.shape: (bs, MAX_LEN, LSTM_DIM) if USE_ATTENTION else (bs, LSTM_DIM)
        
#         q()
        if not training:
            logits = self.dense(reduction_layer_out2)
            logits = logits[:, -1, :]
#             print("logits.shape: ", logits.shape)
            probs = tf.nn.softmax(logits, axis=-1)        
            return probs, state_h, state_c 
        else:
            return reduction_layer_out2  



if __name__ == "__main__":
    from dataset import wynk_sessions_dataset

    dataset = wynk_sessions_dataset(TRAIN_DATA_PATH, TRAIN_SONGS_INFO_PATH)
    
    model = rnn_reco_model(dataset.vocab_size)
    
    
    q("model ok")
    
    
    dummy_inp = tf.random.uniform(shape=(BATCH_SIZE,MAX_LEN), minval=0, maxval=10, dtype=tf.dtypes.int32)
    _, _, _ = model(dummy_inp, eval_mode=True)  
    
    print(model.summary())