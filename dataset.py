import pandas as pd
import numpy as np
import pdb, pickle, os, random, sys, time
# from tqdm import tqdm 
import tensorflow as tf

from config import *    

class wynk_sessions_dataset():
    def __init__(self, train_data_path, train_songs_info_path):
        self.train_data_path = train_data_path
        self.train_songs_info_path = train_songs_info_path

        # Build vocab and make all necessary dictionaries
        self._build_vocab()
        
        # Make song2info dictionary
        self._map_song2info()
        
    def preprocessed_data_generator(self, epoch):
        preprocess_training_data_path_list = sorted(os.listdir(PREPROCESSED_TRAINING_DATA_PATH))
        preprocess_training_data_path_list = [i for i in preprocess_training_data_path_list if i != '.ipynb_checkpoints'] # this file gets created at this path sometimes
        
        preprocess_training_data_path = os.path.join(PREPROCESSED_TRAINING_DATA_PATH, preprocess_training_data_path_list[epoch%len(preprocess_training_data_path_list)])
        
        print(f'using {preprocess_training_data_path} in preprocessed_data_generator()')
        for chunk in pd.read_csv(preprocess_training_data_path, chunksize=BATCH_SIZE):
            chunk_np = chunk.to_numpy() # (bs, max_len+1=21)
            x_batch = chunk_np[:, :-1] # (bs, max_len=20)
            y_batch = chunk_np[:, -1]  # (bs,)
            yield x_batch, y_batch
                
    def _map_song2info(self):
        print('\n##### Mapping song2info... #####')
        
        if os.path.exists(SONG2INFO_PICKLE_PATH):
            print('\nLoading song2info dict...')
            infile = open(SONG2INFO_PICKLE_PATH,'rb')
            self.song2info = pickle.load(infile)
            infile.close()

        else:
            print('\nMapping song2info...')    
            
            # Read file at TRAIN_SONGS_INFO_PATH
            train_songs_info_df = pd.read_parquet(TRAIN_SONGS_INFO_PATH, 
                                                  columns=["song_id", "title", "album", "artist", "frequency", "language", "publishedYear"])
            print("train_songs_info_df.shape: ", train_songs_info_df.shape) #(149345, 7)          
           
            # Make song2info dictionary                
            self.song2info = {row["song_id"]: f"TITLE: {row['title']} | ALBUM: {row['album']} | ARTIST: {row['artist']} | FREQUENCY: {row['frequency']} | LANG: {row['language']} | YEAR: {row['publishedYear']}" 
                             for _, row in train_songs_info_df.iterrows()}
        
            print("len(self.song2info): ", len(self.song2info))
            
            # Save it into a pickle file
            outfile = open(SONG2INFO_PICKLE_PATH, "wb")
            pickle.dump(self.song2info, outfile)
            outfile.close() 

        print('##### Mapped song2info... #####\n')
    
    def _build_vocab(self):
        print('\n##### Building vocab... #####')
        '''
        Songs with higher frequencies have lower embedding ids
        '''
        # Read song_info file as a pandas dataframe
        song_info_df = pd.read_parquet(self.train_songs_info_path, columns=["song_id", "song_embedding_id", "frequency"])

        print('song_info_df.shape: ', song_info_df.shape) # (149_345, 2)
        print('song_info_df.columns: ', song_info_df.columns) # song_id, frequency

        # Sorting song_info_df by frequency values         
        song_info_df = song_info_df.sort_values(by='frequency', ascending=False)
        song_info_df = song_info_df.reset_index(drop=True)
        
        # Store list of popular song (top 5% songs sorted by frequency)
        self.popular_songs_num = int(POPULAR_SONGS_PERCENTAGE*song_info_df.shape[0])
        print('self.popular_songs_num: ', self.popular_songs_num)        
        self.popular_songs = song_info_df.iloc[:self.popular_songs_num, :]['song_id'].to_list()
        
        # Make dictionaries
        self.item2idx = {}    
        self.item2idx[SONG_PAD_TOKEN] = SONG_PAD_INDEX
        self.item2idx[SONG_UNK_TOKEN] = SONG_UNK_INDEX
        for _, row in song_info_df.iterrows():
            self.item2idx[row["song_id"]] = row["song_embedding_id"]        

        self.idx2item = {song_embedding_id:song_id for song_id, song_embedding_id in self.item2idx.items()}
        
        assert len(self.item2idx) == len(self.idx2item), "len(self.item2idx) != len(self.idx2item)"
        
        self.vocab_size = len(self.item2idx) # earlier self.NUM_ITEMS
        print("self.vocab_size: ", self.vocab_size)
        
        print('##### Vocab built... #####\n')
        
        

if __name__ == "__main__":

    dataset = wynk_sessions_dataset(TRAIN_DATA_PATH, TRAIN_SONGS_INFO_PATH)    
    
    
    q('dun.')
