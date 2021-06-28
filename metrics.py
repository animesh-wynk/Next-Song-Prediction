import numpy as np
import pandas as pd
import tensorflow as tf

from config import *

        
def computeSPS(top_k_recommendations, next_items):
    '''
    The Short-term Prediction Success captures the
    ability of the method to predict the next item in the
    sequence. It is 1 if the next item (i.e. the first item
    of recommendation_GT) is present in the recommendations, 0 else.

    Metric Reprted in paper: 33.45 +- 1.17 %

    returns sps (%)
    '''
    print("Computing SPS...")
    count = 0
    # print("Calculating SPS...")
    for i in range(len(next_items)):        
        # print('next_items[i]: ', next_items[i])
        # print('top_k_recommendations[i]: ', top_k_recommendations[i])
        count += next_items[i] in top_k_recommendations[i]     
    
    return (count/len(next_items))*100

    
def computeRecall(top_k_recommendations, recommendations_GT):
    '''
    The usual metrics for top-N recommendation,
    defined as the number of correct recommendations
    divided by the number of unique items in recommendations_GT.
    
    Metric Reprted in paper: 7.52 +- 0.14 %

    returns recall (%)
    '''
    print("Computing Recall...")
    num_seq = len(top_k_recommendations)    
    correct_recommendations_batch = np.zeros((num_seq))
    
#     print('Calculating Recall...')    
    # iterate over every data point
    for i in range(num_seq):        
        num_correct_recommendations = sum([1 for j in top_k_recommendations[i] if j in recommendations_GT[i]])
        correct_recommendations_batch[i] = num_correct_recommendations 
    
    unique_recommendation_GT = [len(set(i)) for i in recommendations_GT]
    recall = np.mean(correct_recommendations_batch/unique_recommendation_GT)*100
    
    return recall


def computeItemCoverage(top_k_recommendations, recommendations_GT):
    '''
    The number of distinct items that were correctly recommended.
    It captures the capacity of the method to make diverse, successful,
    recommendations

    Metric Reprted in paper: 669.75 +- 15.58

    returns item_coverage
    '''
    print("Computing Item Coverage...")
    num_seq = len(top_k_recommendations)
    item_coverage = []

#     print('Calculating Item Coverage...')    
    # iterate over every data point
    for i in range(num_seq):            
        unique_correct_recommendations = [j for j in top_k_recommendations[i] if j in recommendations_GT[i]]
        # print('unique_correct_recommendations: ', unique_correct_recommendations)
        item_coverage.extend(unique_correct_recommendations)

    item_coverage = len(set(item_coverage))
    
    return item_coverage


def computeUserCoverage(top_k_recommendations, recommendation_GT):
    '''
    The fraction of users who received
    at least one correct recommendation. The average
    recall (and precision) hides the distribution of success
    among users. A high recall could still mean that
    many users do not receive any good recommendation.
    This metrics captures the generality of the method.

    Metric Reprted in paper: 87.73 +- 0.98

    returns user_coverage (%)
    '''
    print("Computing User Coverage...")
    
    bs = len(top_k_recommendations)    
    users_with_atleast_one_corect_recommendation_batch = []
    
#     print('Calculating User Coverage...')    
    # iterate over every data point
    for idx, b in enumerate(range(bs)):        
        recommendations = top_k_recommendations[b]
        # print('recommendations: ', recommendations)
        atleast_one_corect_recommendation = any([1 for i in list(recommendations) if i in recommendation_GT[b]])
        users_with_atleast_one_corect_recommendation_batch.append(atleast_one_corect_recommendation)
        
    user_coverage = np.mean(users_with_atleast_one_corect_recommendation_batch)*100
    
    return user_coverage

def computePopularRecommendations(top_k_recommendations, popular_song_ids):
    '''
    The percentage of recommended songs that lies in list of popular songs.
    '''
    print("Computing Popular Recommendations...")
    seq_len = len(top_k_recommendations)
    popular_song_ids_set = set(popular_song_ids)
    popular_songs_reco_count = 0
    for i in range(seq_len):  
        popular_songs_reco_count += len(set(top_k_recommendations[i]).intersection(popular_song_ids_set))

    return (popular_songs_reco_count/(len(top_k_recommendations)*len(top_k_recommendations[0])))*100

def get_metrics(model, dataset):
    
    top_k_recommendations, next_items, recommendations_GT = get_top_k_recommendations(model, dataset) 

    # Compute sps, recall, item_coverage, user_coverage
    print("Evaluating Metrics...")
    sps = computeSPS(top_k_recommendations, next_items)
    recall = computeRecall(top_k_recommendations, recommendations_GT)
    item_coverage = computeItemCoverage(top_k_recommendations, recommendations_GT)
    # user_coverage = computeUserCoverage(top_k_recommendations, recommendations_GT)
    popular_recommendations = computePopularRecommendations(top_k_recommendations, dataset.popular_song_ids)
    
    return sps, recall, item_coverage, popular_recommendations#, user_coverage


def generate_qualitative_results_on_handpicked_songs(model, write_name, handpicked_songs_list, dataset):
    handpicked_songs_list = [[i] for i in handpicked_songs_list]
    
    # Filter out songs that are not present in the vocab
    handpicked_songs_list_processed = []
    for seq in handpicked_songs_list:
        seq_processed = [dataset.item2idx[i] for i in seq if i in dataset.item2idx.keys()]
        if len(seq_processed) > 0:
            handpicked_songs_list_processed.append(seq_processed)

    print(f"\nGenerating Qualitative Results (recommendations in sequence) on Handpicked {len(handpicked_songs_list_processed)} Test Samples ...")
    for inp in handpicked_songs_list_processed:
        visualize_recommendations_in_sequence(model, write_name, dataset, tf.constant([inp]), num_recommendation_timesteps=NUM_RECOMMENDATION_TIMESTEPS)
        
        
def compute_and_store_metrics(model, dataset, count_dict, best_metrics_dict, test_summary_writer):
    '''
    count_dict has keys: "epoch" and "total_batches" 
    '''
        
    # Quantitative results    
    sps, recall, item_coverage, popular_recommendations = get_metrics(model, dataset)

    print(f"sps           : {round(sps, 2)}%")
    print(f"recall        : {round(recall, 2)}%")
    print(f"item_coverage : {item_coverage}")
    print(f"popular_recommendations: {round(popular_recommendations, 2)}%\n")
    # print(f'user_coverage : {round(user_coverage, 2)}%\n')

    if WRITE_SUMMARY:        
        with test_summary_writer.as_default():
            tf.summary.scalar("sps"                    , sps          , step=count_dict["total_batches"])            
            tf.summary.scalar("recall"                 , recall       , step=count_dict["total_batches"])            
            tf.summary.scalar("item_coverage"          , item_coverage, step=count_dict["total_batches"])            
            tf.summary.scalar("popular_recommendations", popular_recommendations, step=count_dict["total_batches"])
            # tf.summary.scalar('user_coverage', user_coverage, step=count_dict['total_batches'])

    SAVE_MODEL = False
    if sps > best_metrics_dict["best_sps"]:
        best_metrics_dict["best_sps"] = sps                
        SAVE_MODEL = True

    if recall > best_metrics_dict["best_recall"]:
        best_metrics_dict["best_recall"] = recall
        SAVE_MODEL = True

    if item_coverage > best_metrics_dict["best_item_coverage"]:
        best_metrics_dict["best_item_coverage"] = item_coverage                    
        SAVE_MODEL = True    
    
    # Qualitative results
    if QUALITATIVE_RESULTS_ON_HANDPICKED_SONGS:
        # Qualitative results on handpicked songs
    
        write_name = os.path.join(QUALITATIVE_RESULTS_DIR, f"epoch_{str(count_dict['epoch']).zfill(2)}_batch_{str(count_dict['total_batches']).zfill(8)}.txt")
        print("write_name: ", write_name)
        
        handpicked_songs_df = pd.read_csv(QUALITATIVE_TEST_DATA_PATH)
        handpicked_songs_list = handpicked_songs_df["song_id"].unique().tolist()
        
        generate_qualitative_results_on_handpicked_songs(model, write_name, handpicked_songs_list, dataset)            
    
    return SAVE_MODEL, best_metrics_dict

def get_top_k_recommendations(model, dataset, k=K):
    '''
    top_k_recommendations -> list of lists of items(not IDs)
    next_items            -> list of items(not IDs)
    recommendation_GT     -> list of items(not IDs)
    '''

    print(f'\nGenerating Top-K Recommendations on {NUM_TEST_SAMPLES_QUANTITATIVE} Test Samples ...')
        
    top_k_recommendations = []
    next_items = []
    recommendations_GT = []
    
    for chunk_idx, chunk in enumerate(pd.read_csv(TEST_DATA_PATH, chunksize=1, nrows=NUM_TEST_SAMPLES_QUANTITATIVE)):        

        if not REDIRECT_STD_OUT_TO_TXT_FILE:
            print(f"{str(chunk_idx).zfill(len(str(NUM_TEST_SAMPLES_QUANTITATIVE)))}/{NUM_TEST_SAMPLES_QUANTITATIVE}", end="\r" )
        
        chunk_array = np.squeeze(chunk.to_numpy())

        user_id = chunk_array[0]
        test_seq_len = chunk_array[1]
        
        song_id_list = chunk_array[2:2+MAX_TEST_SEQ_LEN][-test_seq_len:]
        cut = len(song_id_list)//2
        
        # Discarding the song_id which are not present in the vocab
        song_id_list = [song_id for song_id in song_id_list if song_id in dataset.idx2item.keys()]          
        if len(song_id_list) < 2: continue        
    
        # TODO: HANDLE OOV    

        xs = song_id_list[:cut] # ignoring items that are not present in the vocab (train set)
        xs_batch = np.array([xs])
        
        probs, _, _ = model(xs_batch, training=False)#.numpy()[0]
        probs = probs.numpy()[0] # np array (vocab_size)

        probs[xs] = 0        
        
        top_k = np.argsort(-probs)[:k]
        top_k_recommendations.append(top_k)
        
        next_item = song_id_list[cut]
        next_items.append(next_item)        

        gt = song_id_list[cut:]
        recommendations_GT.append(gt)
    
    return top_k_recommendations, next_items, recommendations_GT


##### FOR GENERATING QUALITATIVE RESULTS ##### 
def visualize_recommendations_in_sequence(model, write_name, dataset, input_idx, gt=None, num_recommendation_timesteps=5):
    '''
    input_idx- np array or tf tensor of shape (1, ?), where 1 is the batch size and ? is the length of the input sequence which is not fixed 
    gt       - np array or tf tensor of shape (1, ?), where 1 is the batch size and ? is the length of the input sequence which is not fixed 
    '''

    inp = input_idx
    recommendations = []

    exclude = np.zeros([1, dataset.vocab_size]) #(1, vocab_size)

    for idx in inp[0]:
        exclude[0, idx] = 1 

    initial_state = None #[state_h, state_c]

    for t in range(num_recommendation_timesteps):
        probs, state_h, state_c = model(inp, training=False, initial_state=initial_state) # probs (1, num_items)

        probs = probs*(1-exclude)             # probs (1, vocab_size)
        pred = tf.math.argmax(probs, axis=-1) # (1, )
        
        exclude[0, int(pred[0])] = 1 #(1, vocab_size)

        inp = tf.expand_dims(pred, axis = -1) # (1, 1)        
        initial_state = [state_h, state_c]
        recommendations.append(int(pred[0]))
    
    show_recommended_song_info(write_name, input_idx[0].numpy().tolist(), dataset, recommendations, gt=None)
    
    
def show_recommended_song_info(write_name, inp, dataset, recommendations, gt=None):    
    '''
    inp             - list of song idx
    recommendations - list of song idx
    gt              - list of song idx
    '''
    if PRINT_QUALITATIVE_RESULTS or WRITE_QUALITATIVE_RESULTS:

        def get_info_from_item(inp):
            info_list = []
            for idx in inp:
                item = dataset.idx2item[idx] 
                if item in dataset.song2info:
                    info_list.append(dataset.song2info[item])
                else:
                    info_list.append(f'titleNotFoundForIdx_{idx}')                
            return info_list
        
        def get_print_str_list(info_list, name):            
            print_str_list = [] 
            print_str_list.append(f'\n{name}\n')            
            for i in info_list:
                print_str_list.append(i+'\n')
            return print_str_list

        write_str_list = [] 
        write_str_list.extend(['\n- - - - - - - GENERATING RECOMMENDATIONS - - - - - - -\n'])

        inp_info   = get_info_from_item(inp)
        inp_print_str_list = get_print_str_list(inp_info, 'input:')
        write_str_list.extend(inp_print_str_list)

        recom_info = get_info_from_item(recommendations)
        recom_print_str_list = get_print_str_list(recom_info, 'recommendations:')
        write_str_list.extend(recom_print_str_list)

        if gt is not None:
            gt_info = get_info_from_item(gt[:K])
            gt_print_str_list = get_print_str_list(gt_info, 'ground truth:')
            write_str_list.extend(gt_print_str_list)

        if PRINT_QUALITATIVE_RESULTS:
            for l in write_str_list:
                print(l[:-1])

        if WRITE_QUALITATIVE_RESULTS:
            f = open(write_name, "a")        
            f.writelines(write_str_list)
            f.close()    


##############################################

if __name__ == "__main__":
    import os
    
    dataset = wynk_songs_dataset(ALL_SONGS_INFO_PATH, TRAIN_SONGS_INFO_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)
    model = rnn_recommendation_system_model(dataset.NUM_ITEMS, EMB_DIM, LSTM_DIM)

    model_path = os.path.join('models', 'models_exp12_4_day_data_2021_01_14_124146_1024_64_20', 'model_05_149999')

    model.build(input_shape=(None, MAX_LEN))
    print(model.summary())
    model.load_weights(model_path)
    print(f'model_path: {model_path}')    
    
    
    q()
    for m in all_models_path_list:
        print(m)
        
        
#             if (batch_id+1)%METRICS_EVALUATION_AND_SAVE_MODEL_EVERY_N_BATCH == 0:
#                 evaluation_n_saving_time_start = time.time()
#                 print('- - - EVALUATING METRICS  - - - ')
#                 # Evaluating metrics for test set

#                 count_dict = {'epoch': e,
#                               'total_batches': batch_num
#                             }
                
#                 SAVE_MODEL, best_metrics_dict = show_and_get_metrics(model, dataset, count_dict, best_metrics_dict, test_summary_writer)
