import argparse
import numpy as np
import math
import pickle
import sys
import json
import time

## Torch related imports
import torch 
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_

## Rel classification related imports
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

from sklearn import preprocessing
 
global graph
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

rel_model = tf.keras.models.load_model('models/reldesc_classificn_trainedmodel.h5')
rel_model._make_predict_function()

## Entity extraction related imports
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

## AllenNLP NER predictor and subject, object order predictor
from allennlp.predictors import Predictor
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
so_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

## Flair embeddings
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

## Extract the Glove vectors and nltk english stop words
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('looks')
stop_words.append('like')
stop_words.append('country')

with open('data/glove.6B.300d.txt','r') as f:
    word_vocab = set() ## Not using list to avoid duplicate entry
    word2vector = {}
    for line in f:
        line_ = line.strip() ## Remove white space
        words_vec = line_.split()
        word_vocab.add(words_vec[0])
        word2vector[words_vec[0]] = np.array(words_vec[1:],dtype=float)

## Model definition and data processing related imports
from conve import ConvE, WikiData

def entity_extraction(recv_msg):
    ## NER model to extract the entity information
    ## Country and Continent information is extracted by NER 
    ## But not shapes information, in that case, just return the utterance

    utt = recv_msg.strip()
    contains_shape = False
    try:
        ent_list = []
        ner_results = ner_predictor.predict(sentence=utt)
        for word, tag in zip(ner_results["words"], ner_results["tags"]): 
            if 'LOC' in tag:
                if 'U-LOC' in tag:
                    ent =  word
                elif 'B-LOC' in tag or begin == 1:
                    begin = 1
                    ent_list.append(word)
                    if 'L-LOC' in tag:
                        begin = 0
                        ent = ' '.join(ent_list)
        return [ent], contains_shape
    except:
        ent = utt
        contains_shape = True
        return [ent], contains_shape

def relation_extraction(recv_msg):
    global rel_model
    global sess
    global graph
    
    ## Label encoding
    rels = ['HasSize', 'HasCapital', 'HasLocation', 'HasShape_reverse', 'InContinent', 
            'IsEastOf', 'IsNorthOf', 'IsSouthOf', 'IsWestOf', 'LocatedInOrNextToBodyOfWater', 
            'SharesBorderWith', 'InContinentLoc_East', 'InContinentLoc_North', 'InContinentLoc_South',
             'InContinentLoc_West', 'InContinentLoc_Central', 'InContinentLoc_North_east', 'InContinentLoc_South_east',
            'HasSize_reverse', 'HasCapital_reverse', 'HasLocation_reverse', 'InContinent_reverse', 
            'IsEastOf_reverse', 'IsNorthOf_reverse',
            'IsSouthOf_reverse', 'IsWestOf_reverse', 'LocatedInOrNextToBodyOfWater_reverse', 
            'SharesBorderWith_reverse',
            'InContinentLoc_Central_reverse', 'InContinentLoc_East_reverse', 'InContinentLoc_North_reverse',
             'InContinentLoc_North_east_reverse', 'InContinentLoc_South_reverse', 'InContinentLoc_South_East_reverse',
             'InContinentLoc_West_reverse', 'OtherRel']
    rel_le = preprocessing.LabelEncoder()
    rel_le.fit(rels)

    pred_rel = None
    utt = recv_msg.strip()

    max_len = 30 # Based on the training data

    X_test = [utt]

    ## Loading the same tokenizer used during training
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    ## Get the utterances after tokenization and pad to the max seq length
    tokenized_utt = tokenizer.texts_to_sequences(X_test)
    padded_utt = pad_sequences(tokenized_utt, padding='post', maxlen=max_len)
    
    with graph.as_default():
        set_session(sess)
        rel_pred = rel_model.predict(padded_utt)
        predicted_res_rel = np.argmax(rel_pred, axis=-1)
        pred_rel = rel_le.inverse_transform(predicted_res_rel)[0]
        return pred_rel

def infer(args, ent, rel, contains_shape=True):
    global word2vector
    global word_vocab
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)

    entitynames_dict = data.ids2entities
    entityids_dict = data.entity_ids

    model = ConvE(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    global stop_words
    ## Pick a random entity id in case there are errors, we don't return null
    target_ent_idx = 100
    if contains_shape:
        ent_emb = model.state_dict()['emb_e.weight']
        entscore_dict = {}
        
        ## Get Glove embeddings
        toks = ent.split(' ')
        ent_tokens = []
        for t in toks:
            if t not in stop_words:
                ent_tokens.append(t)

        np_query_vec = np.random.normal(scale=0.6, size=(args.emb_dim, ))
        found_list = []
        not_found_list = []
        found_list_vec = []
        for i in ent_tokens:
            glove_ent = nltk.word_tokenize(i)
            for w in glove_ent:
                if w.lower() in word_vocab:
                    found_list.append(w.lower())
                    found_list_vec.append(word2vector[w.lower()])
                else:
                    not_found_list.append(w.lower())
            if len(found_list) != 0:
                np_query_vec = np.mean(np.asarray(found_list_vec), axis=0)         
        
        for i, ent in enumerate(ent_emb):
            np_ent = ent.numpy()
            numerator_ = np_query_vec.dot(np_ent)
            denominator_= np.sqrt(np.sum(np.square(np_query_vec))) * np.sqrt(np.sum(np.square(np_ent)))
            sim_score = numerator_/denominator_
            entscore_dict[i] = sim_score
        
        target_ent_idx = max(entscore_dict, key=entscore_dict.get)
    
        ent = entitynames_dict[target_ent_idx]
        print(f'Best entity fit from the knowledge graph entities for the given shape description :{[ent]}')

    ## Pick a random entity id in case there are errors, we don't return null
    try:
        h_idx = data.entity_ids[ent]
    except KeyError:
        h_idx = target_ent_idx

    r_idx = data.rel_ids[rel]

    start_time = time.time()
    logits = model.forward(torch.tensor(h_idx), torch.tensor(r_idx), print_pred=False)
    end_time = time.time()
    pred_scores, pred = torch.topk(logits,args.top_k,1)

    top_k_entities = []
    top_k_scores = pred_scores[0].cpu().detach().numpy().tolist()

    for j, id in enumerate(pred[0].cpu().detach().numpy()): 
        pred_entity = entitynames_dict[id]
        top_k_entities.append(entitynames_dict[id])
    
    return top_k_scores, top_k_entities

def score(args):

    ## We are tracking the top-1, top-3, top-5 predictions to evaluate the knowledge graph performance
    top_1_score = 0
    top_3_score = 0
    top_5_score = 0
    
    ## Track total number of target countries to score the knowledge graph performance
    num_lines = 0
    
    with open(args.scoringfilepath, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            num_lines += 1
            
            utterances = line.strip().split('\t')
            target_country = utterances[0]
            print(f'Target country: {target_country}; Utterances for the target country ====> {utterances[1:]}')
            candidates = {}
            mem_dict = {}
            
            ## For each utterance for the target country, extract the entity and relation 
            ## and get the target country prediction via knowledge graph inference
            for index, utt in enumerate(utterances[1:]):
                candidates[index] = {}
                
                ## Extract the relation type
                ## Predictions can include rel predictions with and without 'reverse' string ==> both are valid
                rel = relation_extraction(utt).split('_reverse')[0]
                
                ## To get the predictions for the target entity, adding 'reverse' to the current relation
                new_rel = rel + '_reverse'

                ## Extract the source entity
                entities, contains_shape = entity_extraction(utt)             
                               
                if 'OtherRel' in rel:
                    contains_shape = True
                    new_rel = 'HasShape'
                
                print(f'{index}. Extracted entities:{entities}')
                print(f'{index}. Extracted relations:{[rel]}')

                ## Track the predictions across utterances
                candidates[index]['rel'] = rel
                top_k_scores, top_k_entities = infer(args, entities[0], new_rel, contains_shape)
                candidates[index]['entities'] = top_k_entities
                candidates[index]['entity_scores'] = top_k_scores
                
                ## Add to the top-k entities dictionary to track across utterances
                for idx, k_ent in enumerate(top_k_entities):
                    if k_ent in mem_dict:
                        mem_dict[k_ent] += top_k_scores[idx]
                    else:
                        mem_dict[k_ent] = top_k_scores[idx]
                
                print (f'==========> Updated mem dict ==> {sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)}\n')
                
            ## Sort the predictions dictionary to get the top-1, top-3 and top-5 predictions
            top_1 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:1]
            top_3 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:3]
            top_5 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:5]
            top_1_ent = [i[0] for i in top_1]
            top_3_ent = [i[0] for i in top_3]
            top_5_ent = [i[0] for i in top_5]

            print(f'Target country ====> {target_country}')
            print(f'Target country predictions ====> Top-1: {top_1_ent}; Top-3: {top_3_ent}; Top-5: {top_5_ent} ')
            print('---------------------------------------------------------------------------')
            if target_country.capitalize() in top_1_ent:
                top_1_score += 1
            if target_country.capitalize() in top_3_ent:
                top_3_score += 1
            if target_country.capitalize() in top_5_ent:
                top_5_score += 1
    
            #break
    print(f'============> Top-1 acc: {top_1_score*100/num_lines}%, \
                          Top-3 acc: {top_3_score*100/num_lines}%, \
                          Top-5 acc: {top_5_score*100/num_lines}%')       

def main(args):
  
    ## Model scoring
    ## Scoring refers to inference on the segmented utterances to get the target entity predictions
    score(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', dest='do_train', default=True, help='Whether to train the model or do eval only')
    parser.add_argument('--do_eval', dest='do_eval', default=True, help='Whether to do evaluation on the model')
    parser.add_argument('--entdatapath', dest='entdatapath', default='data/wikidata/kg_training_entids_withshapes.txt', help='Path to the file containing the entities and entity IDs')
    parser.add_argument('--reldatapath', dest='reldatapath', default='data/wikidata/kg_training_relids_withshapes.txt', help='Path to the file containing the relations and relation IDs')
    parser.add_argument('--traindatapath', dest='traindatapath', default='data/wikidata/e1rel_to_e2_train.json', help='Path to the training data file')
    parser.add_argument('--testdatapath', dest='testdatapath', default='data/wikidata/e1rel_to_e2_ranking_test.json', help='Path to the test data file')
    parser.add_argument('--modelpath', dest='modelpath', default='models/conve_glove.pt', help='Path to save the trained model to')
    parser.add_argument('--scoringfilepath', dest='scoringfilepath', default='data/descriptions_for_select_target_sim_2_des_10_times_with_postentshapes.txt', help='Path to file containing the utterances to be used for scoring')
    
    parser.add_argument('--batch_size', dest='batch_size', default=1, help='Batch size')
    parser.add_argument('--input_dropout', dest='input_dropout', default=0.2, help='Input dropout for the model')
    parser.add_argument('--dropout', dest='dropout', default=0.3, help='Dropout for the model')
    parser.add_argument('--feature_map_dropout', dest='feature_map_dropout', default=0.2, help='Feature map dropout for the model')
    parser.add_argument('--use_bias', dest='use_bias', default=True, help='Entity and relation embedding dimensions')
    parser.add_argument('--emb_dim', dest='emb_dim', default=300, help='Entity and relation embedding dimensions')
    parser.add_argument('--lr', dest='lr', default=0.001, help='Learning rate')
    parser.add_argument('--l2', dest='l2', default=5e-4, help='L2 regularization')
    parser.add_argument('--label_smoothing_epsilon', dest='label_smoothing_epsilon', default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--epochs', dest='max_epochs', default=50, help='Max epochs')
    
    parser.add_argument('--top_k', dest='top_k', default=10, help='Top K vals to consider from the predictions')
    
    args = parser.parse_args()

    #Create model
    model = main(args)