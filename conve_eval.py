# Usage: python conve.py
import numpy as np 
import torch 
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import argparse
import math
import pickle
#from progress.bar import Bar
from tqdm import tqdm
import sys
import json
from sklearn import preprocessing
import time

# NER related
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# AllenNLP NER predictor and subject, object order predictor
from allennlp.predictors import Predictor
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
so_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")



# Rel classification related
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
 
global graph
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

rel_model = tf.keras.models.load_model('models/reldesc_classificn_trainedmodel.h5')
rel_model._make_predict_function()

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('looks')
stop_words.append('like')
stop_words.append('country')

# Flair embeddings
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        # Embedding tables for entity and relations with num_uniq_ent in y-dim, emb_dim in x-dim
        self.emb_e = torch.nn.Embedding(num_entities, args.emb_dim, padding_idx=0)
        self.ent_weights_matrix = np.zeros((num_entities, args.emb_dim))
        self.emb_rel = torch.nn.Embedding(num_relations, args.emb_dim, padding_idx=0)
        self.ne = num_entities
        self.nr = num_relations
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.emb_dim)
        self.ln0 = torch.nn.LayerNorm(args.emb_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(16128,args.emb_dim)
    
    def ent2id(self):
        entitymap = {}
        identmap = {}
        with open('data/wikidata/kg_training_entids_withshapes.txt') as f:
            for ln in f:
                #print(ln)
                ent,entid = ln.strip().split('   ')
                if ent in entitymap:
                    pass
                else:
                    entitymap[ent] = int(entid)
                    identmap[int(entid)] = ent
        return entitymap, identmap
    
    def init_entemb(self):
        self.entitymap, self.identmap = self.ent2id()
        with open('data/glove.6B.300d.txt','r') as f:
            self.word_vocab = set() # not using list to avoid duplicate entry
            self.word2vector = {}
            flag = 0
            for line in f:
                line_ = line.strip() #Remove white space
                words_Vec = line_.split()
                self.word_vocab.add(words_Vec[0])
                self.word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
                
        ent_found = 0
        for i in range(self.ne):
            flag = 0
            found_list = []
            not_found_list = []
            found_list_vec = []
            glove_ent = nltk.word_tokenize(self.identmap[i])
            #glove_ent = ' '.join(glove_ent)
            for w in glove_ent:
                if w.lower() in self.word_vocab:
                    flag += 1
                    found_list.append(w.lower())
                    found_list_vec.append(self.word2vector[w.lower()])
                else:
                    not_found_list.append(w.lower())
            if len(found_list) != 0:
                print('KG entity:{}, Found in glove:{}'.format(' '.join(glove_ent), ' '.join(found_list)))
                np_vec = np.mean(np.asarray(found_list_vec), axis=0)
                self.ent_weights_matrix[i] = np_vec
            else:
                print('KG entity:{}, Not found in glove:{}'.format(' '.join(glove_ent), ' '.join(not_found_list)))
                self.ent_weights_matrix[i] = np.random.normal(scale=0.6, size=(args.emb_dim, ))
    
    def init(self):
        # Xavier initialization
        #xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

        # Pre-trained embeddings initialization
        self.init_entemb()
        self.emb_e.load_state_dict({'weight': torch.from_numpy(self.ent_weights_matrix)})
        
    
    def forward(self, e1, rel, print_pred=False):
        batch_size = 1
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 30)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 30)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        # Try Layer norm instead of batch norm
        #x = self.bn2(x)
        x = self.ln0(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0)) # shape (batch, n_ent)
        x = self.hidden_drop(x)
        x += self.b.expand_as(x)
        pred = torch.nn.functional.softmax(x, dim=1)
        #print(pred.shape)
        return pred

class WikiData():
    def __init__(self, args):
        super(WikiData, self).__init__()
        self.ent_path = args.entdatapath
        self.rel_path = args.reldatapath
        self.train_file = args.traindatapath
        self.test_file = args.testdatapath
        self.entity_ids = self.load_data(self.ent_path) 
        self.ids2entities =  self.id2ent(self.ent_path) 
        self.rel_ids =  self.load_data(self.rel_path)
        self.ids2rel =  self.id2rel(self.rel_path) 
        self.train_triples_list = self.convert_triples(self.train_file)
        self.test_triples_list = self.convert_triples(self.test_file)

    def load_data(self, data_path):
        item_dict = {}
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                name, id = line.strip().split('   ')
                item_dict[name] = int(id)
        return item_dict
    
    def id2ent(self, data_path):
        item_dict = {}
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                name, id = line.strip().split('   ')
                item_dict[int(id)] = name
        return item_dict
    
    def id2rel(self, data_path):
        item_dict = {}
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                name, id = line.strip().split('   ')
                item_dict[int(id)] = name
        return item_dict
    
    def convert_triples(self, data_path):
        triples_list = []
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                item_dict = json.loads(line.strip())
                h = item_dict['e1']
                r = item_dict['rel']
                t = item_dict['e2_multi1'].split('\t')
                hrt_list = []
                hrt_list.append(self.entity_ids[h])
                hrt_list.append(self.rel_ids[r])
                t_ents = []
                for t_idx in t:
                    t_ents.append(self.entity_ids[t_idx])
                hrt_list.append(t_ents)
                triples_list.append(hrt_list)
        return triples_list

def train(args):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)
    triples_list = data.train_triples_list
    num_train_samples = len(triples_list)
    num_train_steps = math.ceil(num_train_samples/int(args.batch_size))
    
    model = ConvE(args, num_entities, num_relations)
    model.init()
    optimizer = torch.optim.Adam(model.parameters())

    # Fitting the Model
    model.train()
    for ep in tqdm(range(args.max_epochs)):
        epoch_loss = 0
        # TODO: Check support for higher batch size > 1 training
        for counter in range(num_train_steps):
            optimizer.zero_grad()
            e2_multi = torch.zeros(args.batch_size, num_entities)
            train_sample = triples_list[counter:counter+int(args.batch_size)]
            h,r,t = train_sample[0]
            logits = model.forward(torch.tensor(h), torch.tensor(r), print_pred=False)
            for t_id in t:
                e2_multi[0][t_id] = 1.0
            loss = model.loss(logits, torch.tensor(e2_multi))
            loss.backward()
            optimizer.step()
            batch_loss = torch.sum(loss)
            print('Batch {}: Batch loss:{}'.format(counter, batch_loss))
            epoch_loss += batch_loss
            counter = counter + int(args.batch_size)
        print('Epoch loss:{}'.format(epoch_loss))
    
    # Save model    
    torch.save(model.state_dict(), args.modelpath)

def eval(args):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)

    names_dict = data.ids2entities
    rel_dict = data.ids2rel

    model = ConvE(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    triples_list = data.test_triples_list
    num_test_samples = len(triples_list)
    for i in range(num_test_samples):
        test_sample = triples_list[i]
        print('Testing sample:{}'.format(test_sample))
        h,r,t = test_sample
        logits = model.forward(torch.tensor(h), torch.tensor(r), print_pred=False)
        score, pred = torch.topk(logits,args.top_k,1)

        #print("--------------------------------------------------------------------------")
        for j, id in enumerate(pred[0].cpu().detach().numpy()): 
            pred_entity = names_dict[id]
            print('id:{} Head:{}, Relation:{}, Pred:{}, Target:{}'.format(id, names_dict[h], rel_dict[r], names_dict[id], names_dict[t[0]]))

# def entity_extraction(recv_msg):
#     label_list = [
#     "O",       # Outside of a named entity
#     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
#     "I-MISC",  # Miscellaneous entity
#     "B-PER",   # Beginning of a person's name right after another person's name
#     "I-PER",   # Person's name
#     "B-ORG",   # Beginning of an organisation right after another organisation
#     "I-ORG",   # Organisation
#     "B-LOC",   # Beginning of a location right after another location
#     "I-LOC"    # Location
#     ]

#     # Bit of a hack to get the tokens with the special tokens
#     tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(recv_msg)))
#     inputs = tokenizer.encode(recv_msg, return_tensors="pt")

#     outputs = model(inputs)[0]
#     predictions = torch.argmax(outputs, dim=2)

#     entities = []
#     #print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])
#     for token, prediction in zip(tokens, predictions[0].tolist()):
#         if 'I-LOC' in label_list[prediction]:
#             entities.append(token)
#     if '[CLS]' in entities:
#         entities.remove('[CLS]')
#     elif '[SEP]' in entities:
#         entities.remove('[SEP]')
#     entities = ' '.join(ent for ent in entities)

#     return [entities]

def entity_extraction(recv_msg):
    # NER model to extract the entity information
    ent_list = []
    ent = None
    utt = recv_msg.strip()
    ner_results = ner_predictor.predict(sentence=utt)
    try:
        for word, tag in zip(ner_results["words"], ner_results["tags"]): 
                if 'LOC' in tag:
                    if 'U-LOC' in tag:
                        #print('NER tag:{}'.format(tag))
                        ent =  word
                    elif 'B-LOC' in tag or begin == 1:
                        begin = 1
                        ent_list.append(word)
                        if 'L-LOC' in tag:
                            begin = 0
                            ent = ' '.join(ent_list)
        return [ent]
    except:
        ent = None
        return None

def relation_extraction(recv_msg):
    global rel_model
    global sess
    global graph
    
    # Label encoding
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

    X_test = []
    X_test.append(utt)

    # loading the same tokenizer used during training
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #print('Loaded the tokenizer')
    tokenized_utt = tokenizer.texts_to_sequences(X_test)
    padded_utt = pad_sequences(tokenized_utt, padding='post', maxlen=max_len)
    #print('padded_utt:{}'.format(padded_utt))
    #try:
    with graph.as_default():
        set_session(sess)
        rel_pred = rel_model.predict(padded_utt)
        #print('rel_pred:{}'.format(rel_pred))
        predicted_res_rel = np.argmax(rel_pred, axis=-1)
        #print('predicted_res_rel:{}'.format(predicted_res_rel))
        pred_rel = rel_le.inverse_transform(predicted_res_rel)[0]
        #print('pred_rel:{}'.format(pred_rel))
        return pred_rel

def infer(args, ent, rel, contains_shape=True):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)

    entitynames_dict = data.ids2entities
    entityids_dict = data.entity_ids

    glove_embeddings = WordEmbeddings('en-crawl')
    document_embeddings = DocumentPoolEmbeddings([glove_embeddings])

    model = ConvE(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    global stop_words
    # Pick a random entity id in case there are errors, we don't return null
    target_ent_idx = 100
    if contains_shape:
        ent_emb = model.state_dict()['emb_e.weight']
        entscore_dict = {}
        #print('Entity:{}'.format(ent))
        # Flair embeddings
        toks = ent.split(' ')
        ent_tokens = []
        for t in toks:
            if t not in stop_words:
                ent_tokens.append(t)

        ent_tokens = ' '.join(ent_tokens)
        query_sentence = Sentence(ent_tokens)
        print('Query sentence:{}'.format(query_sentence))
        document_embeddings.embed(query_sentence)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        np_query_vec = query_sentence.embedding.numpy()
        
        for i, ent in enumerate(ent_emb):
            np_ent = ent.numpy()
            numerator_ = np_query_vec.dot(np_ent)
            denominator_= np.sqrt(np.sum(np.square(np_query_vec))) * np.sqrt(np.sum(np.square(np_ent)))
            sim_score = numerator_/denominator_
            entscore_dict[i] = sim_score
        
        target_ent_idx = max(entscore_dict, key=entscore_dict.get)
    
        ent = entitynames_dict[target_ent_idx]
        print('Chose entity:{}'.format(ent))

    h_idx = data.entity_ids[ent]
    r_idx = data.rel_ids[rel]

    # print('h idx:{}'.format(h_idx))
    # print('r idx:{}'.format(r_idx))

    start_time = time.time()
    logits = model.forward(torch.tensor(h_idx), torch.tensor(r_idx), print_pred=False)
    end_time = time.time()
    print('Inference time:{} ms'.format((end_time-start_time)*1000))
    pred_scores, pred = torch.topk(logits,args.top_k,1)

    # Export to onnx 
    #torch.onnx.export(model, (torch.tensor(h_idx), torch.tensor(r_idx)), "models/conve.onnx", verbose=True, opset_version=9, training=False)

    top_k_entities = []
    top_k_scores = pred_scores[0].cpu().detach().numpy().tolist()

    for j, id in enumerate(pred[0].cpu().detach().numpy()): 
        pred_entity = entitynames_dict[id]
        #ent_score = pred_scores[j].cpu().detach().numpy().tolist()
        #print('id:{} Head:{}, Relation:{}, Pred:{}'.format(id, ent, rel, entitynames_dict[id]))
        top_k_entities.append(entitynames_dict[id])
        #mem_dict[pred_entity] = top_k_scores[j]
        #top_k_scores.append(ent_score)
    #print("--------------------------------------------------------------------------")
    return top_k_scores, top_k_entities

def beam_search_decode(ent_dict):
    pass

def score(args):
    top_1_score = 0
    top_3_score = 0
    top_5_score = 0
    with open(args.scoringfilepath, 'r') as f:
        lines = f.readlines()
        #print("--------------------------------------------------------------------------")
        for line in lines:
            #print(line)
            game_id, target_country, utterances = line.strip().split('\t')
            #print('Game id:{} Target country:{} Utterances:{}'.format(game_id, target_country, utterances))
            utterances = utterances.split('<br>')
            candidates = {}
            mem_dict = {}
            for index, utt in enumerate(utterances):
                candidates[index] = {}
                contains_shape = True
                rel = relation_extraction(utt)
                new_rel = rel + '_reverse'
                entities = entity_extraction(utt)
                if not entities[0]:
                    entities = [utt]
                    contains_shape = True
                if 'Shape' in rel:
                    contains_shape = True
                    new_rel = rel
                
                if 'LocatedIn' in rel or 'InContinentLoc' in rel or 'HasSize' in rel or 'InContinent' in rel:
                    new_rel = rel
                
                if 'OtherRel' in rel:
                    new_rel = 'HasShape'
                    rel = 'HasShape'
                
                print ("==========>", entities[0], new_rel)
                candidates[index]['rel'] = rel
                top_k_scores, top_k_entities = infer(args, entities[0], new_rel, contains_shape)
                candidates[index]['entities'] = top_k_entities
                candidates[index]['entity_scores'] = top_k_scores
                
                # Add to the top-k entities dictionary to track across utterances
                for idx, k_ent in enumerate(top_k_entities):
                    if k_ent in mem_dict:
                        mem_dict[k_ent] += top_k_scores[idx]
                    else:
                        mem_dict[k_ent] = top_k_scores[idx]
                
                contains_shape = False
                
                if index > 0:
                    # print(list(candidates[index-1]['entities'])[0])
                    # print(list(candidates[index-1]['entity_scores'])[0])
                    for prev_idx, prev_ent in enumerate(list(candidates[index-1]['entities'])):
                        top_k_scores, top_k_entities = infer(args, prev_ent, rel, contains_shape)
                        for k_ent in top_k_entities:
                            if k_ent in entities:
                                #k_idx = list(candidates[index]['entities']).index(k_ent)
                                mem_dict[prev_ent] += mem_dict[prev_ent]
                    

            #print(candidates)
            top_1 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:1]
            top_3 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:3]
            top_5 = sorted(mem_dict.items(), key=lambda item: item[1], reverse=True)[:5]
            top_1_ent = [i[0] for i in top_1]
            top_3_ent = [i[0] for i in top_3]
            top_5_ent = [i[0] for i in top_5]
            #print(top_1_ent, top_3_ent, top_5_ent)
            print('{} ====> {},{},{} '.format(line, top_1_ent, top_3_ent, top_5_ent))
            print("--------------------------------------------------------------------------")
            #break
            


def main(args):

    # if args.do_train:
    # Model training
    # train(args)

    # if args.do_eval:
    #     #Model evaluation
    #     eval(args)

    # Model inference
    start_time = time.time()
    infer(args, 'a fetus', 'HasShape_reverse')
    end_time = time.time()
    print('Total inference time:{}'.format(end_time-start_time))
    
    # Model scoring
    #score(args)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', dest='do_train', default=True, help='Whether to train the model or do eval only')
    parser.add_argument('--do_eval', dest='do_eval', default=True, help='Whether to do evaluation on the model')
    parser.add_argument('--entdatapath', dest='entdatapath', default='data/wikidata/kg_training_entids_withshapes.txt', help='Path to the file containing the entities and entity IDs')
    parser.add_argument('--reldatapath', dest='reldatapath', default='data/wikidata/kg_training_relids_withshapes.txt', help='Path to the file containing the relations and relation IDs')
    parser.add_argument('--traindatapath', dest='traindatapath', default='data/wikidata/e1rel_to_e2_train.json', help='Path to the training data file')
    parser.add_argument('--testdatapath', dest='testdatapath', default='data/wikidata/e1rel_to_e2_ranking_test.json', help='Path to the test data file')
    parser.add_argument('--modelpath', dest='modelpath', default='models/conve_glove.pt', help='Path to save the trained model to')
    parser.add_argument('--scoringfilepath', dest='scoringfilepath', default='data/perfect_segmented_targets_short.tsv', help='Path to file containing the utterances to be used for scoring')
    
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