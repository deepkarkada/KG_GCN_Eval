# Usage: python conve.py
import numpy as np 
import torch 
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import argparse
import math
#from progress.bar import Bar
from tqdm import tqdm
import sys
import json

class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        # Embedding tables for entity and relations with num_uniq_ent in y-dim, emb_dim in x-dim
        self.emb_e = torch.nn.Embedding(num_entities, args.emb_dim, padding_idx=0)
        self.ent_weights_matrix = torch.ones([num_entities, args.emb_dim], dtype=torch.float64)
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
    
    def init(self):
        # Xavier initialization
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

        # Pre-trained embeddings initialization
        #self.init_flairemb()
        #self.emb_e.load_state_dict({'weight': self.ent_weights_matrix})
        
    
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

        print("--------------------------------------------------------------------------")
        for j, id in enumerate(pred[0].cpu().detach().numpy()): 
            pred_entity = names_dict[id]
            print('id:{} Head:{}, Relation:{}, Pred:{}, Target:{}'.format(id, names_dict[h], rel_dict[r], names_dict[id], names_dict[t[0]]))

def infer(args, ent, rel):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)

    entitynames_dict = data.ids2entities

    model = ConvE(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    h_idx = data.entity_ids[ent]
    r_idx = data.rel_ids[rel]

    print('h idx:{}'.format(h_idx))
    print('r idx:{}'.format(r_idx))

    logits = model.forward(torch.tensor(h_idx), torch.tensor(r_idx), print_pred=False)
    score, pred = torch.topk(logits,args.top_k,1)

    print("--------------------------------------------------------------------------")
    for j, id in enumerate(pred[0].cpu().detach().numpy()): 
        pred_entity = entitynames_dict[id]
        print('id:{} Head:{}, Relation:{}, Pred:{}'.format(id, ent, rel, entitynames_dict[id]))

def main(args):

    # if args.do_train:
    #     #Model training
    #     train(args)

    # if args.do_eval:
    #     #Model evaluation
    #     eval(args)

    # Model inference
    infer(args, 'a fetus', 'HasShape_reverse')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', dest='do_train', default=True, help='Whether to train the model or do eval only')
    parser.add_argument('--do_eval', dest='do_eval', default=True, help='Whether to do evaluation on the model')
    parser.add_argument('--entdatapath', dest='entdatapath', default='data/wikidata/kg_training_entids_withshapes.txt', help='Path to the file containing the entities and entity IDs')
    parser.add_argument('--reldatapath', dest='reldatapath', default='data/wikidata/kg_training_relids_withshapes.txt', help='Path to the file containing the relations and relation IDs')
    parser.add_argument('--traindatapath', dest='traindatapath', default='data/wikidata/e1rel_to_e2_train.json', help='Path to the training data file')
    parser.add_argument('--testdatapath', dest='testdatapath', default='data/wikidata/e1rel_to_e2_ranking_test.json', help='Path to the test data file')
    parser.add_argument('--modelpath', dest='modelpath', default='models/conve.pt', help='Path to save the trained model to')
    
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