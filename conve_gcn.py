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
#from torch_geometric.nn import GCNConv
import dgl
from dgl.data import DGLDataset
from dgl.nn import GraphConv, DenseGraphConv
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import pickle
import pandas as pd

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ConvE_GCN(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, adjmatrix):
        super(ConvE_GCN, self).__init__()
        # Embedding tables for entity and relations with num_uniq_ent in y-dim, emb_dim in x-dim
        self.emb_e = torch.nn.Embedding(num_entities, args.emb_dim, padding_idx=0)
        self.ent_weights_matrix = torch.ones([num_entities, args.emb_dim], dtype=torch.float64)
        self.emb_rel = torch.nn.Embedding(num_relations, args.emb_dim, padding_idx=0)
        self.ne = num_entities
        self.nr = num_relations
        self.adjmatrix = adjmatrix
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

        # GCN related
        # self.preprocess_dgldata()
        # self.gcn_conv1 = DenseGraphConv(args.emb_dim, self.ne)
        
        self.gcn_conv1 = GraphConvolution(args.emb_dim, args.gcn_dim)
        self.gcn_conv2 = GraphConvolution(args.gcn_dim, self.ne)
        
        self.gcn_fc = torch.nn.Linear(self.ne * self.ne, self.ne)
    
    def init(self):
        # Xavier initialization
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

        # Pre-trained embeddings initialization
        #self.init_flairemb()
        #self.emb_e.load_state_dict({'weight': self.ent_weights_matrix})
    
    def preprocess_dgldata(self):
        edges_data = pd.read_csv(args.interactionsfile)
        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        #self.edge_index = torch.cat((edges_src, edges_dst), 0)
        sp_mat = coo_matrix((edge_features, (edges_src, edges_dst)), shape=(self.ne, self.ne))
        self.graph = dgl.from_scipy(sp_mat)
        #self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.ne)
        self.graph.ndata['feat'] = self.emb_e.weight
        self.graph.ndata['label'] = self.adjmatrix
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.ne
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        self.graph = dgl.add_self_loop(self.graph)

    
    def forward(self, e1, rel, adjmatrix, print_pred=False):
        batch_size = 1
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 30)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 30)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        # KG related 
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

        # # Attention mask from Pytorch geometric
        # data = Data(x=self.emb_e.weight, edge_index=self.edge_index)
        # gcn_x = F.relu(self.gcn_conv1(data.x, data.edge_index))
        # gcn_x = self.gcn_conv2(gcn_x, data.edge_index)
        # print('GCN_x shape:{}'.format(gcn_x.shape))
        # att_mask = torch.nn.functional.softmax(x, dim=1)

        # tkipf implementation of GCNs
        gcn_x = F.relu(self.gcn_conv1(self.emb_e.weight, adjmatrix))
        gcn_x = self.gcn_conv2(gcn_x, adjmatrix)
        gcn_x = gcn_x.view(self.ne, self.ne)
        print('GCN_x shape:{}'.format(gcn_x.shape))
        att_mask = torch.nn.functional.softmax(gcn_x, dim=1)

        # DGL implementation
        # gcn_x = F.relu(self.gcn_conv1(self.graph, self.emb_e.weight))
        # att_mask = torch.nn.functional.softmax(gcn_x, dim=1)
        
        # # Scaling based on attention mask
        x = torch.mul(x, att_mask)
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
        self.num_entities = len(self.entity_ids)
        #self.ids2entities =  self.id2ent(self.ent_path) 
        self.rel_ids =  self.load_data(self.rel_path)
        #self.ids2rel =  self.id2rel(self.rel_path) 
        self.train_triples_map = self.convert_triples(self.train_file)
        #self.test_triples_list = self.convert_triples(self.test_file)

        # Specifically related to GCNs
        #self.adj_matrix = self.create_adj_matrix(args.adjrelpath)
        self.adj_matrix = self.create_adj_matrix(adjrelpath=None)
        # self.I = np.eye(*self.adj_matrix.shape)
        # self.A_hat = self.adj_matrix.copy() + self.I
        # self.degree_matrix = self.create_deg_matrix()
        # self.A_hat_norm = np.dot(self.degree_matrix, self.A_hat)

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
        triples_map = {}
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                item_dict = json.loads(line.strip())
                e1 = item_dict['e1']
                rel = item_dict['rel']
                t = item_dict['e2_multi1'].split('\t')
                if (e1 , rel) not in triples_map:
                    triples_map[(e1, rel)] = set()
                t_ents = []
                for t_idx in t:
                    t_ents.append(self.entity_ids[t_idx])
                triples_map[(e1, rel)].add(tuple(t_ents))
        return triples_map
    
    def create_adj_matrix(self, adjrelpath=None):
        adj_matrix =  None
        if adjrelpath:
            with open(adjrelpath) as f:
                lines = f.readlines()
                adj_matrix = np.zeros((self.num_entities, self.num_entities))
                for line in lines:
                    c1, rel, c2 = line.strip().split('\t')
                    #print('C1:{} Rel:{} C2:{}'.format(c1,rel,c2))
                    c1_id = self.entity_ids[c1]
                    c2_id = self.entity_ids[c2]
                    adj_matrix[c1_id][c2_id] = 1
            
            with open('data/wikidata/adjmatrix.pkl','wb') as f:
                pickle.dump(adj_matrix, f)
        else:
            print('Loading adjacency matrix from pickle file')
            with open('data/wikidata/adjmatrix.pkl','rb') as f:
                adj_matrix = pickle.load(f)
        return adj_matrix
    
    def create_deg_matrix(self):
        deg_matrix = np.sum(self.A_hat, axis=0)
        deg_matrix = np.diag(deg_matrix)

        # For spectral convolutions, get the inverse degree matrix
        # deg_matrix_inv = deg_matrix** -0.5
        # deg_matrix_inv = np.diag(deg_matrix_inv)
        
        return deg_matrix

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def train(args):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)
    triples_map = data.train_triples_map
    adj = sp.coo_matrix(data.adj_matrix)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    num_train_samples = len(triples_map)
    num_train_steps = math.ceil(num_train_samples/int(args.batch_size))
    
    model = ConvE_GCN(args, num_entities, num_relations, adj)
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
            #train_sample = triples_map[counter:counter+int(args.batch_size)]
            train_samples = {k: triples_map[k] for k in list(triples_map)[counter:counter+int(args.batch_size)]}
            h,r,t = train_sample[0]
            logits = model.forward(torch.tensor(h), torch.tensor(r), adj, print_pred=False)
            for t_id in t:
                e2_multi[0][t_id] = 1.0
            loss = model.loss(logits, e2_multi.clone().detach())
            loss.backward()
            optimizer.step()
            batch_loss = torch.sum(loss.detach())
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
    normadjmatrix = data.A_hat_norm

    names_dict = data.ids2entities
    rel_dict = data.ids2rel

    model = ConvE_GCN(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    triples_list = data.test_triples_list
    num_test_samples = len(triples_list)
    for i in range(num_test_samples):
        test_sample = triples_list[i]
        print('Testing sample:{}'.format(test_sample))
        h,r,t = test_sample
        logits = model.forward(torch.tensor(h), torch.tensor(r), torch.tensor(normadjmatrix), print_pred=False)
        score, pred = torch.topk(logits,args.top_k,1)

        print("--------------------------------------------------------------------------")
        for j, id in enumerate(pred[0].cpu().detach().numpy()): 
            pred_entity = names_dict[id]
            print('id:{} Head:{}, Relation:{}, Pred:{}, Target:{}'.format(id, names_dict[h], rel_dict[r], names_dict[id], names_dict[t[0]]))

def infer(args, ent, rel):
    data = WikiData(args)
    num_entities =  len(data.entity_ids)
    num_relations =  len(data.rel_ids)
    normadjmatrix = torch.tensor(data.A_hat_norm, dtype=torch.long)
    normadjmatrix = Tensor.long(normadjmatrix)

    entitynames_dict = data.ids2entities

    model = ConvE_GCN(args, num_entities, num_relations)
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    h_idx = data.entity_ids[ent]
    r_idx = data.rel_ids[rel]

    print('h idx:{}'.format(h_idx))
    print('r idx:{}'.format(r_idx))

    logits = model.forward(torch.tensor(h_idx), torch.tensor(r_idx), normadjmatrix, print_pred=False)
    score, pred = torch.topk(logits,args.top_k,1)

    print("--------------------------------------------------------------------------")
    for j, id in enumerate(pred[0].cpu().detach().numpy()): 
        pred_entity = entitynames_dict[id]
        print('id:{} Head:{}, Relation:{}, Pred:{}'.format(id, ent, rel, entitynames_dict[id]))

def main(args):

    if args.do_train:
        #Model training
        train(args)

    # if args.do_eval:
    #     #Model evaluation
    #     eval(args)

    # Model inference
    #infer(args, 'a fetus', 'HasShape_reverse')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', dest='do_train', default=True, help='Whether to train the model or do eval only')
    parser.add_argument('--do_eval', dest='do_eval', default=True, help='Whether to do evaluation on the model')
    parser.add_argument('--entdatapath', dest='entdatapath', default='data/wikidata/kg_training_entids_withshapes.txt', help='Path to the file containing the entities and entity IDs')
    parser.add_argument('--reldatapath', dest='reldatapath', default='data/wikidata/kg_training_relids_withshapes.txt', help='Path to the file containing the relations and relation IDs')
    parser.add_argument('--traindatapath', dest='traindatapath', default='data/wikidata/e1rel_to_e2_train.json', help='Path to the training data file')
    parser.add_argument('--testdatapath', dest='testdatapath', default='data/wikidata/e1rel_to_e2_ranking_test.json', help='Path to the test data file')
    parser.add_argument('--modelpath', dest='modelpath', default='models/conve_gcn.pt', help='Path to save the trained model to')
    
    # GCN-related params
    parser.add_argument('--gcn_dim', dest="gcn_dim", default=16, type=int, help='GCN hidden dimension')
    parser.add_argument('--adjrelpath', dest='adjrelpath', default='data/wikidata/adjacency_rels_countries.txt', help='Path to the file containing adjacency relations of KG nodes')
    parser.add_argument('--interactionsfile', dest='interactionsfile', default='data/wikidata/gcndata/interactions.csv', help='Path to the interactions file containing adjacency relations of KG nodes')
    
    # Training related params
    parser.add_argument('--batch_size', dest='batch_size', default=1, help='Batch size')
    parser.add_argument('--input_dropout', dest='input_dropout', default=0.2, help='Input dropout for the model')
    parser.add_argument('--dropout', dest='dropout', default=0.3, help='Dropout for the model')
    parser.add_argument('--feature_map_dropout', dest='feature_map_dropout', default=0.2, help='Feature map dropout for the model')
    parser.add_argument('--use_bias', dest='use_bias', default=True, help='Entity and relation embedding dimensions')
    parser.add_argument('--emb_dim', dest='emb_dim', default=300, help='Entity and relation embedding dimensions')
    parser.add_argument('--lr', dest='lr', default=0.001, help='Learning rate')
    parser.add_argument('--l2', dest='l2', default=5e-4, help='L2 regularization')
    parser.add_argument('--label_smoothing_epsilon', dest='label_smoothing_epsilon', default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--epochs', dest='max_epochs', default=1, help='Max epochs')

    # Eval related params    
    parser.add_argument('--top_k', dest='top_k', default=10, help='Top K vals to consider from the predictions')
    
    args = parser.parse_args()

    #Create model
    model = main(args)