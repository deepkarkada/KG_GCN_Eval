import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

np.set_printoptions(threshold=np. inf)

class WikiData():
    def __init__(self, datapath, adjrelpath, feat_dims):
        self.country_dict = self.load_data(datapath)
        self.adj_matrix = self.create_adj_matrix(adjrelpath)
        self.num_nodes, _ = self.adj_matrix.shape
        self.I = np.eye(*self.adj_matrix.shape)
        self.A_hat = self.adj_matrix.copy() + self.I
        self.degree_matrix = self.create_deg_matrix()
        self.num_features = feat_dims
        self.feat_matrix = np.ones((self.num_nodes, self.num_features))
        self.train_mask = self.create_train_mask(adjrelpath)

    def load_data(self, data_path):
        country_dict = {}
        with open(data_path) as df:
            lines = df.readlines()
            for line in lines:
                name, id = line.split('   ')
                country_dict[name] = int(id)     
        #print('Country dict:{}'.format(country_dict))
        return country_dict

        # TODO: Way to get feature matrix from trained conve model

    def create_train_mask(self, adjrelpath):
        train_mask = np.zeros(self.num_nodes)
        with open(adjrelpath) as f:
            lines = f.readlines()
            for line in lines:
                c1, rel, c2 = line.strip().split('\t')
                c1_id = self.country_dict[c1]
                c2_id = self.country_dict[c2]
                train_mask[c1_id] = 1
                train_mask[c2_id] = 1
        return train_mask

    
    def create_adj_matrix(self, adjrelpath):
        with open(adjrelpath) as f:
            lines = f.readlines()
            adj_matrix = np.zeros((len(lines), len(lines)))
            for line in lines:
                c1, rel, c2 = line.strip().split('\t')
                #print('C1:{} Rel:{} C2:{}'.format(c1,rel,c2))
                c1_id = self.country_dict[c1]
                c2_id = self.country_dict[c2]
                adj_matrix[c1_id][c2_id] = 1
        return adj_matrix
    
    def create_deg_matrix(self):
        deg_matrix = np.sum(self.A_hat, axis=0)
        deg_matrix = np.diag(deg_matrix)

        # For spectral convolutions, get the inverse degree matrix
        # deg_matrix_inv = deg_matrix** -0.5
        # deg_matrix_inv = np.diag(deg_matrix_inv)
        
        return deg_matrix

# Ref: https://github.com/svjan5/GNNs-for-NLP/blob/master/pytorch_gcn.py
class KipfGCN(torch.nn.Module):
    def __init__(self, data, params):
        super(KipfGCN, self).__init__()
        self.p     = params
        self.data  = data
        self.conv1 = GCNConv(self.data.num_features, self.p.gcn_dim, cached=True)
        self.conv2 = GCNConv(self.p.gcn_dim, self.data.num_nodes,  cached=True)

    def forward(self, x, edge_index):
        print(x.shape)
        print(edge_index.shape)
        x		= F.relu(self.conv1(x, edge_index))
        #x		= F.dropout(x, p=self.p.dropout, training=self.training)
        x		= self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Main():

    def __init__(self, params):
        self.params = params
        self.data_path = self.params.datapath
        self.adjrel_path = self.params.adjrelpath
        self.feat_dims = self.params.feat_dim

        self.data = WikiData(self.data_path, self.adjrel_path, self.feat_dims)
        # print('Adjacency matrix:{}'.format(self.data.adj_matrix))
        # print('Degree matrix:{}'.format(self.data.degree_matrix))

        model = KipfGCN(self.data, self.params)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.l2)

        # Model fitting
        model.train()
        optimizer.zero_grad()
        for ep in range(self.params.max_epochs):
            logits = model.forward(torch.from_numpy(self.data.feat_matrix), torch.from_numpy(self.data.adj_matrix))
            train_loss	= F.nll_loss(logits, torch.from_numpy(self.data.train_mask))
            train_loss.backward()
            self.optimizer.step()

        # Model evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', dest='datapath', help='Path to the file containing the list of KG nodes')
    parser.add_argument('--adjrelpath', dest='adjrelpath', help='Path to the file containing adjacency relations of KG nodes')
    parser.add_argument('--feat_dim', dest="feat_dim", default=800, type=int, help='Features dimensions')

    # GCN-related params
    parser.add_argument('--gcn_dim', dest="gcn_dim", default=16, type=int, help='GCN hidden dimension')

    # Optimizer related params
    parser.add_argument('--lr', dest="lr", default=0.01, type=float, help='Learning rate')
    parser.add_argument('--epoch', dest="max_epochs", default=200, type=int, help='Max epochs')
    parser.add_argument('--l2', dest="l2", default=5e-4, type=float, help='L2 regularization')

    args = parser.parse_args()

    # Create model
    model = Main(args)
