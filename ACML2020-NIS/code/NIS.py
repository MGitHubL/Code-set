import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import sys
from NISDataSet import NISDataSet
import datetime

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class NIS:
    def __init__(self, file_path, emb_size=128, neg_size=5, hist_len=5, directed=False,
                 learning_rate=0.001, batch_size=128, save_step=50, epoch_num=30):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = NISDataSet(file_path, neg_size, hist_len, directed)
        self.node_dim = self.data.get_node_dim()

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.delta_1 = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_2 = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)
            self.delta_1 = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)
            self.delta_2 = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

        self.opt = torch.optim.Adam(params=[self.node_emb, self.delta_1, self.delta_2], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.98)
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes, t_h_times,
                n_nodes, n_h_nodes, n_h_times):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        s_h_node_emb = self.node_emb.index_select(0, Variable(s_h_nodes.view(-1))).view(batch, self.hist_len, -1)
        t_h_node_emb = self.node_emb.index_select(0, Variable(t_h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1)).long()).view(batch, self.neg_size, -1)
        n_h_node_emb = self.node_emb.index_select(0, Variable(n_h_nodes.view(-1)).long()).view(batch, self.neg_size, self.hist_len, -1)
        delta_1 = self.delta_1.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        delta_2 = self.delta_2.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)

        s_h_d_time = 1 / (1 + torch.abs(c_times.unsqueeze(1) - s_h_times))
        s_h_d_time = s_h_d_time / s_h_d_time.sum(dim=-1, keepdim=True)
        t_h_d_time = 1 / (1 + torch.abs(c_times.unsqueeze(1) - t_h_times))
        t_h_d_time = t_h_d_time / t_h_d_time.sum(dim=-1, keepdim=True)
        s_n_vector = torch.mul(s_h_node_emb, s_h_d_time.unsqueeze(-1)).sum(dim=1, keepdim=False)
        t_n_vector = torch.mul(t_h_node_emb, t_h_d_time.unsqueeze(-1)).sum(dim=1, keepdim=False)
        p_miu = ((s_node_emb - t_node_emb) ** 2).sum(dim=-1, keepdim=True)
        p_hawkes = torch.mul(delta_1, (((s_n_vector - t_node_emb) ** 2) +
                                       ((t_n_vector - s_node_emb) ** 2)).sum(dim=-1, keepdim=True))
        p_neighbor = torch.mul(delta_2, ((s_n_vector - t_n_vector) ** 2).sum(dim=-1, keepdim=True))
        p_lambda = torch.exp((p_miu + p_hawkes + p_neighbor).sum(dim=-1, keepdim=False)).neg()

        n_h_d_time = 1 / (1 + torch.abs((c_times.unsqueeze(-1)).unsqueeze(-1) - n_h_times))
        n_h_d_time = n_h_d_time / (n_h_d_time.sum(dim=-1, keepdim=True) + 1e-6)
        n_n_vector = torch.mul(n_h_node_emb, n_h_d_time.unsqueeze(-1)).sum(dim=2, keepdim=False)
        n_miu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=-1, keepdim=True)
        n_hawkes = torch.mul(delta_1.unsqueeze(1), (((s_n_vector.unsqueeze(1) - n_node_emb) ** 2) +
                                                    ((n_n_vector - s_node_emb.unsqueeze(1)) ** 2)).sum(dim=-1, keepdim=True))
        n_neighbor = torch.mul(delta_2.unsqueeze(1), ((s_n_vector.unsqueeze(1) - n_n_vector) ** 2).sum(dim=-1, keepdim=True))
        n_lambda = torch.exp((n_miu + n_hawkes + n_neighbor).sum(dim=-1, keepdim=False)).neg()

        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes, t_h_times,
                  n_nodes, n_h_nodes, n_h_times):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes,
                                                    t_h_times, n_nodes, n_h_nodes, n_h_times)
                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)
        else:
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes,
                                                t_h_times, n_nodes, n_h_nodes, n_h_times)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes, t_h_times,
               n_nodes, n_h_nodes, n_h_times):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes, t_h_times,
                                      n_nodes, n_h_nodes, n_h_times)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, c_times, s_h_nodes, s_h_times, t_h_nodes, t_h_times,
                                  n_nodes, n_h_nodes, n_h_times)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings('./emb/dblp_nis_%d.emb' % epoch)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta_1: ' + str(
                        self.delta_1.mean().cpu().data.numpy()) + '\tdelta_2: ' + str(
                        self.delta_2.mean().cpu().data.numpy()))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['current_time'].type(FType).cuda(),
                                    sample_batched['source_history_nodes'].type(LType).cuda(),
                                    sample_batched['source_history_times'].type(FType).cuda(),
                                    sample_batched['target_history_nodes'].type(LType).cuda(),
                                    sample_batched['target_history_times'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(FType).cuda(),
                                    sample_batched['neg_history_nodes'].type(FType).cuda(),
                                    sample_batched['neg_history_times'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['current_time'].type(FType),
                                sample_batched['source_history_nodes'].type(LType),
                                sample_batched['source_history_times'].type(FType),
                                sample_batched['target_history_nodes'].type(LType),
                                sample_batched['target_history_times'].type(FType),
                                sample_batched['neg_nodes'].type(FType),
                                sample_batched['neg_history_nodes'].type(FType),
                                sample_batched['neg_history_times'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) +
                             '\ndelta_1: ' + str(self.delta_1.mean().cpu().data.numpy()) +
                             '\tdelta_2: ' + str(self.delta_2.mean().cpu().data.numpy()) +
                             '\tlr: ' + str(self.scheduler.get_lr()) + '\n\n')
            sys.stdout.flush()
            self.scheduler.step()
        self.save_node_embeddings('./emb/dblp_nis_%d.emb' % (self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    nis = NIS('../data/dblp/dblp.txt', directed=False)
    nis.train()
    end = datetime.datetime.now()
    print(end - start)
