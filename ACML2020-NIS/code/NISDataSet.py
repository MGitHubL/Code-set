from torch.utils.data import Dataset
import numpy as np
import sys


class NISDataSet(Dataset):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform

        self.max_d_time = -sys.maxsize

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()

        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                d_time = float(parts[2])
                self.node_set.update([s_node, t_node])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1


        self.node_dim = len(self.node_set)

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0

        for s_node in self.node2hist:
            for s_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = s_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        s_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][s_idx][0]
        c_time = self.node2hist[s_node][s_idx][1]

        if s_idx - self.hist_len < 0:
            s_hist = self.node2hist[s_node][0:s_idx]
        else:
            s_hist = self.node2hist[s_node][s_idx - self.hist_len:s_idx]

        s_hist_nodes = [h[0] for h in s_hist]
        s_hist_times = [h[1] for h in s_hist]

        s_h_nodes = np.zeros((self.hist_len,))
        s_h_nodes[:len(s_hist_nodes)] = s_hist_nodes
        s_h_times = np.zeros((self.hist_len,))
        s_h_times[:len(s_hist_times)] = s_hist_times

        t_idx = self.node2hist[t_node].index((s_node, c_time))

        if t_idx - self.hist_len < 0:
            t_hist = self.node2hist[t_node][0:t_idx]
        else:
            t_hist = self.node2hist[t_node][t_idx - self.hist_len:t_idx]

        t_hist_nodes = [h[0] for h in t_hist]
        t_hist_times = [h[1] for h in t_hist]

        t_h_nodes = np.zeros((self.hist_len,))
        t_h_nodes[:len(t_hist_nodes)] = t_hist_nodes
        t_h_times = np.zeros((self.hist_len,))
        t_h_times[:len(t_hist_times)] = t_hist_times

        n_nodes = self.negative_sampling()
        nodes_len = len(n_nodes)
        n_idxs = [0 for x in range(nodes_len)]
        n_hists = [0 for x in range(nodes_len)]
        n_h_nodes = np.zeros((nodes_len, self.hist_len))
        n_h_times = np.zeros((nodes_len, self.hist_len))

        for i in range(len(n_nodes)):
            for the_id in range(len(self.node2hist[n_nodes[i]])):
                if c_time <= self.node2hist[n_nodes[i]][the_id][1]:
                    if the_id <= 1:
                        n_idxs[i] = 1
                    else:
                        n_idxs[i] = the_id - 1
                    break

        for i in range(len(n_idxs)):
            if n_idxs[i] == 1:
                n_hists[i] = self.node2hist[n_nodes[i]][0:1]
            elif n_idxs[i] == 0:
                n_hists[i] = [(0, -sys.maxsize)]
            else:
                n_hists[i] = self.node2hist[n_nodes[i]][n_idxs[i] - self.hist_len:n_idxs[i]]
            n_hist_nodes = [h[0] for h in n_hists[i]]
            n_hist_times = [h[1] for h in n_hists[i]]
            n_h_nodes[i][:len(n_hist_nodes)] = n_hist_nodes
            n_h_times[i][:len(n_hist_times)] = n_hist_times

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'current_time': c_time,
            'source_history_nodes': s_h_nodes,
            'source_history_times': s_h_times,
            'target_history_nodes': t_h_nodes,
            'target_history_times': t_h_times,
            'neg_nodes': n_nodes,
            'neg_history_nodes': n_h_nodes,
            'neg_history_times': n_h_times,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
