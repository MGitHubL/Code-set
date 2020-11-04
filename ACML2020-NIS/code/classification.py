from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def format_training_data_for_dnrl(emb_file, i2l_file):
    i2l = dict()
    with open(i2l_file, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            i2l[n_id] = l_id

    i2e = dict()
    with open(emb_file, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = embeds[0]
            if node_id in i2l:
                i2e[node_id] = embeds[1:]

    Y = []
    X = []
    i2l_list = sorted(i2l.items(), key=lambda x: x[0])
    for (the_id, label) in i2l_list:
        Y.append(label)
        X.append(i2e[the_id])

    X = np.stack(X)
    return X, Y


def lr_classification(X, Y, cv):
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, Y, cv=cv, scoring='f1_micro', n_jobs=8)
    scores = scores.sum() / 6
    return scores


if __name__ == '__main__':
    X, Y = format_training_data_for_dnrl('./emb/dblp_nis_30.emb', '../data/dblp/node2label.txt')
    print(lr_classification(X, Y, cv=6))
