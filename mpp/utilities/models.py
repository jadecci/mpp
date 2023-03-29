import numpy as np

from sklearn.linear_model import ElasticNetCV

def elastic_net(train_x, train_y, test_x, test_y, n_alphas):
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas)
    en.fit(train_x, train_y)

    test_ybar = en.predict(test_x)
    r = np.corrcoef(test_y, test_ybar)[0, 1]
    cod = en.score(test_x, test_y)

    return r, cod, en.coef_

def permutation_test(acc, null_acc):
    all_acc = np.vstack((acc, null_acc))
    rank = all_acc.argsort(axis=0)
    ind = [rank.shape[0] - np.where(rank[:, i]==0)[0][0] for i in range(rank.shape[1])]
    p = np.divide(ind, all_acc.shape[0])

    return p