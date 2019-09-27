import time
import numpy as np
import tensorflow as tf

from models import GAT, HeteGAT, HeteGAT_multi
from utils import process

# 禁用gpu
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'freebase'
featype = 'fea'
checkpt_file = '{}.ckpt'.format(dataset)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 2000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [64,32]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp

def set_seeds(seed=0):
    np.random.seed(seed)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

scheme_ind = {'dblp':['APA','APAPA','APCPA'],'yelp':['BRURB','BRKRB'],'freebase':['MAM','MDM','MWM']}
feat_ind = {'dblp':None,'yelp':None,'freebase':None}
label_ind = {'dblp':'labels','yelp':'true_cluster','freebase':'labels'}
path_ind = {'dblp':'~/Git/HINGCN/data/dblp2/',
        'yelp':'~/Git/HINGCN/data/yelp/',
        'freebase':'~/Git/HINGCN/data/freebase/'}

def load_data(dataset):
    scheme = scheme_ind[dataset]
    feat_file = feat_ind[dataset]
    label_file = label_ind[dataset]
    path = os.path.expanduser(path_ind[dataset])

    set_seeds(42)
    data = []

    for s in scheme:
        data.append(sp.load_npz("{}{}_ps.npz".format(path, s) ).todense())


    if dataset == 'freebase':
        movies = []
        with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
            for line in f:
                movies.append(line.split()[0])

        n_movie = len(movies)
        movie_dict = {a: i for (i, a) in enumerate(movies)}
        labels_raw = []
        with open('{}{}.txt'.format(path, label_file), 'r', encoding='UTF-8') as f:
            for line in f:
                arr = line.split()
                labels_raw.append([int(movie_dict[arr[0]]), int(arr[1])])
        labels_raw = np.asarray(labels_raw)

        labels = np.zeros(n_movie,dtype=np.int32)
        labels[labels_raw[:, 0]] = labels_raw[:, 1]
    else:
        labels = np.genfromtxt("{}{}.txt".format(path, label_file),
                           dtype=np.int32)

    reordered = np.random.permutation(np.arange(labels.shape[0]))
    total_labeled = labels.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    y = np.zeros((total_labeled,3))
    for i in range(total_labeled):
        y[i,labels[i]] = 1

    if feat_file is not None:
        feat = np.genfromtxt("{}{}.txt".format(path, feat_file),
                             dtype=np.float)
        features = feat[:, :2]
    else:
        features = np.eye(total_labeled)

    rownetworks = data

    train_idx = idx_train
    val_idx = idx_val
    test_idx = idx_test

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    # y_train[train_idx] = y[train_idx]
    # y_val[val_idx] = y[val_idx]
    # y_test[test_idx] = y[test_idx]
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [features for _ in range(len(scheme))]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_dblp(path='/home/jhy/allGAT/acm_hetesim/ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
if featype == 'adj':
    fea_list = adj_list



import scipy.sparse as sp




nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    y_pred = tf.argmax(log_resh, axis=1)
    y_true = tf.argmax(lab_resh, axis=1)
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
           
            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}
          
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            # import pdb; pdb.set_trace()
            # print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step))

            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0
        pred=[]
        true=[]
        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
            
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}
        
            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding,y_pred,y_true = sess.run([loss, accuracy, final_embedding,y_pred,y_true],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
            pred.append(y_pred)
            true.append(y_true)
        from sklearn import metrics
        pred=np.concatenate(pred)
        true=np.concatenate(true)

        #print(pred.shape,true.shape)


        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step,
              "micro" , float(metrics.f1_score(true, pred,sample_weight=test_mask[0], average="micro")),
              "macro" , float(metrics.f1_score(true, pred,sample_weight=test_mask[0], average="macro")),
              )

        # print('start knn, kmean.....')
        # xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]
  
        # from numpy import linalg as LA

        # # xx = xx / LA.norm(xx, axis=1)
        # yy = y_test[test_mask]

        # print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        # from jhyexp import my_KNN, my_Kmeans#, my_TSNE, my_Linear

        # my_KNN(xx, yy)
        # my_Kmeans(xx, yy)

        sess.close()
