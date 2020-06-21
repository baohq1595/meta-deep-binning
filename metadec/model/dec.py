import time, os
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans

from metadec.model.keras_callback import MetricsLog
from metadec.utils.metrics import genome_acc
from metadec.model.layer import autoencoder, ClusteringLayer

class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, grps=None, n_clusters=2, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp', wandb=False):
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            cb.append(MetricsLog(x, y, grps, n_clusters))
        
        if wandb:
            cb.append(WandbCallback())

        # begin pretraining
        t0 = time.time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time.time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, grps=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/dec'):

        best_model = self.model
        best_acc = 0.0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        if save_interval == 0:
            save_interval = 10
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time.time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'precision', 'recall', 'f1_score', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    # acc = np.round(metrics.acc(y, y_pred), 5)
                    # nmi = np.round(metrics.nmi(y, y_pred), 5)
                    # ari = np.round(metrics.ari(y, y_pred), 5)
                    prec, recall, f1_score = genome_acc(grps, y_pred, y, self.n_clusters)
                    if f1_score > best_acc:
                      best_acc = f1_score
                      best_model = self.model
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, precision=prec, recall=recall, f1_score=f1_score, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: precision = %.5f, recall = %.5f, f1_score = %.5f,\
                            nmi = --, ari = --' % (ite, prec, recall, f1_score), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        best_model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred