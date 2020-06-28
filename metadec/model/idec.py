"""
Implementation for Improved Deep Embedded Clustering as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

Usage:
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python IDEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5
    for USPS and REUTERSIDF10K datasets
        python IDEC.py usps --update_interval 30 --ae_weights ./ae_weights/usps_ae_weights.h5
        python IDEC.py reutersidf10k --n_clusters 4 --update_interval 3 --ae_weights ./ae_weights/reutersidf10k_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30
"""

import time, os
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

from sklearn import metrics
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans

from metadec.model.keras_callback import MetricsLog
from metadec.utils.metrics import genome_acc
from metadec.model.layer import autoencoder, ClusteringLayer

class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 gamma=0.1,
                 init='glorot_uniform',
                 dropout=0.0):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init, dropout)

        # Prepare IDEC model
        latent_layer = self.autoencoder.get_layer(name='encoder_latent').output
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(latent_layer)
        self.model = Model(inputs=self.autoencoder.input, outputs=[clustering_layer, self.autoencoder.output])

    def compile(self, optimizer='sgd', autoencoder_loss='mse', clustering_loss='kld'):
        self.model.compile(loss={'clustering': clustering_loss,
                                'decoder_reconstruction': autoencoder_loss},
                                loss_weights=[self.gamma, 1],
                                optimizer=optimizer)

    def pretrain(self, x, y=None, grps=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp', wandb=False):
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            cb.append(MetricsLog(x, y, grps, self.n_clusters))
        
        if wandb:
            cb.append(WandbCallback())

        # begin pretraining
        t0 = time.time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time.time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    # def initialize_model(self, gamma=0.1, optimizer='adam'):
    #     hidden = self.autoencoder.get_layer(name='encoder_latent').output
    #     self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

    #     # prepare IDEC model
    #     clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
    #     self.model = Model(inputs=self.autoencoder.input,
    #                        outputs=[clustering_layer, self.autoencoder.output])
    #     self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
    #                        loss_weights=[gamma, 1],
    #                        optimizer=optimizer)

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_latent').output)
        return encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x, y=None,
                   grps=None, 
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   batch_size=4,
                   save_dir='./results/idec'):

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
        logfile = open(save_dir + '/idec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'precision', 'recall', 'f1_score', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    # acc = np.round(cluster_acc(y, y_pred), 5)
                    # nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    # ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    prec, recall, f1_score = genome_acc(grps, y_pred, y, self.n_clusters)
                    if f1_score > best_acc:
                        best_acc = f1_score
                        best_model = self.model
                        # save IDEC model checkpoints
                        print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
                        self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, precision=prec, recall=recall, f1_score=f1_score, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter %d: precision = %.5f, recall = %.5f, f1_score = %.5f,\
                            nmi = --, ari = --' % (ite, prec, recall, f1_score), ' ; loss=', loss)

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            # if ite % save_interval == 0:
            #     # save IDEC model checkpoints
            #     print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
            #     self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')
        
        return y_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'reutersidf10k'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/idec')
    args = parser.parse_args()
    print(args)

    # load dataset
    optimizer = SGD(lr=0.1, momentum=0.99)
    from datasets import load_mnist, load_reuters, load_usps

    if args.dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x, y = load_mnist()
        optimizer = 'adam'
    elif args.dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        x, y = load_usps('data/usps')
    elif args.dataset == 'reutersidf10k':  # recommends: n_clusters=4, update_interval=3
        x, y = load_reuters('data/reuters')

    # prepare the IDEC model
    idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=args.n_clusters, batch_size=args.batch_size)
    idec.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time.time()
    y_pred = idec.clustering(x, y=y, tol=args.tol, maxiter=args.maxiter,
                             update_interval=args.update_interval, save_dir=args.save_dir)
    print('acc:', cluster_acc(y, y_pred))
    print('clustering time: ', (time.time() - t0))