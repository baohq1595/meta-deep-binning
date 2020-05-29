from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans
import numpy as np

import tqdm

from metadec.utils.metrics import genome_acc

class MetricsLog(callbacks.Callback):
    def __init__(self, x, y, groups, n_clusters):
        self.x = x
        self.y = y
        self.groups = groups
        self.n_clusters = n_clusters
        super(MetricsLog, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            # Define model by using input layer and encoder layer that produces latent feature
            feature_model = Model(self.model.input,
                                    self.model.get_layer('encoder_latent').output)

            # Get latent feature given input
            features = feature_model.predict(self.x)

            # Using kmeans to cluster the data
            km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
            y_pred = km.fit_predict(features)

            # Logging results
            print(' '*8 + '|==>  f1-score: %.4f <==|' % (genome_acc(self.groups, y_pred, self.y, self.n_clusters)[2]))

# class TqdmCallbackWithLog(tqdm.keras.TqdmCallback):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         super(TqdmCallbackWithLog, self).__init__()

#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % 10 == 0:
#             # Define model by using input layer and encoder layer that produces latent feature
#             feature_model = Model(self.model.input,
#                                     self.model.get_layer('encoder_latent').output)

#             # Get latent feature given input
#             features = feature_model.predict(self.x)

#             # Using kmeans to cluster the data
#             km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
#             y_pred = km.fit_predict(features)

#             # Calculate metrics
#             precision, recall, f1 = genome_acc(grps, y_pred, self.y, n_clusters)

#             logs['precision'] = precision
#             logs['recall'] = recall
#             logs['f1-measure'] = f1

#             tqdm.tqdm.write(f'Precision: {precision}\nRecall: {recall}\nF1-score: {f1}')