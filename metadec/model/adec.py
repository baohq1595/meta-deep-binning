import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, InputSpec
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Reshape, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import keras.backend as K

import matplotlib.pyplot as plt

import os
import io
from PIL import Image

import numpy as np
import tqdm
import os
from collections import defaultdict

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from cuml.manifold import TSNE # If available, use it or
from tsnecuda import TSNE # If available, use it
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert input_shape.ndims == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))

        return q
        # q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        # q **= (self.alpha + 1.0) / 2.0
        # q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        # return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ADEC():
    def __init__(self, n_clusters, ae_dims, lambda_coef,
                critic_dims=None, discriminator_dims=None,
                dropout=0.4, initializer=None):
        self.n_clusters = n_clusters
        self.ae_dims = ae_dims
        self.lambda_coef = lambda_coef
        self.dropout = dropout
        self.critic_dims = ae_dims if critic_dims is None else critic_dims
        self.discriminator_dims = ae_dims if discriminator_dims is None else discriminator_dims
        self.tol = 0.001

        if initializer == None:
            self.initializer = tf.keras.initializers.VarianceScaling(
                            scale=1. / 3., mode='fan_in', distribution='uniform')
        else:
            self.initializer = initializer

        self.initialize_model()
        self.init_pretrain_optims()
        self.init_cluster_optims()

    def init_pretrain_optims(self):
        self.ae_optim = Adam(0.0001, epsilon=1e-8)
        self.critic_optim = Adam(0.0001)

        return self.ae_optim, self.critic_optim

    def init_cluster_optims(self):
        # self.cluster_optim = SGD(learning_rate=0.001, momentum=0.9)
        self.cluster_optim = Adam(0.0001, epsilon=1e-8)
        return self.cluster_optim


    def initialize_model(self):
        # Initialize model component
        self.encoder = self.build_encoder(activation=tf.nn.relu)
        self.decoder = self.build_decoder(activation=tf.nn.relu, out_activation='sigmoid')
        self.critic = self.build_critic(activation=tf.nn.relu)
        self.discriminator = self.build_discriminator(activation='relu', out_activation='sigmoid')

        # Summary
        # self.encoder.summary()
        # self.decoder.summary()
        # self.critic.summary()
        # self.discriminator.summary()

        inp = Input(shape=(self.ae_dims[0],))
        latent = self.encoder(inp)
        decoded = self.decoder(latent)
        self.autoencoder = Model(inputs=inp, outputs=decoded)
        clustering_layer = ClusteringLayer(n_clusters=self.n_clusters, name='clustering')
        cluster_center = clustering_layer(self.encoder.output)
        self.cluster = Model(inputs=self.encoder.input, outputs=cluster_center)


    def build_encoder(self, activation='relu'):
        x = Input(shape=(self.ae_dims[0],), name='enc_input')

        h = x
        for i, dim in enumerate(self.ae_dims[1:-1]):
            h = Dense(dim, activation=activation, kernel_initializer=self.initializer, name=f'enc_layer_{i+1}')(h)
            h = Dropout(self.dropout)(h)

        latent = Dense(self.ae_dims[-1], kernel_initializer=self.initializer, name='enc_latent')(h)

        return Model(inputs=x, outputs=latent, name='Encoder')

    def build_decoder(self, activation='relu', out_activation=None):
        dims = list(reversed(self.ae_dims))

        x = Input(shape=(dims[0],), name='dec_input')
        h = Dense(dims[1], activation=activation, kernel_initializer=self.initializer, name='dec_layer_0')(x)
        h = Dropout(self.dropout)(h)
        for i, dim in enumerate(dims[2:-1]):
            h = Dense(dim, activation=activation, kernel_initializer=self.initializer, name=f'dec_layer_{i+2}')(h)
            h = Dropout(self.dropout)(h)

        reconstruction = Dense(dims[-1], activation=out_activation, kernel_initializer=self.initializer, name='dec_final')(h)

        return Model(inputs=x, outputs=reconstruction, name='Decoder')

    def build_critic(self, activation='relu'):
        x = Input(shape=(self.critic_dims[0],), name='critic_input')

        h = x
        for i, dim in enumerate(self.critic_dims[1:-1]):
            h = Dense(dim, activation=activation, kernel_initializer=self.initializer, name=f'critic_layer_{i+1}')(h)
            h = Dropout(self.dropout)(h)

        out = Dense(self.critic_dims[-1], kernel_initializer=self.initializer, name='critic_final')(h)
        out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(out)

        return Model(inputs=x, outputs=out, name='Critic')

    def build_discriminator(self, activation='relu', out_activation=None):
        x = Input(shape=(self.discriminator_dims[0],), name='discriminator_input')
        h = x
        for i, dim in enumerate(self.discriminator_dims[1:-1]):
            h = Dense(dim, activation=activation, kernel_initializer=self.initializer, name=f'discriminator_layer_{i+1}')(h)
            h = Dropout(self.dropout)(h)

        out = Dense(self.discriminator_dims[-1], activation=out_activation, kernel_initializer=self.initializer, name='discriminator_final')(h)

        return Model(inputs=x, outputs=out, name='discriminator')

    def save(self, save_path):
        ae_path = os.path.join(save_path, 'autoencoder')
        d_path = os.path.join(save_path, 'discriminator')
        cr_path = os.path.join(save_path, 'critic')
        cl_path = os.path.join(save_path, 'cluster')

        self.encoder.save(os.path.join(ae_path, 'encoder'))
        self.decoder.save(os.path.join(ae_path, 'decoder'))
        self.cluster.save(cl_path)
        self.critic.save(cr_path)
        self.discriminator.save(d_path)

    def load(self, load_path):
        ae_path = os.path.join(load_path, 'autoencoder')
        d_path = os.path.join(load_path, 'discriminator')
        cr_path = os.path.join(load_path, 'critic')
        cl_path = os.path.join(load_path, 'cluster')

        # Load encoder/decoder
        self.encoder = tf.keras.models.load_model(os.path.join(ae_path, 'encoder'))
        self.decoder = tf.keras.models.load_model(os.path.join(ae_path, 'decoder'))

        # Reset module autoencoder
        inp = Input(shape=(self.ae_dims[0],))
        latent = self.encoder(inp)
        decoded = self.decoder(latent)
        self.autoencoder = Model(inputs=inp, outputs=decoded)

        # Load discriminator
        self.discriminator = tf.keras.models.load_model(d_path)

        # Load critic
        self.critic = tf.keras.models.load_model(cr_path)

        # Load cluster
        self.cluster = tf.keras.models.load_model(cl_path)


    @staticmethod
    # @tf.function
    def target_distribution(q):
        weight = q ** 2 / tf.reduce_sum(q, axis=0)
        return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))


    @tf.function
    def pretrain_on_batch(self, x, y):
        # Randomzie interpolated coefficient alpha
        alpha = tf.random.uniform((x.shape[0], 1), 0, 1)
        alpha = 0.5 - tf.abs(alpha - 0.5)  # Make interval [0, 0.5]

        a_gamma = tf.random.uniform((x.shape[0], 1), 0, 1)[0][0]
        gamma = tf.fill((x.shape[0], 1), a_gamma)

        with tf.GradientTape() as ae_tape, tf.GradientTape() as critic_tape:
            # Constructs non-interpolated latent space and decoded input
            latent = self.encoder(x, training=True)
            res_x = self.decoder(latent, training=True)

            # Reconstruction loss
            ae_loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.reshape(x, (x.shape[0], -1)), tf.reshape(res_x, (res_x.shape[0], -1))))

            # Interpolated latent space and reconstruction of it
            inp_latent = alpha * latent + (1 - alpha) * latent[::-1]
            res_x_hat = self.decoder(inp_latent, training=True)

            # Get the predicited alpha value from critic
            pred_alpha = self.critic(res_x_hat, training=True)
            temp = self.critic(res_x + gamma * (x - res_x), training=True)

            # Critic losses
            critic_loss_term_1 = tf.reduce_mean(tf.square(pred_alpha - alpha))
            critic_loss_term_2 = tf.reduce_mean(tf.square(temp))

            # Total loss for autoencoder
            reg_ae_loss = self.lambda_coef * tf.reduce_mean(tf.square(pred_alpha))
            total_ae_loss = ae_loss + reg_ae_loss

            # Total loss for critic
            total_critic_loss = critic_loss_term_1 + critic_loss_term_2


        # Computing gradient and perform backward
        grad_ae = ae_tape.gradient(total_ae_loss, self.autoencoder.trainable_variables)
        grad_critic = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)

        self.ae_optim.apply_gradients(zip(grad_ae, self.autoencoder.trainable_variables))
        self.critic_optim.apply_gradients(zip(grad_critic, self.critic.trainable_variables))

        return {
            'res_ae_loss': ae_loss,
            'reg_ae_loss': reg_ae_loss,
            'critic_loss': critic_loss_term_1,
            'reg_critic_loss': critic_loss_term_2

        }

    @tf.function
    def pretrain_on_batch_phase2(self, x, y):
        # First, train the discriminator
        ## Split the input into 2 halfs, one for real, one for fake
        half_batch = x.shape[0] // 2
        x_half_1 = x[:half_batch,:]
        x_half_2 = x[half_batch:, :]

        with tf.GradientTape() as d_tape:
            d_fake = tf.zeros((x_half_2.shape[0], 1))
            d_real = tf.ones((x_half_1.shape[0], 1))

            d_real_logits = self.discriminator(x_half_1, training=True)

            # Get fake pred from discriminator and computing the losses
            latent_half_2 = self.encoder(x_half_2, training=False)
            x_hat_half_2 = self.decoder(latent_half_2, training=False)

            d_fake_logits = self.discriminator(x_hat_half_2, training=True)
            d_real_loss = tf.reduce_mean(tf.losses.binary_crossentropy(d_real, d_real_logits))
            d_fake_loss = tf.reduce_mean(tf.losses.binary_crossentropy(d_fake, d_fake_logits))

            d_loss = d_real_loss + d_fake_loss

        # Computing gradient and backprop
        grad_d = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.cluster_optim.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))

        return {
            'disc_loss': d_loss
        }

    # @tf.function
    def train_on_batch(self, x, y, alternate):
        # First, train the discriminator
        ## Split the input into 2 halfs, one for real, one for fake
        half_batch = x.shape[0] // 2
        x_half_1 = x[:half_batch,:]
        x_half_2 = x[half_batch:, :]

        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as d_tape:
            d_fake = tf.zeros((x_half_2.shape[0], 1))
            d_real = tf.ones((x_half_1.shape[0], 1))

            d_real_logits = self.discriminator(x_half_1, training=True)

            # Get fake pred from discriminator and computing the losses
            latent_half_2 = self.encoder(x_half_2, training=True)
            x_hat_half_2 = self.decoder(latent_half_2, training=True)

            d_fake_logits = self.discriminator(x_hat_half_2, training=True)
            d_real_loss = tf.reduce_mean(tf.losses.binary_crossentropy(d_real, d_real_logits))
            d_fake_loss = tf.reduce_mean(tf.losses.binary_crossentropy(d_fake, d_fake_logits))

            d_loss = d_real_loss + d_fake_loss

            # Get latent space and cluster
            latent = self.encoder(x, training=True)
            x_hat = self.decoder(latent, training=True)

            cluster_pred = self.cluster(x, training=True)

            # Computing kl loss
            kl_loss = tf.reduce_mean(tf.losses.kl_divergence(y, cluster_pred))

            # Generator-similar loss for encoder
            total_enc_loss = kl_loss + d_fake_loss
            g_loss = d_fake_loss

            # Reconstruction loss
            res_loss = tf.reduce_mean(tf.losses.mse(x, x_hat))

        # Computing gradient and backprop
        if not alternate:
            grad_enc = enc_tape.gradient(total_enc_loss, self.cluster.trainable_variables)
            grad_d = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        grad_dec = dec_tape.gradient(res_loss, self.decoder.trainable_variables)

        if not alternate:
            self.cluster_optim.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))
            self.cluster_optim.apply_gradients(zip(grad_enc, self.cluster.trainable_variables))
        self.cluster_optim.apply_gradients(zip(grad_dec, self.decoder.trainable_variables))

        return {
            'kl_loss': kl_loss,
            'g_loss': g_loss,
            'dec_loss': res_loss,
            'disc_loss': d_loss
        }


def pretrain(model: ADEC, x_train, y_train, x_test,
        batch_size, epochs=1000, save_interval=200,
        save_path='./images',
        early_stopping=False):
    n_epochs = tqdm.tqdm_notebook(range(epochs))
    total_batches = x_train.shape[0] // batch_size
    EARLY_STOPPING_THRESHOLD = 1e-4
    PATIENCE = 20
    last_ae_loss = 10e10
    p_count = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in n_epochs:
        offset = 0
        losses = []
        random_idx = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        x_train_shuffle = x_train[random_idx,:]
        # y_train = y_train[random_idx]
        for batch_iter in range(total_batches):
            # Randomly choose each half batch
            imgs = x_train_shuffle[offset:offset + batch_size,:] if (batch_iter < (total_batches - 1)) else x_train[:batch_size,:]
            augs = []
            for img in imgs:
                aug_img = medium(image=img)['image']
                # aug_img = np.reshape(aug_img, (aug_img.shape[0]*aug_img.shape[1],))
                augs.append(aug_img)

            augs = np.array(augs)
            augs = tf.reshape(augs, (augs.shape[0], -1))
            offset += batch_size

            loss = pretrain_on_batch(augs, None, model)
            losses.append(loss)

        avg_loss = avg_losses(losses)
        wandb.log({'pretrain_losses': avg_loss})
            
        if epoch % save_interval == 0 or (epoch == epochs - 1):
            sampled_imgs = model.autoencoder.predict(np.reshape(x_test, (x_test.shape[0], -1))[:100])
            res_img = make_image_grid(sampled_imgs, (28,28), str(epoch), save_path)
            
            latent = model.encoder.predict(np.reshape(x_train, (x_train.shape[0], -1)))
            latent_space_img = visualize_latent_space(latent, y_train, 10, is_save=True, save_path=f'{save_path}/latent_{epoch}.png')
            wandb.log({'pretrain_res_test_img': [wandb.Image(res_img, caption="Reconstructed images")],
                        'pretrain_latent_space': [wandb.Image(latent_space_img, caption="Latent space")]
                    })
            
            if early_stopping:
                if last_ae_loss - avg_loss['res_ae_loss'] < EARLY_STOPPING_THRESHOLD:
                    p_count += 1
                    if p_count == PATIENCE:
                        print(f'No improvement after {PATIENCE} epochs. Stop!')
                        break # Stop training
        
def pretrain_phase2(model: ADEC, x_train, y_train, x_test,
        batch_size, epochs=1000, save_interval=200,
        save_path='./images',
        early_stopping=False):
    n_epochs = tqdm.tqdm_notebook(range(epochs))
    total_batches = x_train.shape[0] // batch_size
    EARLY_STOPPING_THRESHOLD = 1e-4
    PATIENCE = 20
    last_ae_loss = 10e10
    p_count = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs_for_disc = epochs // 3
    for epoch in n_epochs:
        offset = 0
        losses = []
        random_idx = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        x_train_shuffle = x_train[random_idx,:]
        # y_train = y_train[random_idx]
        for batch_iter in range(total_batches):
            # Randomly choose each half batch
            imgs = x_train_shuffle[offset:offset + batch_size,::] if (batch_iter < (total_batches - 1)) else x_train_shuffle[:batch_size,:]
            offset += batch_size

            loss = pretrain_on_batch_phase2(imgs, y_train, model)
            losses.append(loss)

        avg_loss = avg_losses(losses)
        wandb.log({'pretrain_phase2_losses': avg_loss})
            
        if epoch % save_interval == 0 or (epoch == epochs - 1):
            # Save the visualization of latent space, decoded input
            sampled_imgs = model.autoencoder.predict(x_test[:100])
            res_img = make_image_grid(sampled_imgs, (28,28), str(epoch), save_path)
            
            latent = model.encoder.predict(x_train)
            latent_space_img = visualize_latent_space(latent, y_train, 10, is_save=True, save_path=f'{save_path}/latent_{epoch}.png')

            wandb.log({'res_test_img': [wandb.Image(res_img, caption="Reconstructed images")],
                        'latent_space': [wandb.Image(latent_space_img, caption="Latent space")]
                    })

        if early_stopping:
            if last_ae_loss - avg_loss['res_ae_loss'] < EARLY_STOPPING_THRESHOLD:
                p_count += 1
                if p_count == PATIENCE:
                    print(f'No improvement after {PATIENCE} epochs. Stop!')
                    break # Stop training

def cluster(model: ADEC, x_train, y_train, x_test,
        batch_size, epochs=1000, save_interval=200,
        save_path='./images'):
    n_epochs = tqdm.tqdm_notebook(range(epochs))
    total_batches = x_train.shape[0] // batch_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Using kmeans result to initialize clusters for model
    latent = adec.encoder.predict(x_train)
    pca = PCA(n_components=latent.shape[1], whiten=True)
    whiten_latent = pca.fit_transform(latent)
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=100)
    init_cluster_pred = kmeans.fit_predict(whiten_latent)

    # Get initialize performance
    last_cluster_pred = np.copy(init_cluster_pred)
    init_f1 = acc(y_true=y_train, y_pred=init_cluster_pred)
    print('Initialized performance: ', init_f1)

    # Get initialize performance without whitening
    kmeans_2 = KMeans(n_clusters=model.n_clusters, n_init=100)
    init_cluster_pred_2 = kmeans_2.fit_predict(latent)
    init_f1_2 = acc(y_true=y_train, y_pred=init_cluster_pred_2)
    print('Initialized performance: ', init_f1_2)


    model.cluster.get_layer(name='clustering').set_weights([kmeans_2.cluster_centers_])
    # Check model cluster performance
    non_wh_cluster_res = model.cluster.predict(x_train).argmax(1)
    init_f1_1 = acc(y_true=y_train, y_pred=non_wh_cluster_res)
    print('Model cluster performance: ', init_f1_1)
    
    stop = False
    for epoch in n_epochs:
        offset = 0
        losses = []

        if epoch % save_interval == 0 or (epoch == epochs - 1):
            # Save the visualization of latent space, decoded input
            sampled_imgs = model.autoencoder.predict(x_test[:100])
            res_img = make_image_grid(sampled_imgs, (28,28), str(epoch), save_path)
            
            latent = model.encoder.predict(x_train)
            latent_space_img = visualize_latent_space(latent, y_train, 10, is_save=True, save_path=f'{save_path}/latent_{epoch}.png')

            # Log the clustering performance
            cluster_res = model.cluster.predict(x_train)
            y_pred = cluster_res.argmax(1)
            accuracy = acc(y_true=y_train, y_pred=y_pred)

            try:
                wandb.log({'res_test_img': [wandb.Image(res_img, caption="Reconstructed images")],
                            'latent_space': [wandb.Image(latent_space_img, caption="Latent space")],
                            'cluster_accuracy': accuracy
                        })
            except:
                print('cluster_accuracy: ', accuracy)
            
            delta_label = np.sum(y_pred != last_cluster_pred).astype(np.float32) / y_pred.shape[0]
            # if epoch > 0 and delta_label < model.tol:
            #     stop = False
            #     break

            last_cluster_pred = np.copy(cluster_res)

            # Update target distribution
            targ_dist = model.target_distribution(last_cluster_pred)

        is_alternate = False
        for batch_iter in range(total_batches):
            # Randomly choose each half batch
            imgs = x_train[offset:offset + batch_size,:] if (batch_iter < (total_batches - 1)) else x_train[:batch_size,:]
            y_cluster = targ_dist[offset:offset + batch_size,:] if (batch_iter < (total_batches - 1)) else targ_dist[:batch_size,:]
            offset += batch_size

            if batch_iter < int(2 * total_batches / 3):
                is_alternate = True
            else:
                is_alternate = False

            loss = train_on_batch(imgs, y_cluster, model, is_alternate)
            losses.append(loss)

        avg_loss = avg_losses(losses)
        try:
            wandb.log({'clustering_losses': avg_loss})
        except:
            pass
            
        # if stop:
        #     # Reach stop condition, stop training
        #     break