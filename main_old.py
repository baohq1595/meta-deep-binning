import glob, os, time
import json
import argparse

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling

from metadec.dataset.genome import SimGenomeDataset
from metadec.model.dec import DEC
from metadec.model.idec import IDEC
from metadec.utils.utils import load_genomics
from metadec.debug.visualize import store_results
from metadec.utils.metrics import genome_acc

import sys
sys.path.append('.')


RESULT_DIR = 'results'
GEN_DATA_DIR = 'data/simulated'

# Versioning each runs
ARCH = 'dec_genomics'
DATE = '20200621'

# Training batchsize
BATCH_SIZE = 1
# Maximum iterations for clustering optimization step
MAX_ITERS = 10
# Number of epochs for pretraining
PRETRAIN_EPOCHS = 5
# Interval for updating the training status
UPDATE_INTERVAL = 1
# Tolerance threshold to stop clustering optimization step
TOL = 0.000001
# Trained weight for pretrained autoencoder
AE_WEIGHTS = None
# Dir contains raw fasta data
DATASET_DIR = GEN_DATA_DIR
# Specifc dataset or all of them
DATASET_NAME = 'S1'

# MODEL_DIR = f'/content/drive/My Drive/DL/{ARCH}/{DATE}/models'
# LOG_DIR = f'/content/drive/My Drive/DL/{ARCH}/{DATE}/logs/'
# GEN_DATA_DIR = f'/content/drive/My Drive/DL/data/gene/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory contain fasta file')
parser.add_argument('--dataset_name', type=str, help='specific name of dataset to run'\
                    '(all is run every found dataset), e.g S1.fna, name is S1')
parser.add_argument('--result_dir', type=str, default='results', help='directory for saving the resuls')
parser.add_argument('--verbose', help='Whether to print log in terminal', action='store_true')

args = parser.parse_args()

verbose = args.verbose

DATASET_NAME = args.dataset_name if args.dataset_name else DATASET_NAME
GEN_DATA_DIR = args.data_dir if args.data_dir else GEN_DATA_DIR
RESULT_DIR = args.result_dir if args.result_dir else RESULT_DIR

LOG_DIR = f'{RESULT_DIR}\log'
MODEL_DIR = f'{RESULT_DIR}\model'


for d in [LOG_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Dir for saving training results
SAVE_DIR = os.path.join(MODEL_DIR, ARCH, DATE)

# Dir for saving log results: visualization, training logs
LOG_DIR = os.path.join(LOG_DIR, ARCH, DATE)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Hyperparameters
# Follows metaprob
KMERS = [4]
LMER = 30
NUM_SHARED_READS = (5, 45)
ONLY_SEED = True
MAXIMUM_SEED_SIZE = 5000

raw_dir = os.path.join(DATASET_DIR, 'raw')
processed_dir = os.path.join(DATASET_DIR, 'processed')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
if DATASET_NAME == 'all':
    raw_datasets = glob.glob(raw_dir + '/*.fna')
else:
    raw_datasets = [os.path.join(raw_dir, DATASET_NAME + '.fna')]

# Mapping of dataset and its corresponding number of clusters
# with open('config/dataset_metadata.json', 'r') as f:
#     n_clusters_mapping = json.load(f)['simulated']

for dataset in raw_datasets:
    # Get some parameters
    dataset_name = os.path.basename(dataset).split('.fna')[0]
    save_dir = os.path.join(SAVE_DIR, dataset_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_shared_read = NUM_SHARED_READS[1] if 'R' in dataset_name else NUM_SHARED_READS[0]
    is_deserialize = os.path.exists(os.path.join(processed_dir, dataset_name + '.json'))
    # n_clusters = n_clusters_mapping[dataset_name]
    n_clusters = 5
    num_shared_read = 5
    
    if verbose:
        # Read dataset
        print(f'Processing dataset {dataset_name}...')
        print(f'Prior number of clusters: {n_clusters}...')
        print(f'Prior number of shared reads: {num_shared_read}...')

    # try:
    #     genome_dataset = load_genomics(
    #         dataset,
    #         kmers=KMERS,
    #         lmer=LMER,
    #         maximum_seed_size=MAXIMUM_SEED_SIZE,
    #         num_shared_reads=num_shared_read,
    #         is_deserialize=is_deserialize,
    #         is_serialize=~is_deserialize,
    #         is_normalize=True,
    #         only_seed=ONLY_SEED,
    #         graph_file=os.path.join(processed_dir, dataset_name + '.json'),
    #         is_amd=True
    #     )
    # except:
    genome_dataset = load_genomics(
        dataset,
        kmers=KMERS,
        lmer=LMER,
        maximum_seed_size=MAXIMUM_SEED_SIZE,
        num_shared_reads=num_shared_read,
        is_deserialize=False,
        is_serialize=False,
        is_normalize=True,
        only_seed=ONLY_SEED,
        graph_file=os.path.join(processed_dir, dataset_name + '.json'),
        is_amd=False
    )

    seed_kmer_features, labels, groups, seeds, label2idx = genome_dataset.kmer_features, genome_dataset.labels,\
                                                            genome_dataset.groups, genome_dataset.seeds,\
                                                            genome_dataset.label2idx

    exit(1)
    
    # continue
    # Initialize model
    init_lr = 0.1
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    pretrain_optimizer = SGD(lr=init_lr, momentum=0.9, decay=init_lr/PRETRAIN_EPOCHS)

    dec = IDEC(dims=[seed_kmer_features.shape[-1], 500, 1000, 10], n_clusters=n_clusters, init=init)
    
    # Start clustering
    if AE_WEIGHTS is None:
        dec.pretrain(x=seed_kmer_features, y=labels, grps=groups, optimizer=pretrain_optimizer,
                epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
                save_dir=save_dir)
    else:
        dec.autoencoder.load_weights(AE_WEIGHTS)

    dec.model.summary()
    t0 = time.time()
    optim_lr = 0.01
    dec.compile(optimizer=SGD(optim_lr))
    y_pred = dec.fit(x=seed_kmer_features, y=labels, grps=groups, tol=TOL, maxiter=MAX_ITERS, batch_size=BATCH_SIZE,
                      update_interval=UPDATE_INTERVAL, save_dir=save_dir)
    
    if verbose:
        print('...')
    latent = dec.encoder.predict(seed_kmer_features)
    y_pred = dec.predict(seed_kmer_features)

    if verbose:
        print('Saving results...')
    store_results(groups, seed_kmer_features, latent, labels, y_pred,
                  n_clusters, dataset_name, save_dir=os.path.join(LOG_DIR, dataset_name))
    if verbose:
        print(f'Finish clustering for dataset {dataset_name}.')
        print('F1-score:', genome_acc(groups, y_pred, labels, n_clusters)[2])
        print('Clustering time: ', (time.time() - t0))