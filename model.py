import os
import time
import logging
import pickle
from core.utils import Shrink
import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import block_reduce


class HybridFeatures:
    def __init__(self, channel='Y'):
        self.dc_pca_kernels = dict()
        self.channel_pca_kernels = dict()
        self.relevant_dimension = dict()

        self.trained = False
        self.channel = channel

    def load(self, model_dir):
        self.trained = True
        self.dc_pca_kernels, self.channel_pca_kernels, self.relevant_dimension \
            = pickle.load(open(os.path.join(model_dir, "{}_extractor.pickle".format(self.channel)), "rb"))

    def save(self, model_dir):
        if not self.trained:
            logging.info('Model not trained, run fit_transform first...')
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump((self.dc_pca_kernels, self.channel_pca_kernels, self.relevant_dimension),
                    open(os.path.join(model_dir, "{}_extractor.pickle".format(self.channel)), "wb"))

    def fit_transform(self, dct_imgs, mos):
        # input: (n, 48, 48, 64)
        n = dct_imgs.shape[0]
        logging.info('# of training samples: {}'.format(n))

        spatial_features = []
        spectral_features = []

        ## DC
        dc_imgs = dct_imgs[:, :, :, :1]
        # non-overlapping 4x4 PCA: (nx12x12, 16)
        pca_h1, dc_h1 = self.block_pca(dc_imgs, 4)
        self.dc_pca_kernels['h1'] = pca_h1
        # Hop 1 spatial: 2D
        dc_h1_0 = dc_h1[:, 0].reshape(n, -1)
        spatial_features.append(np.mean(dc_h1_0, axis=1).reshape(n, -1))  # (n, 1)
        spatial_features.append(np.std(dc_h1_0, axis=1).reshape(n, -1))  # (n, 1)
        # Hop 1 spectral: non-overlapping 4x4 PCA -> (nx3x3, 16) -> (n, 9, 16)
        dc_h1_0 = dc_h1_0.reshape(n, 12, 12, 1)
        pca_h2, dc_h2 = self.block_pca(dc_h1_0, 4)
        self.dc_pca_kernels['h2'] = pca_h2
        spectral_features.append(dc_h2.reshape(n, -1, 16))  # (n, 9, 16)

        # Hop 2 channel 1-15: (n, 12, 12, 15)
        dc_h1_ac = dc_h1[:, 1:].reshape(n, 12, 12, -1)
        dc_h1_ac_spatial, dc_h1_ac_spectral = self.train_ac_features(dc_h1_ac, 3, 2, 2, component='DC')
        spatial_features.append(dc_h1_ac_spatial)  # (n, 90)
        spectral_features.append(dc_h1_ac_spectral)  # (n, 4, 60)

        ## AC
        ac_imgs = dct_imgs[:, :, :, 1:]  # (n, 48, 48, 63)
        ac_spatial, ac_spectral = self.train_ac_features(ac_imgs, 4, 4, 4, component='AC')
        spatial_features.append(ac_spatial)  # (n, 693)
        spectral_features.append(ac_spectral)  # (n, 9, 1008)

        spatial_features = np.concatenate(spatial_features, axis=-1)
        logging.info('Feature extractor trained...')

        logging.info('Start relevant feature test...')

        self.RFT(spatial_features, spectral_features, mos)
        hybrid_features = self.combine_features(spatial_features, spectral_features)

        logging.info('Finish relevant feature test...')

        self.trained = True

        return hybrid_features

    def transform(self, dct_imgs):
        if not self.trained:
            logging.info('Run fit_transform first...')
            return None
        logging.info('Start extracting features for channel {}...'.format(self.channel))
        # t = time.time()

        # input: (n, 48, 48, 64)
        spatial_features = []
        spectral_features = []

        # DC
        dc_imgs = dct_imgs[:, :, :, :1]
        n = dc_imgs.shape[0]
        # non-overlapping 4x4 PCA: (nx12x12, 16)
        pca_h1 = self.dc_pca_kernels['h1']
        dc_h1 = self.block_pca_test(dc_imgs, 4, pca_h1)
        # Hop 1 spatial: 2D
        dc_h1_0 = dc_h1[:, 0].reshape(n, -1)
        spatial_features.append(np.mean(dc_h1_0, axis=1).reshape(n, -1))  # (n, 1)
        spatial_features.append(np.std(dc_h1_0, axis=1).reshape(n, -1))  # (n, 1)
        # Hop 1 spectral: non-overlapping 4x4 PCA -> (nx3x3, 16) -> (n, 9, 16)
        dc_h1_0 = dc_h1_0.reshape(n, 12, 12, 1)
        pca_h2 = self.dc_pca_kernels['h2']
        dc_h2 = self.block_pca_test(dc_h1_0, 4, pca_h2)
        spectral_features.append(dc_h2.reshape(n, -1, 16))  # (n, 9, 16)

        # Hop 2 channel 1-15: (n, 12, 12, 15)
        dc_h1_ac = dc_h1[:, 1:].reshape(n, 12, 12, -1)
        dc_h1_ac_spatial, dc_h1_ac_spectral = self.extract_ac_features(dc_h1_ac, 3, 2, 2, component='DC')
        spatial_features.append(dc_h1_ac_spatial)  # (n, 90)
        spectral_features.append(dc_h1_ac_spectral)  # (n, 4, 60)

        # AC
        ac_imgs = dct_imgs[:, :, :, 1:]  # (n, 48, 48, 63)
        ac_spatial, ac_spectral = self.extract_ac_features(ac_imgs, 4, 4, 4, component='AC')
        spatial_features.append(ac_spatial)  # (n, 693)
        spectral_features.append(ac_spectral)  # (n, 9, 1008)

        spatial_features = np.concatenate(spatial_features, axis=-1)
        hybrid_features = self.combine_features(spatial_features, spectral_features)

        logging.info('Finish extracting features for channel {}...'.format(self.channel))
        # logging.info('Use {:.2f} seconds to extract {} features'.format(time.time() - t, n))

        return hybrid_features

    def train_ac_features(self, ac_imgs, win1, win2, win3, component='DC'):
        # abs. max pooling: (n, R, R, c) -> (n, r, r, c)
        ac_imgs = np.abs(ac_imgs)
        ac_imgs = block_reduce(ac_imgs, block_size=(1, win1, win1, 1), func=np.max)
        n = ac_imgs.shape[0]
        r = ac_imgs.shape[1]
        c = ac_imgs.shape[-1]
        ac_imgs = ac_imgs.reshape(n, -1, c)

        spatial_features = []
        for channel in range(c):
            # (n, r^2, c) -> [(n, d_1), (n, d_2), ...]
            f = ac_imgs[:, :, channel].reshape(n, r, r)
            # spatial features
            spatial_features.append(block_reduce(f, block_size=(1, win2, win2), func=np.max).reshape(n, -1))
            spatial_features.append(block_reduce(f, block_size=(1, r, r), func=np.mean).reshape(n, -1))
            spatial_features.append(block_reduce(f, block_size=(1, r, r), func=np.std).reshape(n, -1))

        spatial_features = np.concatenate(spatial_features, axis=-1)

        spectral_features = []
        self.channel_pca_kernels[component] = []
        for channel in range(c):
            f = ac_imgs[:, :, channel].reshape(n, r, r, 1)
            # spectral features
            pca, spectral_f = self.block_pca(f, win3)
            self.channel_pca_kernels[component].append(pca)
            spectral_features.append(spectral_f.reshape(n, -1, win3 * win3))

        spectral_features = np.concatenate(spectral_features, axis=-1)

        return spatial_features, spectral_features

    def extract_ac_features(self, ac_imgs, win1, win2, win3, component='DC'):
        # abs. max pooling: (n, R, R, c) -> (n, r, r, c)
        ac_imgs = np.abs(ac_imgs)
        ac_imgs = block_reduce(ac_imgs, block_size=(1, win1, win1, 1), func=np.max)
        n = ac_imgs.shape[0]
        r = ac_imgs.shape[1]
        c = ac_imgs.shape[-1]
        ac_imgs = ac_imgs.reshape(n, -1, c)

        spatial_features = []
        for channel in range(c):
            # (n, r^2, c) -> [(n, d_1), (n, d_2), ...]
            f = ac_imgs[:, :, channel].reshape(n, r, r)
            # spatial features
            spatial_features.append(block_reduce(f, block_size=(1, win2, win2), func=np.max).reshape(n, -1))
            spatial_features.append(block_reduce(f, block_size=(1, r, r), func=np.mean).reshape(n, -1))
            spatial_features.append(block_reduce(f, block_size=(1, r, r), func=np.std).reshape(n, -1))

        spatial_features = np.concatenate(spatial_features, axis=-1)

        spectral_features = []
        for channel in range(c):
            f = ac_imgs[:, :, channel].reshape(n, r, r, 1)
            # spectral features
            pca = self.channel_pca_kernels[component][channel]
            spectral_f = self.block_pca_test(f, win3, pca)
            spectral_features.append(spectral_f.reshape(n, -1, win3 * win3))

        spectral_features = np.concatenate(spectral_features, axis=-1)

        return spatial_features, spectral_features

    @staticmethod
    def block_pca(X, win, n_dim=0):
        X_tmp = Shrink(X, win).reshape(-1, win * win)
        if n_dim == 0:
            pca_tmp = PCA(n_components=win * win)
        else:
            pca_tmp = PCA(n_components=n_dim)
        return pca_tmp, pca_tmp.fit_transform(X_tmp)

    @staticmethod
    def block_pca_test(X, win, pca):
        X_tmp = Shrink(X, win).reshape(-1, win * win)
        return pca.transform(X_tmp)

    def RFT(self, spatial, spectral, mos):
        if self.channel == 'Y':
            spatial_dimension = 200
            spectral_dimension = [5, 20, 200]
        else:
            spatial_dimension = 50
            spectral_dimension = [2, 5, 100]

        # spatial
        record_mse_spatial = dict()

        for d in range(spatial.shape[1]):
            spatial_1d = spatial[:, d]
            best_mse, best_partition_index = self.find_best_partition(spatial_1d, mos, bins=32)
            record_mse_spatial[d] = best_mse

        sorted_mse_spatial = {k: v for k, v in sorted(record_mse_spatial.items(), key=lambda item: item[1])}

        mse_list_spatial = list(record_mse_spatial.values())
        mse_list_spatial.sort()

        self.relevant_dimension['spatial'] = np.array(list(sorted_mse_spatial.keys()))[np.arange(spatial_dimension)]

        # spectral
        for i in range(len(spectral)):
            record_mse_spectral = dict()

            for c in range(spectral[i].shape[-1]):
                Y_spectral_ho = spectral[i][:, :, c]
                best_mse, best_partition_index = self.find_best_partition_ho(Y_spectral_ho, mos)
                record_mse_spectral[c] = best_mse

            sorted_mse_spectral = {k: v for k, v in sorted(record_mse_spectral.items(), key=lambda item: item[1])}

            mse_list_spectral = list(record_mse_spectral.values())
            mse_list_spectral.sort()

            self.relevant_dimension['spectral_{}'.format(i)] = \
                np.array(list(sorted_mse_spectral.keys()))[np.arange(spectral_dimension[i])]

    def find_best_partition(self, f_1d, mos, bins=32, metric='mse'):
        f_1d, mos = self.remove_outliers(f_1d, mos)
        best_error = float('inf')
        best_partition_index = 0
        f_min, f_max = f_1d.min(), f_1d.max()
        bin_width = (f_max - f_min) / bins
        for i in range(1, bins):
            partition_point = f_min + i * bin_width
            left_mos, right_mos = mos[f_1d <= partition_point], mos[f_1d > partition_point]
            partition_error = self.get_partition_error(left_mos, right_mos, metric)
            if partition_error < best_error:
                best_error = partition_error
                best_partition_index = i
        return best_error, best_partition_index

    @staticmethod
    def get_partition_error(left_mos, right_mos, metric='mse'):
        if metric == 'mse':
            n1, n2 = len(left_mos), len(right_mos)
            left_mse = ((left_mos - left_mos.mean()) ** 2).sum()
            right_mse = ((right_mos - right_mos.mean()) ** 2).sum()
            return np.sqrt((left_mse + right_mse) / (n1 + n2))
        else:
            print('Unsupported error')
            return 0

    @staticmethod
    def remove_outliers(f_1d, mos, n_std=2.0):
        # f is a 1D feature
        new_f_1d = []
        new_mos = []
        f_mean, f_std = f_1d.mean(), f_1d.std()
        for i in range(len(mos)):
            if np.abs(f_1d[i] - f_mean) <= n_std * f_std:
                new_f_1d.append(f_1d[i])
                new_mos.append(mos[i])
        return np.array(new_f_1d), np.array(new_mos)

    def find_best_partition_ho(self, f_ho, mos, bins=32, metric='mse'):
        pca = PCA(n_components=1)
        f_1d = pca.fit_transform(f_ho).reshape(-1)
        return self.find_best_partition(f_1d, mos, bins=bins, metric=metric)

    def combine_features(self, spatial, spectral):
        hybrid_features = []
        n = spatial.shape[0]
        hybrid_features.append(spatial[:, self.relevant_dimension['spatial']])
        for i in range(len(spectral)):
            hybrid_features.append(spectral[i][:, :, self.relevant_dimension['spectral_{}'.format(i)]].reshape(n, -1))
        hybrid_features = np.concatenate(hybrid_features, axis=-1)
        return hybrid_features
