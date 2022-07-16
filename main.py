import warnings
import pickle
import argparse
import os
import time
import logging

warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from scipy import stats
from model import HybridFeatures
from core.utils.biqa import load_data, augment, jpeg_dct
import matplotlib.pyplot as plt


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and testing Green-BIQA',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--yuv', action='store_true')
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--num_aug', type=int, default=4)

    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.output_dir, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main(args):
    if args.yuv:
        assert (args.height != 0) and (args.width != 0), "Please specify height and width when using yuv."

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_logger(args)

    logging.info('Loading images from {}...'.format(args.data_dir))
    t = time.time()
    images, mos = load_data(args)
    logging.info('{} images loaded in {} secs...'.format(len(images), time.time() - t))

    plt.figure()
    plt.imsave('test_u.png', images[0][:, :, 1], cmap='gray')

    logging.info('Augmenting data: {} times...'.format(args.num_aug))
    t = time.time()
    images, aug_mos = augment(images, mos, args.num_aug)
    logging.info('Data augmented in {} secs...'.format(time.time() - t))

    logging.info('Extracting DCT coefficients...')
    t = time.time()
    train_Y_images_block_dct = jpeg_dct(images, 'Y')
    train_U_images_block_dct = jpeg_dct(images, 'U')
    train_V_images_block_dct = jpeg_dct(images, 'V')
    logging.info('DCT coefficients extracted in {} secs...'.format(time.time() - t))

    Y_feature_extractor = HybridFeatures(channel='Y')
    U_feature_extractor = HybridFeatures(channel='U')
    V_feature_extractor = HybridFeatures(channel='V')

    if args.do_train:
        logging.info('Start training the feature extractor...')
        t = time.time()
        y_features = Y_feature_extractor.fit_transform(train_Y_images_block_dct, aug_mos)
        Y_feature_extractor.save(args.model_dir)
        u_features = U_feature_extractor.fit_transform(train_U_images_block_dct, aug_mos)
        U_feature_extractor.save(args.model_dir)
        v_features = V_feature_extractor.fit_transform(train_V_images_block_dct, aug_mos)
        V_feature_extractor.save(args.model_dir)
        logging.info('Feature extractor trained in {} secs...'.format(time.time() - t))

        features = np.concatenate([y_features, u_features, v_features], axis=-1)
        logging.info('Feature Size = {}'.format(features.shape))

        if args.save:
            logging.info('Saving extracted features in {}'.format(args.output_dir))
            pickle.dump(features, open(os.path.join(args.output_dir, "features.pickle"), "wb"))

        n_train = int(0.9 * len(mos)) * args.num_aug

        all_index = np.arange(features.shape[0])
        train_index = all_index[:n_train]
        valid_index = all_index[n_train:]

        X_train, X_valid = features[train_index], features[valid_index]
        y_train, y_valid = aug_mos[train_index], aug_mos[valid_index]

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        reg = xgb.XGBRegressor(objective='reg:squarederror',
                               max_depth=5,
                               n_estimators=1500,
                               subsample=0.6,
                               eta=0.08,
                               colsample_bytree=0.4,
                               min_child_weight=4)

        logging.info('Start training the regressor...')
        t = time.time()
        reg.fit(X_train, y_train, eval_set=eval_set, eval_metric=['rmse'],
                early_stopping_rounds=100, verbose=False)
        logging.info('Regressor trained in {} secs...'.format(time.time() - t))

        bst = reg.get_booster()
        bst.save_model(os.path.join(args.model_dir, 'xgboost.json'))
        
        logging.info('Validating...')
        pred_valid_mos = reg.predict(X_valid)

        SRCC = stats.spearmanr(pred_valid_mos, y_valid)
        logging.info("SRCC: {}".format(SRCC[0]))

        corr, _ = pearsonr(pred_valid_mos, y_valid)
        logging.info("PLCC: {}".format(corr))

    if args.do_test:
        logging.info('Loading pre-trained feature extractors...')
        Y_feature_extractor.load(args.model_dir)
        U_feature_extractor.load(args.model_dir)
        V_feature_extractor.load(args.model_dir)

        logging.info('Extracting features for {} patches...'.format(train_V_images_block_dct.shape[0]))
        t = time.time()
        y_features = Y_feature_extractor.transform(train_Y_images_block_dct)
        u_features = U_feature_extractor.transform(train_U_images_block_dct)
        v_features = V_feature_extractor.transform(train_V_images_block_dct)
        logging.info('Features for {} images are extracted in {} secs...'
                     .format(train_V_images_block_dct.shape[0], time.time() - t))

        features = np.concatenate([y_features, u_features, v_features], axis=-1)

        if args.save:
            logging.info('Saving extracted features in {}'.format(args.output_dir))
            pickle.dump(features, open(os.path.join(args.output_dir, "features.pickle"), "wb"))

        reg = xgb.XGBRegressor(objective='reg:squarederror',
                               max_depth=5,
                               n_estimators=1500,
                               subsample=0.6,
                               eta=0.08,
                               colsample_bytree=0.4,
                               min_child_weight=4)

        reg.load_model(os.path.join(args.model_dir, 'xgboost.json'))

        logging.info('Testing...')

        pred_mos = []
        for start in range(0, features.shape[0], args.num_aug):
            test_features = features[start:start + args.num_aug]
            pred_test_mos = reg.predict(test_features)
            pred_mos.append(np.mean(pred_test_mos))

        pred_mos = np.array(pred_mos)
        SRCC = stats.spearmanr(pred_mos, mos)
        logging.info("SRCC: {}".format(SRCC[0]))

        corr, _ = pearsonr(pred_mos, mos)
        logging.info("PLCC: {}".format(corr))


if __name__ == '__main__':
    main(parse_args())
