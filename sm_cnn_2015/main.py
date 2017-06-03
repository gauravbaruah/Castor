import argparse

import os
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import utils
from external_features import stopped, stemmed, compute_idf_weighted_overlap, compute_overlap,\
    get_qadata_only_idf, set_external_features_as_per_paper,\
    set_external_features_as_per_paper_and_stem
from train import Trainer
from model import QAModel

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def logargs(func):
    def inner(*args, **kwargs):
        logger.info('%s : %s %s' % (func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return inner


def compute_map_mrr(dataset_folder, set_folder, test_scores, run_name_prefix=None):
    # logger.info("Running trec_eval script...")
    N = len(test_scores)

    qids_test, y_test = utils.get_test_qids_labels(dataset_folder, set_folder)

    # Call TrecEval code to calc MAP and MRR
    df_submission = pd.DataFrame(index=np.arange(N), \
        columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = np.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = test_scores
    df_submission['run_id'] = 'smmodel'
    df_submission.to_csv(os.path.join(dataset_folder, 'submission.txt'), \
        header=False, index=False, sep=' ')
    if run_name_prefix:
        df_submission.to_csv('{}.{}.smrun'.format(run_name_prefix, set_folder),\
                             header=False, index=False, sep=" ")

    df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = np.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(os.path.join(args.dataset_folder, 'gold.txt'), header=False, index=False, sep=' ')

    # subprocess.call("/bin/sh run_eval.sh '{}'".format(args.dataset_folder), shell=True)
    pargs = shlex.split("/bin/sh run_eval.sh '{}'".format(dataset_folder))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    lines = pout.split(b'\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    return map, mrr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='pytorch port of the SM model', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('model-outfile', \
        help='file to save final model')
    ap.add_argument('--word-vectors-file', \
        help='NOTE: a cache will be created for faster loading for word vectors',\
        default="../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.bin")
    ap.add_argument('--dataset-folder', \
        help='directory containing train, dev, test sets', \
        default="../../data/TrecQA")

    # external features related arguments
    ap.add_argument('--paper-ext-feats', action="store_true", \
        help="external features as per the paper", \
        default=True)
    # system arguments
    # TODO: add arguments for CUDA
    ap.add_argument('--num-threads', type=int, \
        help="the number of simultaneous processes to run", \
        default=4)
    # training arguments
    ap.add_argument('--batch-size', type=int, \
        default=1, \
        help="training mini-batch size")
    ap.add_argument('--filter-width', type=int, \
        default=5, \
        help="number of convolution channels")
    ap.add_argument('--eta', \
        help='Initial learning rate', \
        default=0.001, type=float)
    ap.add_argument('--mom', \
        help='SGD Momentum', \
        default=0.0, type=float)
    ap.add_argument('--train', help='switches to train set', action="store_true")
    # epoch related arguments
    ap.add_argument('--epochs', \
        help="number of training epochs", \
        default=25, type=int)
    ap.add_argument('--patience', type=int, \
        default=5, \
        help="if there is no appreciable change in model after <patience> epochs, then stop")
    # debugging arguments
    ap.add_argument('--num-conv-filters', \
        default=100, type=int, \
        help="the number of convolution channels (lesser is faster)")
    ap.add_argument('--test-on-each-epoch', action="store_true", \
        help='runs test on each epoch to track final performance')
    ap.add_argument("--skip-training", action="store_true", \
        help="will load pre-trained model")
    ap.add_argument("--run-name-prefix", \
        help="will output train|dev|test runs with provided prefix")
    ap.add_argument("--index-for-corpusIDF", \
        help="fetches idf from Index. provide index path. will generate a vocabFile")

    args = ap.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.set_num_threads(args.num_threads)

    train_set, dev_set, test_set = 'train-all', 'raw-dev', 'raw-test'

    # initialize the trainer
    trainer = Trainer(args.dataset_folder, train_set, dev_set, test_set, # input data
                      args.word_vectors_file,                            # word embeddings
                      args.eta, args.mom,                                # optimization params
                      args.filter_width, args.num_conv_filters           # model params
                     )

    # load input data    
    logger.info("Setting up external features...")
    # setup external features
    # TODO: remember to update args.* in testing loop below
    if args.paper_ext_feats:
        logger.info("--paper-ext-feats")
        ext_feats_for_splits = \
            set_external_features_as_per_paper(trainer, args.index_for_corpusIDF)
        # ^^ we are saving the features to be used while testing at the end of training

    if not args.skip_training:
        best_map = 0.0
        best_model = 0

        for i in range(args.epochs):
            logger.info('------------- Training epoch {} --------------'.format(i+1))
            train_accuracy = trainer.train(train_set, args.batch_size)
            sys.exit()
            dev_scores = trainer.test(dev_set, args.batch_size)
            
            dev_map, dev_mrr = compute_map_mrr(args.dataset_folder, dev_set, dev_scores)
            logger.info("------- MAP {}, MRR {}".format(dev_map, dev_mrr))

            if dev_map - best_map > 1e-3: # new map is better than best map
                best_model = i
                best_map = dev_map

                QAModel.save(net, args.model_outfile)
                logger.info('Achieved better dev_map ... saved model')

            if args.test_on_each_epoch:
                test_scores = trainer.test(test_set, args.batch_size)
                map, mrr = compute_map_mrr(args.dataset_folder, test_set, test_scores)
                logger.info("------- MAP {}, MRR {}".format(map, mrr))

            if (i - best_model) >= args.patience:
                logger.warning('No improvement since the last {} epochs. Stopping training'\
                               .format(i - best_model))
                break

        logger.info(' ------------ Training epochs completed! ------------')
        logger.info('Best dev MAP in training phase = {:.4f}'.format(best_map))
    
    evaluator = Trainer(args.eta, args.mom,
                      args.dataset_folder, train_set, dev_set, test_set,
                      args.filter_width, args.num_conv_filters, args.model_outfile)

    for split in [test_set, dev_set]:
        evaluator.load_input_data(args.dataset_folder, cache_file, None, None, split)
        if args.paper_ext_feats or args.paper_ext_feats_stem:
            evaluator.data_splits[split][-1] = ext_feats_for_splits[split]
            #set_external_features_as_per_paper(evaluator)
        split_scores = evaluator.test(split, args.batch_size)
        map, mrr = compute_map_mrr(args.dataset_folder, split, split_scores, args.run_name_prefix)
        logger.info("-------{} MAP {}, MRR {}".format(split, map, mrr))
