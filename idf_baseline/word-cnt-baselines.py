# implementation of Word Cnt and Word Cnt Weighted baselines as described in 
# Yih, W. T., Chang, M. W., Meek, C., Pastusiak, A., Yih, S. W. T., & Meek, C. (2013).
# "Question answering using enhanced lexical semantic models.", ACL
# (cf. Section 6.2)

import argparse
import os
import shlex
import subprocess
import re
import string
from collections import defaultdict
from collections import Counter

import numpy as np

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def read_in_data(datapath, set_name, file):
    data = []
    with open(os.path.join(datapath, set_name, file)) as inf:
        data = [line.strip() for line in inf.readlines()]
    return data

def read_data_and_preprocess(datapath, set_name, file):
    data = read_in_data(datapath, set_name, file)

    # remove punctuation
    regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    def remove_punctuation(sentence):
        rtokens = []
        for term in sentence.split():
            for t in regex.sub(' ', term).strip().split():
                if t:
                    rtokens.append(t)
        return ' '.join(rtokens)
    data = [remove_punctuation(sentence) for sentence in data]

    # remove stopwords
    stoplist = set(stopwords.words('english'))
    def remove_stopwords(sentence):
        return ' '.join([word for word in sentence.split() if word not in stoplist])
    data = [remove_stopwords(sentence) for sentence in data]

    return data

def read_in_wikiQA_data_file(filepath):
    dataset = []
    qid_count = 0
    qid_old = None
    with open(filepath) as inf:
        inf.readline() # header
        for line in inf:
            fields = line.lower().strip().split('\t')
            qid = fields[0]
            question = fields[1]
            sentence = fields[5]
            label = fields[6]
            if qid != qid_old:
                qid_old = qid
                qid_count += 1
            dataset.append((qid_count, question, sentence, label))
    return [z for z in zip(*dataset)]

def compute_word_count_with_raw_data(args):
    if 'WikiQA' in args.qa_data:
        train_set = \
            read_in_wikiQA_data_file(os.path.join(args.qa_data, 'WikiQACorpus', 'WikiQA-train.tsv'))
        dev_set = \
            read_in_wikiQA_data_file(os.path.join(args.qa_data, 'WikiQACorpus', 'WikiQA-dev.tsv'))
        test_set = \
            read_in_wikiQA_data_file(os.path.join(args.qa_data, 'WikiQACorpus', 'WikiQA-test.tsv'))

        all_data = train_set[1] + train_set[2] + dev_set[1] + dev_set[2] + test_set[1] + test_set[2]

        term_idfs = None
        if args.weighted:
            if not args.index_for_corpusIDF:
                term_idfs = compute_idfs(set(all_data))
            else:
                term_idfs = fetch_idfs_from_index(set(all_data), args.index_for_corpusIDF)

        scores = compute_word_cnt(dev_set[1], dev_set[2], term_idfs)
        write_out_word_cnt_run(dev_set[0],
                            scores,
                            args.qa_data,
                            '{}.{}.{}WordCnt.run'.format(\
                            args.outfile_prefix, "dev", "Weighted" if args.weighted else ""))

        scores = compute_word_cnt(test_set[1], test_set[2], term_idfs)
        write_out_word_cnt_run(test_set[0],
                            scores,
                            args.qa_data,
                            '{}.{}.{}WordCnt.run'.format(\
                            args.outfile_prefix, "test", "Weighted" if args.weighted else ""))

 

def compute_idfs(data):
    term_idfs = defaultdict(float)
    for doc in list(data):
        for term in list(set(doc.split())):
            term_idfs[term] += 1.0
    N = len(data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/(1+n_t))
    return term_idfs


def fetch_idfs_from_index(data, indexPath):
    term_idfs = defaultdict(float)
    all_terms = set([term for doc in list(data) for term in doc.split()])
    with open('dataset.vocab', 'w') as vf:
        for term in list(all_terms):
            print(term, file=vf)

    fetchIDF_cmd = \
        "sh ../idf_baseline/target/appassembler/bin/FetchTermIDF -index {} -vocabFile {}".\
            format(indexPath, 'dataset.vocab')
    pargs = shlex.split(fetchIDF_cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                             bufsize=1, universal_newlines=True)
    pout, perr = p.communicate()

    lines = str(pout).split('\n')
    for line in lines:
        if not line:
            continue
        fields = line.strip().split("\t")
        term, weight = fields[0], fields[-1]
        term_idfs[term] = float(weight)

    for line in str(perr).split('\n'):
        print('Warning: '+line)
    return term_idfs


def compute_word_cnt(questions, answers, term_idfs=None):
    scores = np.zeros(len(questions))
    for i in range(len(questions)):
        # Approach 1
        q_terms = set(questions[i].split())
        a_terms = set(answers[i].split())
        common_terms = q_terms.intersection(a_terms)
        scores[i] = len(common_terms)
        if term_idfs:
            scores[i] = np.sum([term_idfs[term] for term in common_terms])
        
        # Approach 2 and 3
        # q_counts = Counter(questions[i].split())
        # a_counts = Counter(answers[i].split())
        # common_terms = defaultdict(float)
        # for qterm in q_counts:
        #     if qterm in a_counts:
        #         # Approach 2
        #         # common_terms[qterm] = min(q_counts[qterm], a_counts[qterm])

        #         # Approach 3
        #         # common_terms[qterm] = a_counts[qterm]
        # scores[i] = np.sum([count for term, count in common_terms.items()])
        # if term_idfs:
        #     scores[i] = np.sum([count*term_idfs[term] for term, count in common_terms.items()])
            
    return scores


def write_out_word_cnt_run(qids, scores, dataset, outfile):

    with open(outfile, 'w') as outf:
        old_qid = 0
        docid_c = 0
        for i in range(len(scores)):
            if qids[i] != old_qid and dataset.endswith('WikiQA'):
                docid_c = 0
                old_qid = qids[i]
            print('{} 0 {} 0 {} wordCnt'.format(\
                qids[i], docid_c, scores[i]), file=outf)
            docid_c += 1


def main(args):
    if args.use_raw_data:
        return compute_word_count_with_raw_data(args)


    # read in the data
    train_data, dev_data, test_data = 'train', 'dev', 'test'
    if args.qa_data.endswith('TrecQA'):
        train_data, dev_data, test_data = 'train-all', 'raw-dev', 'raw-test'

    train_que = read_data_and_preprocess(args.qa_data, train_data, 'a.toks')
    train_ans = read_data_and_preprocess(args.qa_data, train_data, 'b.toks')

    dev_que = read_data_and_preprocess(args.qa_data, dev_data, 'a.toks')
    dev_ans = read_data_and_preprocess(args.qa_data, dev_data, 'b.toks')

    test_que = read_data_and_preprocess(args.qa_data, test_data, 'a.toks')
    test_ans = read_data_and_preprocess(args.qa_data, test_data, 'b.toks')

    all_data = train_que + dev_que + train_ans + dev_ans
    all_data += test_ans
    all_data += test_que

    term_idfs = None
    if args.weighted:
        if not args.index_for_corpusIDF:
            term_idfs = compute_idfs(set(all_data))
        else:
            term_idfs = fetch_idfs_from_index(set(all_data), args.index_for_corpusIDF)

    scores = compute_word_cnt(dev_que, dev_ans, term_idfs)
    write_out_word_cnt_run(read_in_data(args.qa_data, dev_data, 'id.txt'),
                           scores,
                           args.qa_data,
                           '{}.{}.{}WordCnt.run'.format(\
                           args.outfile_prefix, dev_data, "Weighted" if args.weighted else ""))

    scores = compute_word_cnt(test_que, test_ans, term_idfs)
    write_out_word_cnt_run(read_in_data(args.qa_data, test_data, 'id.txt'),
                           scores,
                           args.qa_data,
                           '{}.{}.{}WordCnt.run'.format(\
                           args.outfile_prefix, test_data, "Weighted" if args.weighted else ""))



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="word cnt and weighted word cnt baselines")
    ap.add_argument("qa_data", help="path to the QA dataset",\
        choices=['../../data/TrecQA', '../../data/WikiQA'])
    ap.add_argument("outfile_prefix", help="output file prefix")
    ap.add_argument("--weighted", help="computed weighted word count",\
        action="store_true")
    ap.add_argument("--index-for-corpusIDF",\
        help="fetches idf from Index. Provide index path with argument. Will generate a vocabFile")
    ap.add_argument("--use-raw-data", help="use the data in the form it was distributed",
        action="store_true")

    args = ap.parse_args()

    main(args)
