# file input output

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

def word2vec_load_bin_vec(word_embeddings_file, words):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("Loading word embeddings ", word_embeddings_file)
    vocab = set(words)
    word_vecs = {}
    with open(word_embeddings_file, "rb") as f:
        header = f.readline()
        vocab_size, vec_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * vec_dim
        print('vocab_size, vec_dim', vocab_size, vec_dim)
        count = 0
        for line in tqdm(range(vocab_size)):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = str(b''.join(word), errors='ignore').encode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            if word.decode('utf-8') in vocab:
                count += 1
                word_vecs[word.decode('utf-8')] = np.fromstring(f.read(binary_len), dtype='float32')                
            else:
                f.read(binary_len)
        print("done")
        print("Words found in wor2vec embeddings", count)
        return word_vecs, count, vec_dim


def read_in_data(datapath, set_name, file, stop_and_stem=False, stop_punct=False, dash_split=False):
    data = []
    with open(os.path.join(datapath, set_name, file)) as inf:
        data = [line.strip() for line in inf.readlines()]

        if dash_split:
            def split_hyphenated_words(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in term.split('-'):
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            data = [split_hyphenated_words(sentence) for sentence in data]

        if stop_punct:
            regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
            def remove_punctuation(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in regex.sub(' ', term).strip().split():
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            data = [remove_punctuation(sentence) for sentence in data]

        if stop_and_stem:
            stemmer = PorterStemmer()
            stoplist = set(stopwords.words('english'))
            def stop_stem(sentence):
                return ' '.join([stemmer.stem(word) for word in sentence.split() \
                                                        if word not in stoplist])
            data = [stop_stem(sentence) for sentence in data]
    return data

def read_in_dataset(dataset_folder, set_folder, stop_punct=False, dash_split=False):
    """
    read in the data to return (question, sentence, label)
    set_folder = {train|dev|test}
    """
    max_q = 0
    max_s = 0
    set_path = os.path.join(dataset_folder, set_folder)

    #questions = [line.strip() for line in open(os.path.join(set_path, 'a.toks')).readlines()]
    questions = read_in_data(dataset_folder, set_folder, "a.toks", False, stop_punct, dash_split)
    len_q_list = [len(q.split()) for q in questions]

    #sentences = [line.strip() for line in open(os.path.join(set_path, 'b.toks')).readlines()]
    sentences = read_in_data(dataset_folder, set_folder, "b.toks", False, stop_punct, dash_split)
    len_s_list = [len(s.split()) for s in sentences]

    #labels = [int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt')).readlines()]
    labels = [int(lbl) for lbl in read_in_data(dataset_folder, set_folder, "sim.txt")]

    #vocab = [line.strip() for line in open(os.path.join(dataset_folder, 'vocab.txt')).readlines()]
    #all_data = list(set(questions)) + list(set(sentences))
    all_data = questions + sentences
    vocab_set = set()
    for sentence in all_data:
        for term in sentence.split():
            vocab_set.add(term)
    vocab = list(vocab_set)
    
    return [questions, sentences, labels, max(len_q_list), max(len_s_list), vocab]


def get_test_qids_labels(dataset_folder, set_folder):
    set_path = os.path.join(dataset_folder, set_folder)
    qids = [line.strip() for line in open(os.path.join(set_path, 'id.txt')).readlines()]
    labels = np.array([int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt')).readlines()])
    return qids, labels


if __name__ == "__main__":

    vocab = ["unk", "idontreallythinkthiswordexists", "hello"]

    # w2v_dict = {}
    # load_cached_embeddings("../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache", vocab, w2v_dict)
    
    w2v_dict, num_words, vec_dim = word2vec_load_bin_vec("../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.bin", vocab)
    for w, v in w2v_dict.items():
        print(w, v)

    # vocab, emb = load_word_embeddings("../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.bin")
    # print(len(vocab))
    # print(emb[vocab['apple']])

    