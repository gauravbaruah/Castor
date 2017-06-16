import time

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm

import utils
from model import QAModel

class Trainer(object):

    def __init__(self, dataset_folder, train_set, dev_set, test_set,    # input data
                 word_vectors_file,                                     # word embeddings
                 eta, mom,                                              # optimization params
                 filter_width, num_conv_filters,                        # convolution params
                 cuda=False,
                 load_model_file=None):                                 # load model file
                
        # reset the random seeds for every instance of trainer.
        # needed to ensure reproduction of random word vectors for out of vocab terms
        torch.manual_seed(1234)
        np.random.seed(1234)

        # setup data splits and load embeddings
        self.data_splits = {}
        self.embeddings = None
        
        # read in data splits
        vocabulary = set()
        for set_folder in [test_set, dev_set, train_set]:
            if set_folder:
                questions, sentences, labels, maxlen_q, maxlen_s, vocab = \
                    utils.read_in_dataset(dataset_folder, set_folder)

                vocabulary = vocabulary.union(set(vocab))

                self.data_splits[set_folder] = [questions, sentences, labels, maxlen_q, maxlen_s]
                default_ext_feats = [np.zeros(4)] * len(self.data_splits[set_folder][0])
                self.data_splits[set_folder].append(default_ext_feats)

        # load word vectors
        w2v, num_words, self.vec_dim = \
            utils.word2vec_load_bin_vec(word_vectors_file, list(vocabulary))
        w2v["random.unknown"] = np.random.uniform(-0.25, 0.25, self.vec_dim)
        w2v["zero.padding"] = np.zeros(self.vec_dim)
        num_words += 2
        self.w2i = defaultdict(int) # word to index for embedding object
        all_embeddings = []
        for index, word_embedding in enumerate(w2v.items()):
            word, embedding = word_embedding
            self.w2i[word] = index
            all_embeddings.append(embedding)
        embeddings_matrix = np.vstack(all_embeddings)
        print(embeddings_matrix.shape)
        self.embeddings = nn.Embedding(num_words, self.vec_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_matrix))
        self.embeddings.requires_grad = False

        # create model
        if load_model_file:
            self.model = QAModel.load(load_model_file)
        else:
            self.model = QAModel(self.vec_dim, filter_width, num_conv_filters, cuda)
        self.cuda = cuda

        # optimization parameters
        self.reg = 1e-5
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=eta, momentum=mom, \
            weight_decay=self.reg)


    def save_model(self, model_file):
        QAModel.save(self.model, model_file)

    def regularize_loss(self, loss):

        flattened_params = []

        for p in self.model.parameters():
            f = p.data.clone()
            flattened_params.append(f.view(-1))

        fp = torch.cat(flattened_params)

        loss = loss + 0.5 * self.reg * fp.norm() * fp.norm()

        # for p in self.model.parameters():
        #     loss = loss + 0.5 * self.reg * p.norm() * p.norm()

        return loss


    def _train(self, xq, xa, ext_feats, ys):

        self.optimizer.zero_grad()
        output = self.model(xq, xa, ext_feats)
        loss = self.criterion(output, ys)
        # logger.debug('loss after criterion {}'.format(loss))

        # NOTE: regularizing location 1
        flattened_params = []
        for p in self.model.parameters():
            f = p.data.clone()
            flattened_params.append(f.view(-1))
        fp = torch.cat(flattened_params)
        loss.add_(0.5 * self.reg * fp.norm() * fp.norm())
        #     logger.debug('loss after regularizing {}'.format(loss))

        loss.backward()

        # logger.debug('AFTER backward')
        #logger.debug('params {}'.format([p for p in self.model.parameters()]))
        # logger.debug('params grads {}'.format([p.grad for p in self.model.parameters()]))

        # NOTE: regularizing location 2. It would seem that location 1 is correct?
        #if not self.no_loss_reg:
        #    loss = self.regularize_loss(loss)
            # logger.debug('loss after regularizing {}'.format(loss))

        self.optimizer.step()

        # logger.debug('AFTER step')
        #logger.debug('params {}'.format([p for p in self.model.parameters()]))
        # logger.debug('params grads {}'.format([p.grad for p in self.model.parameters()]))

        return loss.data[0], self.pred_equals_y(output, ys)


    def pred_equals_y(self, pred, y):
        _, best = pred.max(1)
        best = best.data.long().squeeze()
        return torch.sum(y.data.long() == best)


    def test(self, set_folder, batch_size):
        print('----- Predictions on {} '.format(set_folder))

        questions, sentences, labels, maxlen_q, maxlen_s, ext_feats = \
            self.data_splits[set_folder]
        word_vectors, vec_dim = self.embeddings, self.vec_dim

        self.model.eval()

        batch_size = 1

        total_loss = 0.0
        total_correct = 0.0
        num_batches = np.ceil(len(questions)/batch_size)
        y_pred = np.zeros(len(questions))
        ypc = 0

        for k in range(int(num_batches)):
            batch_start = k * batch_size
            batch_end = (k+1) * batch_size
            # convert raw questions and sentences to tensors
            xq, xa, x_ext_feats, y = self.get_tensorized_inputs(
                questions[batch_start:batch_end], maxlen_q,
                sentences[batch_start:batch_end], maxlen_s,
                labels[batch_start:batch_end],
                ext_feats[batch_start:batch_end]
                )

            
            pred = self.model(xq, xa, x_ext_feats)
            loss = self.criterion(pred, y)
            pred = torch.exp(pred)
            total_loss += loss
            total_correct += self.pred_equals_y(pred, y)

            y_pred[ypc] = pred.data.squeeze()[1]
            # ^ we want to score for relevance, NOT the predicted class
            ypc += 1

        print('{}_correct {}'.format(set_folder, total_correct))
        print('{}_loss {}'.format(set_folder, total_loss.data[0]))
        print('{} total {}'.format(set_folder, len(labels)))
        print('{}_loss = {:.4f}, acc = {:.4f}'.format(set_folder, total_loss.data[0]/len(labels), float(total_correct)/len(labels)))
        #print('{}_loss = {:.4f}'.format(set_folder, total_loss.data[0]/len(labels)))

        return y_pred


    def train(self, set_folder, batch_size, debug_single_batch):
        train_start_time = time.time()

        questions, sentences, labels, maxlen_q, maxlen_s, ext_feats = \
            self.data_splits[set_folder]

        # set model for training modep
        self.model.train()

        train_loss, train_correct = 0., 0.
        num_batches = np.ceil(len(questions)/float(batch_size))

        for k in tqdm(range(int(num_batches))):
            batch_start = k * batch_size
            batch_end = (k+1) * batch_size

            # convert raw questions and sentences to tensors
            xq, xa, x_ext_feats, y = self.get_tensorized_inputs(
                questions[batch_start:batch_end], maxlen_q, 
                sentences[batch_start:batch_end], maxlen_s,
                labels[batch_start:batch_end],
                ext_feats[batch_start:batch_end]
                )

            

            batch_loss, batch_correct = self._train(xq, xa, x_ext_feats, y)

            #logger.debug('batch_loss {}, batch_correct {}'.format(batch_loss, batch_correct))
            train_loss += batch_loss
            train_correct += batch_correct
            if debug_single_batch:
                break


        print('train_correct {}'.format(train_correct))
        print('train_loss {}'.format(train_loss))
        print('total training batches = {}'.format(num_batches))
        print('train_loss = {:.4f}'.format(
            train_loss/num_batches
        ))
        print('training time = {:.3f} seconds'.format(time.time() - train_start_time))
        return train_correct/num_batches


    def make_input_matrix(self, sentence, maxlen):
        terms = sentence.strip().split()[:maxlen]
        # NOTE: we are truncating the inputs to 60 words.
        
        #terms.extend(["random.unknown"]*(maxlen-len(terms)))
        terms.extend(["zero.padding"]*(maxlen-len(terms)))
        
        indices = Variable(torch.LongTensor(
            [i for i in map(lambda t: self.w2i[t if t in self.w2i else "zero.padding"], terms)]
        ))
        embeddings_matrix = self.embeddings(indices)
        
        # word_embeddings = torch.zeros(len(terms), vec_dim).type(torch.DoubleTensor)
        # for i in range(len(terms)):
        #     word = terms[i]
        #     emb = torch.from_numpy(word_vectors[word])
        #     word_embeddings[i] = emb

        #input_tensor = Variable(torch.zeros(1, self.embeddings.embedding_dim, len(terms)))
        #input_tensor[0] = torch.transpose(embeddings_matrix, 0, 1)
        
        input_tensor = torch.transpose(embeddings_matrix, 0, 1)
        
        return input_tensor


    def get_tensorized_inputs(self, batch_ques, maxlen_q, batch_sents, maxlen_s, batch_labels, batch_ext_feats):
        batch_size = len(batch_ques)
        # NOTE: ideal batch size is one, because sentences are all of different length.
        # In other words, we have no option but to feed in sentences one by one into the model
        # and compute loss at the end.

        # TODO: what if the sentences in a batch are all of different lengths?
        # - should be have the longest sentence as 2nd dim?
        #   - would zero endings work for other smaller sentences?

        maxlen_q = min(60, maxlen_q)
        maxlen_s = min(60, maxlen_s)

        y = torch.LongTensor(batch_size).type(torch.LongTensor)
        q_tensor = Variable(torch.zeros(batch_size, self.embeddings.embedding_dim, maxlen_q))
        a_tensor = Variable(torch.zeros(batch_size, self.embeddings.embedding_dim, maxlen_s))
        ext_feats = Variable(torch.zeros(batch_size, len(batch_ext_feats[0]), 1))

        # print(y.size())
        # print(q_tensor.size())
        # print(a_tensor.size())
        # print(ext_feats.size())

        
        for i in range(len(batch_ques)):
            # xq = Variable(self.make_input_matrix(batch_ques[i], word_vectors, vec_dim))
            # xs = Variable(self.make_input_matrix(batch_sents[i], word_vectors, vec_dim))
            q_tensor[i] = self.make_input_matrix(batch_ques[i], maxlen_q)
            a_tensor[i] = self.make_input_matrix(batch_sents[i], maxlen_s)
            xf = Variable(torch.FloatTensor(batch_ext_feats[i]))
            ext_feats[i] = torch.unsqueeze(xf, 0)
            y[i] = batch_labels[i]

        # print(y)
        # print(q_tensor)
        # print(a_tensor)
        # print(ext_feats)

        return q_tensor, a_tensor, ext_feats, Variable(y)


