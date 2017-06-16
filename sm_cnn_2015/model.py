import torch
import torch.nn as nn
import torch.nn.functional as F

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class QAModel(nn.Module):

    @staticmethod
    def save(model, model_fname):
        torch.save(model, model_fname)


    @staticmethod
    def load(model_fname):
        return torch.load(model_fname)


    def __init__(self, input_n_dim, filter_width, \
            conv_filters=100, no_ext_feats=False, ext_feats_size=4, n_classes=2, cuda=False):
        super(QAModel, self).__init__()

        self.no_ext_feats = no_ext_feats

        self.conv_channels = conv_filters
        self.n_hidden = 2*self.conv_channels + (0 if no_ext_feats else ext_feats_size)

        self.conv_q = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()
        )

        self.conv_a = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()
        )

        self.combined_feature_vector = nn.Linear(self.n_hidden, self.n_hidden)

        self.combined_features_activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hidden = nn.Linear(self.n_hidden, n_classes)
        self.logsoftmax = nn.LogSoftmax()

        if cuda and torch.cuda.is_available():
            self.conv_q, self.conv_a = self.conv_q.cuda(), self.conv_a.cuda()
            self.combined_feature_vector = self.combined_feature_vector.cuda()
            self.combined_features_activation = self.combined_features_activation.cuda()
            self.dropout, self.hidden, self.logsoftmax = self.dropout.cuda(), self.hidden.cuda(), self.logsoftmax.cuda()

    def forward(self, question, answer, ext_feats):

        # print(question.size())
        q = self.conv_q.forward(question)
        # print(q.size())
        q = F.max_pool1d(q, q.size()[2])
        # print(q.size())
        q = q.view(-1, 1, self.conv_channels)
        # print(q.size())
        logger.debug('forward q: {}'.format(q))

        # print(answer.size())
        a = self.conv_a.forward(answer)
        # print(a.size())
        a = F.max_pool1d(a, a.size()[2])
        # print(a.size())
        a = a.view(-1, 1, self.conv_channels)
        # print(a.size())

        # print(ext_feats.size())
        e = ext_feats.view(-1, 1, ext_feats.size()[1])
        # print(e.size())
        x = None
        if self.no_ext_feats:
            x = torch.cat([q, a], 1)
            # logger.debug('no_ext_feats')
        else:
            x = torch.cat([q, a, e], 2)
            # logger.debug('with ext_feats')
        # print(x.size())
        # logger.debug('featvec x: {}'.format(x))
        # logger.debug(x.creator)
        x = x.view(-1, self.n_hidden)
        # print(x.size())

        # print(self.n_hidden)

        x = self.combined_feature_vector.forward(x)
        x = self.combined_features_activation.forward(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.logsoftmax(x)

        return x


