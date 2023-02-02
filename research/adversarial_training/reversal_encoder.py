from research.adversarial_training.common_layers import *
import json
import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, n_speaker, hidden_size):
        super().__init__()
        self.L1 = nn.Linear(hidden_size, n_speaker)
        nn.init.xavier_uniform(self.L1.weight.data)
    def forward(self, input):
        x = self.L1(input)
        return x



class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l, c):
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return ctx.l * grad_output.neg(), None, None


class GradientClippingFunction(torch.autograd.Function):
    """Clip gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return grad_output, None

class Classifier(torch.nn.Module):
    """Adversarial classifier (with two FC layers) with a gradient reversal layer.

    Arguments:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        hidden_dim -- size of the hiden layer
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
    Keyword arguments:
        scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradientts
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self._output_dim = output_dim
        self._classifier = nn.Sequential(
            Linear(input_dim, hidden_dim),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self._classifier(x)
        return x

class ReversalEncoder(torch.nn.Module):
    """Adversarial classifier (with two FC layers) with a gradient reversal layer.

    Arguments:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        hidden_dim -- size of the hiden layer
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
    Keyword arguments:
        scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradientts
    """

    def __init__(self, n_mel_channels, model_hidden_size, model_embedding_size, n_speakers,
                 gradient_clipping_bounds, scale_factor=1.0):
        super(ReversalEncoder, self).__init__()
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds
        self._classifier = SpeakerEncoder(n_mel_channels, model_hidden_size, model_embedding_size, n_speakers)

    def forward(self, x, input_lengths):
        x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        x = self._classifier(x, input_lengths)
        return x




class SpeakerEncoder(nn.Module):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling

    '''

    def __init__(self, n_mel_channels, model_hidden_size, model_embedding_size, n_speakers):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(n_mel_channels, int(model_hidden_size / 2),
                            num_layers=2, batch_first=True,  bidirectional=True, dropout=0.2)
        self.projection1 = LinearNorm(model_hidden_size,
                                      model_embedding_size,
                                      w_init_gain='tanh')
        self.projection2 = LinearNorm(model_embedding_size, n_speakers)



    def forward(self, x, input_lengths):
        '''
         x  [batch_size, mel_bins, T]

         return
         logits [batch_size, n_speakers]
         embeddings [batch_size, embedding_dim]
        '''
        x = x.transpose(0,1)
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        outputs = torch.sum(outputs, dim=1) / sorted_lengths.unsqueeze(1).float()  # mean pooling -> [batch_size, dim]

        outputs = F.tanh(self.projection1(outputs))
        outputs = outputs[initial_index]
        # L2 normalizing #
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        return logits, embeddings

    def inference(self, x):
        x = x.transpose(0, 1)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs = torch.sum(outputs, dim=1) / float(outputs.size(1))  # mean pooling -> [batch_size, dim]
        outputs = F.tanh(self.projection1(outputs))
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        pid = torch.argmax(logits, dim=1)

        return pid, embeddings


def sort_batch(data, lengths):
    '''
    sort data by length
    sorted_data[initial_index] == data
    '''
    sorted_lengths, sorted_index = lengths.sort(0, descending=True)
    sorted_data = data[sorted_index]
    _, initial_index = sorted_index.sort(0, descending=False)

    return sorted_data, sorted_lengths, initial_index