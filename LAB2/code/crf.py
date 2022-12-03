import torch
import torch.nn as nn
import crf_utils
from conv import Conv


class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.conv_layer = Conv(5)
        self.params = nn.Parameter(torch.zeros(num_labels * embed_dim + num_labels**2))

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    # def init_params(self):
    #     """
    #     Initialize trainable parameters of CRF here
    #     """
    #     self.conv_layer = Conv(5)
    #     self.params = nn.Parameter(torch.zeros(num_labels * embed_dim + num_labels**2))

    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        X = self.__reshape_before_conv__(X)
        features = self.conv_layer(X)
        features = self.__reshape_after_conv__(features)

        prediction = crf_utils.dp_infer(features, self.params, self.num_labels, self.embed_dim)
        return (prediction)

    def loss(self, X, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        C = 1000
        X = self.__reshape_before_conv__(X)
        self.features = self.conv_layer(X)
        self.features = self.__reshape_after_conv__(self.features)
        self.saved_for_backward = [labels, C]

        loss = crf_utils.obj_func(self.features, labels, self.params, C, self.num_labels, self.embed_dim)

        return loss

    def __reshape_before_conv__(self, X):
        X = torch.reshape(X, (X.shape[0]*X.shape[1], 1, 16, 8))
        return X

    def __reshape_after_conv__(self, X):
        X = torch.reshape(X, (X.shape[0]//14, 14, X.shape[2]*X.shape[3]))
        return X

    def backward(self):
        """
        Return the gradient of the CRF layer
        :return:
        """
        features = self.features
        labels, C = self.saved_for_backward
        gradient = crf_utils.crfFuncGrad(features, labels, self.params, C, self.num_labels, self.embed_dim)
        return gradient

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = blah
        return convfeatures
