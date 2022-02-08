from core import config

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import BatchNorm2d
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import functional as F

from torchvision.transforms import CenterCrop

import torch


class Block(Module):
    """ A base class to store the convolution and RELU layers
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.batch_norm = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.batch_norm(self.conv1(x))))


class Encoder(Module):
    """ Encoder class of the Unet model.
    """
    def __init__(self, channels=(3, 16, 32, 64)):
        """ Initialise the encoder by storing the encoder blocks and maxpooling layer
        """
        super().__init__()
        self.enc_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        block_outputs = []  # initialize an empty list to store the intermediate outputs

        """ loop through the encoder blocks, pass the inputs through the current encoder block,
        store the outputs, and then apply maxpooling on the output """
        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs


class Decoder(Module):
    """ Decoder class of the Unet model.
    """
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        """ Initialise the number of channels, upsampler blocks, and decoder blocks
        """
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, enc_features):
        for i in range(len(self.channels) - 1):  # loop through the number of channels
            x = self.upconvs[i](x)  # pass the inputs through the upsampler blocks

            """ crop the current features from the encoder blocks, 
            concatenate them with the current upsampled features,
            and pass the concatenated output through the current decoder block """
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)

        return x  # return the final decoder output

    def crop(self, enc_features, x):
        """grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        """
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        return enc_features  # return the cropped features


class UNet(Module):
    """ Unet model
    """
    def __init__(self, enc_channels=(3, 16, 32, 64), dec_channels=(64, 32, 16), nb_classes=1, retain_dim=True,
                 out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        """ initialize the encoder and decoder """
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        """ initialize the regression head and store the class variables """
        self.head = Conv2d(dec_channels[-1], nb_classes, 1)
        self.retain_dim = retain_dim
        self.outSize = out_size

    def forward(self, x):
        enc_features = self.encoder(x)  # grab the features from the encoder

        """ pass the encoder features through decoder making sure that
        their dimensions are correct """
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])

        """" pass the decoder features through the regression head to
        obtain the annotations """
        mapping = self.head(dec_features)

        """ if we are retaining the original output dimensions,
        we need to resize the output to match them """
        if self.retain_dim:
            mapping = F.interpolate(mapping, self.outSize)

        return mapping  # return the mapping

