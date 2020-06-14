""" MISR-GRU for multi-image super resolution """

import torch.nn as nn
import torch
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        '''
        Args:
            channel_size : int (c), number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''

        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (b, c, w, h), hidden representations
        Returns:
            x + residual: tensor (b, c, w, h), new hidden representations
        '''

        residual = self.block(x)
        return x + residual


class EncoderUnit(nn.Module):
    def __init__(self, in_channels, num_res_layers, kernel_size, channel_size):
        '''

        Each unit in the encoder is composed 2 conv layers and 2 residual blocks and parametric-relu as the
            activation function
        Args:
            in_channels : int (c_in), number of input channels
            num_res_blocks : int, number of resnet blocks
            kernel_size : int, shape of a 2D kernel,
            channel_size : int (c), number of hidden channels
        '''
        super(EncoderUnit, self).__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_res_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (b, c_in, W, H), input
        Returns:
            out: tensor (b, C, W, H), hidden representations
        '''

        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class ConvGRUUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        '''
            Args:
                in_channels : int (i_c), number of input channels
                hidden_channels : int (h_c), number of hidden channels
                kernel_size : kernel size
        '''
        super(ConvGRUUnit, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.reset_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, prev_state):
        '''
            Args:
                x : tensor (b, c, w, h), input to gru unit
            Returns:
                out: tensor (b, h_c, w, h), hidden states
        '''

        # generate empty prev_state, if it is None
        if prev_state is None:
            state_size = [x.shape[0], self.hidden_channels] + list(x.shape[2:])
            prev_state = Variable(torch.zeros(state_size))
            if torch.cuda.is_available():
                prev_state = prev_state.cuda()

        stacked = torch.cat([x, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))
        candidate_state = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        new_state = (1 - update) * prev_state + update * candidate_state

        return new_state


class FusionModule(nn.Module):
    def __init__(self, fuse_config):
        '''
        Args:
            fuse_config : dict, fusion configuration
        '''
        super(FusionModule, self).__init__()
        self.input_channels = fuse_config["in_channels"]
        self.num_hidden_layers = fuse_config["num_hidden_layers"]
        hidden_channels = fuse_config["hidden_channels"]
        kernel_sizes = fuse_config["kernel_sizes"]

        gru_units = []
        for i in range(0, self.num_hidden_layers):
            cur_input_dim = self.input_channels if i == 0 else hidden_channels[i - 1]
            gru_unit = ConvGRUUnit(in_channels=cur_input_dim, hidden_channels=hidden_channels[i],
                                   kernel_size=kernel_sizes[i])
            gru_units.append(gru_unit)
        self.gru_units = nn.ModuleList(gru_units)

    def forward(self, x, alphas, h=None):

        '''
            Args:
                x : tensor (B, L, C, W, H), hidden states
                alphas : tensor (B, L, 1, 1, 1), boolean indicator (0 if padded low-res view, 1 otherwise)
            Returns:
                out: tensor (B, C, W, H), fused hidden state
        '''

        if h is None:
            hidden_states = [None] * self.num_hidden_layers
        num_low_res = x.shape[1]
        cur_layer_input = x

        for l in range(self.num_hidden_layers):
            gru_unit = self.gru_units[l]
            h = hidden_states[l]

            out = []
            for t in range(num_low_res):
                h = gru_unit(cur_layer_input[:, t, :, :, :], h)
                out.append(h)

            out = torch.stack(out, dim=1)
            cur_layer_input = out

        fused_representations = torch.sum(cur_layer_input * alphas, 1) / torch.sum(alphas, 1)

        return fused_representations


class Decoder(nn.Module):
    def __init__(self, dec_config):
        '''
        Args:
            dec_config : dict, decoder configuration
        '''

        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=dec_config["deconv"]["in_channels"],
                                                       out_channels=dec_config["deconv"]["out_channels"],
                                                       kernel_size=dec_config["deconv"]["kernel_size"],
                                                       stride=dec_config["deconv"]["stride"]),
                                    nn.PReLU())

        self.final = nn.Conv2d(in_channels=dec_config["final"]["in_channels"],
                               out_channels=dec_config["final"]["out_channels"],
                               kernel_size=dec_config["final"]["kernel_size"],
                               padding=dec_config["final"]["kernel_size"] // 2)

    def forward(self, x):
        '''
        Decodes a hidden state x.
        Args:
            x : tensor (b, c, w, h), fused hidden state
        Returns:
            out: tensor (b, c_out, 3xw, 3xw), super resolved image

        '''
        x = self.deconv(x)
        x = self.final(x)

        return x


class MISRGRU(nn.Module):
    ''' MISRGRU, a neural network for multi-image super resolution (MISR) using ConvGRU. '''

    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''

        super(MISRGRU, self).__init__()
        self.unit1 = EncoderUnit(in_channels=config["encoder"]["in_channels"],
                                 num_res_layers=config["encoder"]["num_res_blocks"],
                                 kernel_size=config["encoder"]["kernel_size"],
                                 channel_size=config["encoder"]["channel_size"])
        self.fuse = FusionModule(config["fusion"])
        self.decode = Decoder(config["decoder"])
        self.unit2 = EncoderUnit(in_channels=config["encoder"]["channel_size"] * 2,
                                 num_res_layers=config["encoder"]["num_res_blocks"],
                                 kernel_size=config["encoder"]["kernel_size"],
                                 channel_size=config["encoder"]["channel_size"])

    def forward(self, lrs, alphas):
        '''
        Super resolves a batch of low-resolution images.
        Args:
            lrs : tensor (B, L, W, H), low-resolution images
            alphas : tensor (B, L), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved images
        '''

        ###################### Encode #########################
        batch_size, num_low_res, height, width = lrs.shape
        # Extend channel dimension
        lrs = lrs.view(batch_size, num_low_res, 1, height, width)
        # create a reference view based on median statistics
        refs, _ = torch.median(lrs[:, :9], 1)

        alphas = alphas.view(-1, num_low_res, 1, 1, 1)

        # encode LR views
        lrs = lrs.view(batch_size * num_low_res, 1, height, width)
        lrs = self.unit1(lrs)
        lrs = lrs.view(batch_size, num_low_res, -1, height, width)

        # encode ref view
        refs = self.unit1(refs)
        refs = refs.unsqueeze(1)
        refs = refs.repeat(1, num_low_res, 1, 1, 1)

        # Co-register ref encoded features with LR encoded features
        out = torch.cat([lrs, refs], 2)
        out = out.view(batch_size * num_low_res, -1, height, width)
        out = self.unit2(out)
        out = out.view(batch_size, num_low_res, -1, height, width)  # tensor (b, l, c, w, h)

        ###################### Fuse #########################
        out = self.fuse(out, alphas)

        ###################### Decode #########################
        out = self.decode(out)

        return out
