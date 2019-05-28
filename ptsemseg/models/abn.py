import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ptsemseg.models._util import try_index

from modules import IdentityResidualBlock, ABN, GlobalAvgPool2d
from modules.bn import ABN, InPlaceABN, InPlaceABNSync



class abn(nn.Module):
    def __init__(self,
                 structure = [3, 3, 6, 3, 1, 1],
                 norm_act=partial(InPlaceABN, activation="leaky_relu", slope=.01), # PUT THIS INSIDE??????
                 n_classes=0,
                 dilation=(1, 2, 4, 4),
                 in_channels_head = 4096,      # THIS AND BELOW ARGS FOR HEAD, VALS TAKEN FROM TEST FILE
                 out_channels_head = 256,
                 hidden_channels=256,
                 dilations_head=(12, 24, 36),
                 pooling_size=(84, 84)):
        """Wider ResNet with pre-activation (identity mapping) blocks. With the DeeplabV3 head.

        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(abn, self).__init__()
        self.structure = structure
        self.dilation = dilation




        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil,
                                          dropout=drop)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
#        if n_classes != 0:
#            self.classifier = nn.Sequential(OrderedDict([
#                ("avg_pool", GlobalAvgPool2d()),
#                ("fc", nn.Linear(in_channels, n_classes))
#            ]))
        
        
        ####### HEAD
        
        self.pooling_size = pooling_size

        # IN THE PAPER THEY USE 9 INSTEAD OF 3 HERE. BUT IN THE GIT TEST FILE THEY USE 3 AS IT USES THESE IN DEEPLAB.PY. SUGGESTS THEIR BEST RESULT IS WITH 3
        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels_head, hidden_channels, 1, bias=False),  
            nn.Conv2d(in_channels_head, hidden_channels, 3, bias=False, dilation=dilations_head[0], padding=dilations_head[0]),
            nn.Conv2d(in_channels_head, hidden_channels, 3, bias=False, dilation=dilations_head[1], padding=dilations_head[1]),
            nn.Conv2d(in_channels_head, hidden_channels, 3, bias=False, dilation=dilations_head[2], padding=dilations_head[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels_head, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels_head, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels_head, 1, bias=False)
        self.red_bn = norm_act(out_channels_head)

        self.reset_parameters(self.map_bn.activation, self.map_bn.slope)
        
        self.cls = nn.Conv2d(out_channels_head, n_classes, 1)
        
        
    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ABN):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)        
        
    def forward(self, img):
        
        #print("FORWARD: START")
        out_size = img.shape[-2:]   # maybe move to init
        
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out_body = self.bn_out(out)
        #print("FORWARD: END OF BODY")
        ####### HEAD

        # Map convolutions
        out = torch.cat([m(out_body) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(out_body)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, out_body.size(2), out_body.size(3))

        out += pool
        out = self.red_bn(out)
        
        out = self.cls(out)
        
        #out = functional.interpolate(out, size=out_size, mode="bilinear")
        out = functional.upsample(out, size=out_size, mode="bilinear") # gives deprecation warning
        
        # Note: Mapillary use online bootstrapping for training which is not included here.
        #print("FORWARD: END")
        
        return out
    
    
    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
        return pool




