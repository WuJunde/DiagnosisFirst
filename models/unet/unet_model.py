""" Full assembly of the parts to form the complete network """
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
from tag.tag import Stage
from .unet_parts import *
from torch import nn
import torch
from .res_net import resnet34, resnet18, resnet50, resnet101, resnet152, BasicBlock, Bottleneck, ResNet
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class SaveFeatures():
    features = None

    # def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    # def hook_fn(self, module, input, output): self.features = output

    # def remove(self): self.hook.remove()

    def __init__(self,m):
        self._outputs_lists = {}
        self.mymodule = m
        m.register_forward_hook(hook=self.save_output_hook)

    def save_output_hook(self, _, input, output):
        self._outputs_lists[input[0].device.index] = output
        self.features = self._outputs_lists

    def forward(self, x) -> list:
        self._outputs_lists[x.device.index] = []
        self.mymodule(x)
        return self._outputs_lists[x.device.index]


class UnetStageBlock(nn.Module):
    def __init__(self, stage, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.g_conv = nn.Conv2d(x_in, x_out * 2, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.stage = stage

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p, give):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        res = self.bn(F.relu(cat_p))
        g_p = self.g_conv(give)
        res = self.stage(g_p,res)
        return res

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        res = self.bn(F.relu(cat_p))
        return res

class TransUNet(nn.Module):

    def __init__(self, args, resnet='resnet34', num_classes=2, pretrained=False,
                in_chans=3,
                inplanes=64,
                num_layers=(3, 4, 6, 3),
                num_chs=(256, 512, 1024, 2048),
                num_strides=(1, 2, 2, 2),
                num_heads=(1, 1, 1, 1),
                num_parts=(1, 1, 1, 1),
                patch_sizes=(1, 1, 1, 1),
                drop_path=0.1,
                num_enc_heads=(1, 1, 1, 1),
                act=nn.GELU,
                ffn_exp=3,
                has_last_encoder=False
                ):
        super().__init__()
        # super(ResUnet, self).__init__()

        ''' ~~~~~ For the embedding transformer~~~~~'''
        cut, lr_cut = [8, 6]

        dim = args.dim #dim of transformer sequence, D of E
        img_size = 8 #emebeding 8*8*512
        channels = 512
        patch_size = args.patch_size
        depth = args.depth
        heads = args.heads
        mlp_dim = args.mlp_dim
        dim_head = 64

        assert img_size % patch_size == 0 , 'Image dimensions must be divisible by the patch size.'

        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        '''~~~~~~End of embedding transformer~~~~~'''

        'unet and goinnet parameters'
        if resnet == 'resnet34':
            base_model = resnet34
            self.goin = GoinNet(args, resnet,  **args.model_cfg)
        elif resnet == 'resnet18':
            base_model = resnet18
            self.goin = GoinNet(args, resnet,  **args.model_cfg)
        elif resnet == 'resnet50':
            base_model = resnet50
            self.goin = GoinNet(args, resnet,  **args.model_cfg)
        elif resnet == 'resnet101':
            base_model = resnet101
            self.goin = GoinNet(args, resnet,  **args.model_cfg)
        elif resnet == 'resnet152':
            base_model = resnet152
            self.goin = GoinNet(args, resnet,  **args.model_cfg)
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')
        
        '''define the stage for goinnet giving'''
        last_chs = (256,256,256,256)
        num_chs = (256, 256, 256, 256)
        down_samples = (2,4,8,16)
        n_l = 1
        stage_list = []
        for i in range(4):
            stage_list.append(
                Stage(last_chs[i],
                    num_chs[i],
                    n_l,
                    num_heads=num_heads[i], #1,2,4,8
                    num_parts = (patch_sizes[i]**2 * (args.image_size // down_samples[i] // patch_sizes[i])**2),
                    patch_size=patch_sizes[i],  #8,8,8,8
                    drop_path=drop_path, #0.05
                    ffn_exp=ffn_exp,    #mlp hidden fea
                    last_enc=has_last_encoder and i == len(num_layers) - 1)
                        )
        self.stages = nn.ModuleList(stage_list)

        # self.stage_encoding = Stage(512,
        #             512,
        #             1,
        #             num_heads=16, #1,2,4,8
        #             num_parts = (8**2),
        #             patch_size=8,  #8,8,8,8
        #             drop_path=drop_path, #0.05
        #             ffn_exp=ffn_exp,    #mlp hidden fea
        #             last_enc=has_last_encoder and i == len(num_layers) - 1)
        '''end'''

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        self.check_layer = layers
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        ''' coarse decoder'''
        self.cup1 = UnetBlock(512, 256, 256)
        self.cup2 = UnetBlock(256, 128, 256)
        self.cup3 = UnetBlock(256, 64, 256)
        self.cup4 = UnetBlock(256, 64, 256)
        self.cup5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

        '''coarse decoder end'''

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetStageBlock(self.stages[3], 512, 256, 256)
        self.up2 = UnetStageBlock(self.stages[2], 256, 128, 256)
        self.up3 = UnetStageBlock(self.stages[1], 256, 64, 256)
        self.up4 = UnetStageBlock(self.stages[0], 256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)


        '''~~~ self definition ~~~'''
        self.fc = nn.Linear(6,dim)
        self.tr_conv = nn.ConvTranspose2d(512, 512, 2, stride=2)

    def forward(self, x):
        img = x
        # x = torch.cat((x,heatmap),1)
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]

        '''coarse decoder'''
        c = self.cup1(x, self.sfs[3].features[x.device.index])
        c = self.cup2(c, self.sfs[2].features[x.device.index])
        c = self.cup3(c, self.sfs[1].features[x.device.index])
        c = self.cup4(c, self.sfs[0].features[x.device.index])
        coarse = self.cup5(c)
        '''coarse end'''

        '''~~~ main f'''
        emb, give = self.goin(img, coarse, self.sfs)
        # x = self.stage_encoding(emb,x)
        '''~~~ 1: ENDs ~~~'''


        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features[x.device.index], give[3])
        x = self.up2(x, self.sfs[2].features[x.device.index], give[2])
        x = self.up3(x, self.sfs[1].features[x.device.index], give[1])
        x = self.up4(x, self.sfs[0].features[x.device.index], give[0])
        fea = x
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class UNet(nn.Module):

    def __init__(self, args, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        # super(ResUnet, self).__init__()

        ''' ~~~~~ For the embedding transformer~~~~~'''
        cut, lr_cut = [8, 6]

        'unet and goinnet parameters'
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained,inplanes = 9).children())[:cut]
        self.check_layer = layers
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers


        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x, heatmap):
        x = torch.cat((x,heatmap),1)
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]


        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features[x.device.index])
        x = self.up2(x, self.sfs[2].features[x.device.index])
        x = self.up3(x, self.sfs[1].features[x.device.index])
        x = self.up4(x, self.sfs[0].features[x.device.index])
        fea = x
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''

        '''
        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]
        '''
        return output

    def close(self):
        for sf in self.sfs: sf.remove()


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class GoinNet(nn.Module):
    def __init__(self,
                args, 
                resnet,
                in_chans=3,
                inplanes=64,
                num_layers=(3, 4, 6, 3),
                num_chs=(256, 512, 1024, 2048),
                num_strides=(1, 2, 2, 2),
                num_heads=(1, 1, 1, 1),
                num_parts=(1, 1, 1, 1),
                patch_sizes=(1, 1, 1, 1),
                drop_path=0.1,
                num_enc_heads=(1, 1, 1, 1),
                act=nn.GELU,
                ffn_exp=3,
                has_last_encoder=False,
                pretrained=False
                ):
        self.inplanes = 64
        super(GoinNet, self).__init__()

        cut, lr_cut = [8, 6]
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        'select diagnosis backbone'
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained,inplanes = 4).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]

        last_chs = (64,64,128,256)
        num_chs = (64, 64, 128, 256)
        down_samples = (2,4,8,16)
        n_l = 1
        stage_list = []
        for i in range(4):
            stage_list.append(
                Stage(last_chs[i],
                    num_chs[i],
                    n_l,
                    num_heads=num_heads[i], #1,2,4,8
                    num_parts = (patch_sizes[i]**2 * (args.image_size // down_samples[i] // patch_sizes[i])**2) ,
                    patch_size=patch_sizes[i],  #8,8,8,8
                    drop_path=drop_path, #0.05
                    ffn_exp=ffn_exp,    #mlp hidden fea
                    last_enc=has_last_encoder and i == len(num_layers) - 1)
                        )
        self.stages = nn.ModuleList(stage_list)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img, x, p):

        x = torch.cat((img,x),1)
        x = F.relu(self.rn(x))

        x = self.stages[0](p[0].features[x.device.index], self.sfs[0].features[x.device.index])
        turn0 = x

        x = self.stages[1](p[1].features[x.device.index], self.sfs[1].features[x.device.index])
        turn1 = x

        x = self.stages[2](p[2].features[x.device.index], self.sfs[2].features[x.device.index])
        turn2 = x

        x = self.stages[3](p[3].features[x.device.index], self.sfs[3].features[x.device.index])
        turn3 = x

        return x, [turn0, turn1, turn2, turn3]
