import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head=4, block_exp=4, n_layer=8, 
                    vert_anchors=7, horz_anchors=7, seq_len=1, 
                    embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # positional embedding parameter (learnable), image + rssi
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * seq_len * vert_anchors * horz_anchors, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, rssi_tensor):
        bz = rssi_tensor.shape[0] // self.seq_len
        h, w = rssi_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.seq_len, -1, h, w)
        rssi_tensor = rssi_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, rssi_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # add (learnable) positional embedding embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, 2 * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out = x[:, :self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        rssi_tensor_out = x[:, self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)

        return image_tensor_out, rssi_tensor_out


class FusionNet(nn.Module):
    def __init__(self, config, use_dropout=False):
        super(FusionNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.backbone_img = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone_rssi = models.resnet18()
        self.backbone_rssi.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.use_transformer = config.get("use_transformer")
        if self.use_transformer == True:
            self.transformer1 = GPT(n_embd=64)
            self.transformer2 = GPT(n_embd=128)
            self.transformer3 = GPT(n_embd=256)
            self.transformer4 = GPT(n_embd=512)
        backbone_dim = 512
        latent_dim = 1024

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3)
        self.fc3 = nn.Linear(latent_dim, 4)
        self.fc4 = nn.Linear(backbone_dim, latent_dim)

        if use_dropout:
            self.dropout = nn.Dropout(p=0.1)
        self.use_dropout = use_dropout

        # Initialize FC layers
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        for m in [self.backbone_rssi]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x_img = self.backbone_img.conv1(x.get('img'))
        x_img = self.backbone_img.bn1(x_img)
        x_img = self.backbone_img.relu(x_img)
        x_img = self.backbone_img.maxpool(x_img)

        x_rssi = self.backbone_rssi.conv1(x.get('rssi'))
        x_rssi = self.backbone_rssi.bn1(x_rssi)
        x_rssi = self.backbone_rssi.relu(x_rssi)
        x_rssi = self.backbone_rssi.maxpool(x_rssi)

        x_img = self.backbone_img.layer1(x_img)
        x_rssi = self.backbone_rssi.layer1(x_rssi)
        if self.use_transformer == True:
            x_img1 = self.avgpool(x_img)
            x_rssi1 = self.avgpool(x_rssi)
            x_img1, x_rssi1 = self.transformer1(x_img1, x_rssi1)
            x_img1 = F.interpolate(x_img1, scale_factor=8, mode='bilinear', align_corners=False)
            x_rssi1 = F.interpolate(x_rssi1, scale_factor=8, mode='bilinear', align_corners=False)
            x_img = x_img + x_img1
            x_rssi = x_rssi + x_rssi1

        x_img = self.backbone_img.layer2(x_img)
        x_rssi = self.backbone_rssi.layer2(x_rssi)
        if self.use_transformer == True:
            x_img2 = self.avgpool(x_img)
            x_rssi2 = self.avgpool(x_rssi)
            x_img2, x_rssi2 = self.transformer2(x_img2, x_rssi2)
            x_img2 = F.interpolate(x_img2, scale_factor=4, mode='bilinear', align_corners=False)
            x_rssi2 = F.interpolate(x_rssi2, scale_factor=4, mode='bilinear', align_corners=False)
            x_img = x_img + x_img2
            x_rssi = x_rssi + x_rssi2

        x_img = self.backbone_img.layer3(x_img)
        x_rssi = self.backbone_rssi.layer3(x_rssi)
        if self.use_transformer == True:
            x_img3 = self.avgpool(x_img)
            x_rssi3 = self.avgpool(x_rssi)
            x_img3, x_rssi3 = self.transformer3(x_img3, x_rssi3)
            x_img3 = F.interpolate(x_img3, scale_factor=2, mode='bilinear', align_corners=False)
            x_rssi3 = F.interpolate(x_rssi3, scale_factor=2, mode='bilinear', align_corners=False)
            x_img = x_img + x_img3
            x_rssi = x_rssi + x_rssi3

        x_img = self.backbone_img.layer4(x_img)
        x_rssi = self.backbone_rssi.layer4(x_rssi)
        if self.use_transformer == True:
            x_img4 = self.avgpool(x_img)
            x_rssi4 = self.avgpool(x_rssi)
            x_img4, x_rssi4 = self.transformer4(x_img4, x_rssi4)
            x_img = x_img + x_img4
            x_rssi = x_rssi + x_rssi4

        x_img = self.backbone_img.avgpool(x_img)
        x_img = x_img.flatten(start_dim=1)
        x_rssi = self.backbone_rssi.avgpool(x_rssi)
        x_rssi = x_rssi.flatten(start_dim=1)

        if self.use_dropout:
            x = self.dropout(F.relu(self.fc1(x_img+x_rssi)))
            p_x = self.fc2(x)
            p_q = self.fc3(self.dropout(F.relu(self.fc4(x_img))))
        else:
            x = F.relu(self.fc1(x_img+x_rssi))
            p_x = self.fc2(x)
            p_q = self.fc3(F.relu(self.fc4(x_img)))
        return {'pose': torch.cat((p_x, p_q), dim=1)}
    


if __name__ == "__main__":
    x = {'img': torch.randn(2,3,224,224), 'rssi': torch.randn(2,1,224,224)}
    config = {"use_transformer":False}
    m = FusionNet(config)
    y = m(x)


