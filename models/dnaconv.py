import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

# should I use the same architecture? The sequence length in Stark et al. is 1024
class PromoterModel(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, args, embed_dim=256, time_dependent_weights=None, time_step=0.01):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.alphabet_size = 4
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = 256
        # expanded_simplex_input = (args.mode == 'dirichlet' or args.mode == 'riemannian')
        # inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1) + 1 # plus one for signal input
        # if (args.mode == 'ardm' or args.mode == 'lrar'):
        #     inp_size += 1  # plus one for the mask token of these models
        inp_size = self.alphabet_size + 1 # plus one for the mask token
        self.linear = nn.Conv1d(inp_size, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step


    # the signal here is that they follow Avdeyev et al. to condict conditional generation conditioned on a profile 
    # (which is related to the chormotain and position on the chromotain)
    # Should we do similar conditional generation on our dataset? need to check what is the common practice (the Gosai data is enhancers)
    def forward(self, x, signal, t):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))

        x = torch.cat([x,signal], dim=-1)

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out

# this is for the enhancer data in Stark et al. (the sequence length is 200)
# use this one for the Gosai data
class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            # expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            # inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            # if (args.mode == 'ardm' or args.mode == 'lrar') and not classifier:
            #     inp_size += 1 # plus one for the mask token of these models
            inp_size = self.alphabet_size #+ 1
            self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = 5 * args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(args.hidden_dim, args.hidden_dim if classifier else self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
    
    def forward(self, seq, t, cls = None, return_embedding=False):
        seq = F.one_hot(seq, num_classes=self.alphabet_size).float()
        if self.args.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.args.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.args.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.args.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat
