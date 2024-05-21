import math
import torch.nn.functional as F
import torchaudio

from nas_operations import *
from nas_utils import Genotype
from nas_utils import gumbel_softmax, drop_path


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        This is a forward function.
        :param x: Feature map
        :param weights: A tensor of weight controlling the path flow
        :return: A weighted sum of several path
        """
        output = 0
        for op_idx, op in enumerate(self._ops):
            if weights[op_idx].item() != 0:
                if math.isnan(weights[op_idx]):
                    raise OverflowError(f'weight: {weights}')
            output += weights[op_idx] * op(x)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class NASSEARCH(nn.Module):

    def __init__(self, C, emb_size, layers, primitives, criterion,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(NASSEARCH, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )
        self.specaug = FbankAug()

        self._C = C
        self._emb_size = emb_size
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._criterion = criterion
        self.drop_path_prob = drop_path_prob

        nn.Module.PRIMITIVES = primitives

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(C_prev, self._emb_size)

        self._initialize_alphas()

    def new(self):
        model_new = NASSEARCH(self._C, self._embed_dim, self._layers,
                              self.PRIMITIVES, drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x, aug, discrete=False):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = (x - torch.mean(x, dim=-1, keepdim=True))
            if aug:
                x = self.specaug(x)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if discrete:
                    weights = self.alphas_reduce
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_reduce, dim=-1))
            else:
                if discrete:
                    weights = self.alphas_normal
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_normal, dim=-1))
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
        v = F.relu(s1, inplace=True)
        v = self.global_pooling(v)
        v = torch.flatten(v, 1)
        # if not self.training:
        #     return v

        y = self.classifier(v)

        return y

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x

    def _loss(self, input, target):
        logits = self.forward(input, aug=True)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start + j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
