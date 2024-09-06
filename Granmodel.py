import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
import numpy as np
from deform_conv import DeformConv2D, depthwise_separable_conv
import math
import matmult
import torch.linalg as LA


class FrozenBatchNorm2d(torch.nn.Module):
    """`
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class GRAN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GRAN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Granular_Transformer(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Granular_Transformer(nn.Module):
    def __init__(self, args, channels=256):
        super(Granular_Transformer, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.ra1 = GranularAttention(args, channels)
        self.ra2 = GranularAttention(args, channels)
        self.ra3 = GranularAttention(args, channels)
        self.ra4 = GranularAttention(args, channels)

    def forward(self, x):

        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.ra1(x)
        x2 = self.ra2(x1)
        x3 = self.ra3(x2)
        x4 = self.ra4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

def Granulation(q, k):
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)


class GranularAttention(nn.Module):
    def __init__(self, args, channels):
        super(GranularAttention, self).__init__()
        self.args = args
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.head_dim = channels // 8
        self.v_linear = nn.Linear(self.head_dim, 1, bias=False)

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.before_norm = nn.BatchNorm1d(channels // 4)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.offsets = depthwise_separable_conv(self.head_dim, 32, kernel_size=3, stride=1, padding=1)
        self.deformconv = DeformConv2D(self.head_dim, self.head_dim, kernel_size=4, stride=8, padding=1)
        self.deform_norm = nn.Sequential(nn.BatchNorm2d(self.head_dim), nn.ReLU(inplace=True))
        self.conv_norm = nn.Sequential(nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.norm = nn.Sequential(nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.q_norm = nn.Sequential(nn.BatchNorm1d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        batch_size, q_len, input_channels = x.size()
        x_q = self.v_conv(x)  # .permute(0, 2, 1)
        x_k = self.v_conv(x)
        x_v = self.v_conv(x).reshape(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        x_d = x.reshape(batch_size, -1, self.head_dim, q_len).reshape(batch_size*8, self.head_dim, -1, 16)
        q = x_q.reshape(batch_size, -1, 8, self.head_dim).transpose(1, 2)
        N, hide_dim, height, weight = x_d.size()

        offsets = self.offsets(x_d)
        q_deformconv = self.deformconv(x_d, offsets)
        q_deformconv = self.deform_norm(q_deformconv)
        q_deformconv = q_deformconv.flatten(2).reshape(batch_size, int(N/batch_size), hide_dim, -1).transpose(-1, -2).contiguous()

        x_r = _prob_QK(q, q_deformconv, x_v, self.args.max_iter)

        # 计算注意力权重
        x_r = x_r / torch.sqrt(torch.tensor(self.head_dim).float())

        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def _prob_QK(query, q_deformconv, v, max_iter):  # (Q, K, v, sample_k, n_top):

    k_deformconv = q_deformconv
    kernel_1 = calMat(query, k_deformconv.transpose(-1, -2).contiguous())
    kernel_1 = torch.exp(-kernel_1 / 3.6)  # 1 / (1 + torch.exp(-0.2 * kernel_1))  # torch.sqrt(kernel_1 + 0.1**2)  # torch.sqrt(kernel_1)
    kernel_2 = calMat(q_deformconv, k_deformconv.transpose(-1, -2).contiguous())
    kernel_2 = torch.exp(-kernel_2 / 3.6)  # 1 / (1 + torch.exp(-0.2 * kernel_2))  # torch.sqrt(kernel_2 + 0.1**2)  # torch.sqrt(kernel_2)
    kernel_3 = kernel_1.transpose(-1, -2)

    Q_K = torch.matmul(torch.matmul(kernel_1, kerton_inv(kernel_2, max_iter)), torch.matmul(kernel_3, v))
    return Q_K.transpose(1, 2).flatten(2)  # , index_sample  QK_top


#该方法的作用其实是返回一个扩张比例值（即占长宽比例）
def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def kernel_function(q, k):
    matA_square = q @ torch.ones(k.shape[-2:]).cuda()
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k
    return matA_square - matB_square


def calMat(x, y):
    batch_size, num_head, q_len, input_channels = x.size()
    batch_size, num_head, input_channels, k_len = y.size()
    c = x.new(batch_size, num_head, q_len, k_len)

    matmult.torch_launch_matmult(x, y, c, batch_size, num_head, q_len, k_len, input_channels)
    return c


def metricFrob(I, X, A):
    N_X = torch.matmul(torch.matmul(A, X), A) - A
    FrobX = torch.max(LA.matrix_norm(N_X, ord=2))
    FrobA = torch.max(LA.matrix_norm(A, ord=2))
    metric = FrobX / FrobA
    return metric, FrobX


def kerton_inv(kernel2, max_iter=12):

    Frob = LA.matrix_norm(kernel2, ord='fro') 
    I = torch.eye(kernel2.size(-1), device=kernel2.device)
    I = torch.from_numpy(np.broadcast_to(I[None, None, ...].cpu().detach().numpy(), shape=kernel2.shape)).to(device=kernel2.device)
    A = kernel2  # + 0.01 * I
    Lambda = torch.from_numpy(np.broadcast_to(np.float32(0.2 / Frob.cpu().detach().numpy())[..., None, None], shape=kernel2.shape)).to(device=kernel2.device)
    X = Lambda

    sigma = 0.2
    E = I - torch.matmul(A, X)
    init_matNum = torch.max(LA.matrix_norm(E, ord=2))
    err_cnt = 0
    while init_matNum > 1.00001 and err_cnt < 20:
        Lambda *= sigma
        X = Lambda  # * A
        initX = I - torch.matmul(A, X)
        init_matNum = torch.max(LA.matrix_norm(initX, ord=2))
        err_cnt += 1

    for i in range(max_iter):
        X = X @ (3 * I - 3 * A @ X + (A @ X) ** 2)
    return X
