import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
from RST import _C
import numpy as np
from deform_conv import DeformConv2D, depthwise_separable_conv
import math
import matmult
from deformable_model.ops.modules import MSDeformAttn


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

class RST(nn.Module):
    def __init__(self, args, output_channels=40):
        super(RST, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Rough_Set_Transformer(args)

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

class Rough_Set_Transformer(nn.Module):
    def __init__(self, args, channels=256):
        super(Rough_Set_Transformer, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.ra1 = RoughsetAttention(args, channels)
        self.ra2 = RoughsetAttention(args, channels)
        self.ra3 = RoughsetAttention(args, channels)
        self.ra4 = RoughsetAttention(args, channels)

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


class RoughsetAttention(nn.Module):
    def __init__(self, args, channels):
        super(RoughsetAttention, self).__init__()
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

        # self attention
        self.self_attn = MSDeformAttn(channels, 1, 8, 4)

    def forward(self, x):
        batch_size, q_len, input_channels = x.size()
        x_q = self.v_conv(x)  # .permute(0, 2, 1)
        x_k = self.v_conv(x)
        x_v = self.v_conv(x).reshape(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        # q = x.reshape(batch_size, -1, q_len, self.head_dim)  # F.softmax(x, dim=-1)  # self.act(self.after_norm(x))
        x_d = x.reshape(batch_size, -1, self.head_dim, q_len).reshape(batch_size*8, self.head_dim, -1, 16)
        # q = self.deform_norm(x_d).flatten(2).transpose(-1, -2).reshape(batch_size, -1, q_len, self.head_dim)
        # q = x_q.reshape(batch_size, -1, 8, self.head_dim).transpose(1, 2)
        q = x.transpose(-1, -2).reshape(batch_size, input_channels, -1, 16)
        N, hide_dim, height, weight = x_d.size()

        # batch_size,  q_len, input_channels = x_q.size()
        # batch_size,  input_channels, k_len = x_k.size()
        # output = x_q.new(batch_size, 1, q_len, k_len)

        # offsets = self.offsets(x_d)
        # q_deformconv = self.deformconv(x_d, offsets)
        # q_deformconv = self.deform_norm(q_deformconv)
        # q_deformconv = q_deformconv.flatten(2).reshape(batch_size, int(N/batch_size), hide_dim, -1)
        # q_deformconv = q_deformconv.flatten(2).reshape(batch_size, int(N/batch_size), hide_dim, -1).transpose(-1, -2).contiguous()
        # q_deformconv = self.deform_norm(q_deformconv)
        # q_deformconv = x_q.reshape(batch_size, -1, 8, self.head_dim).transpose(1, 2)
        # q_deformconv = self.conv_norm(q_deformconv)
        # k_deformconv = x_k.reshape(batch_size, -1, 8, self.head_dim).transpose(1, 2)
        # x_r = _prob_QK(q, q_deformconv, x_v, self.args.max_iter)
        # x_r = self.softmax(x_r)
        x_r = deform_attention(q, self.self_attn)

        # kernel = calMat(q_deformconv, k_deformconv.transpose(-1, -2))
        # kernel = F.softmax(kernel, dim=-1)
        # kernel = calMat(q_deformconv.transpose(-1, -2).contiguous(), q_deformconv)
        # kernel = torch.exp(-kernel / 8)
        # kernel = F.softmax(kernel, dim=-1) # torch.sqrt(kernel)  # torch.exp(-kernel / 2)  # 1 / (1 + torch.exp(-0.2 * kernel))
        # kernel = kernel.reshape(batch_size, 1, q_len, k_len)
        #
        # _C.Approximation(kernel, kernel, output)
        # output = output.reshape(batch_size, q_len, k_len)  # output.reshape(batch_size, q_len, k_len)
        #
        # x_r = torch.matmul(kernel, x_v.transpose(-1, -2)).transpose(-1, -3).flatten(2).contiguous()
        # x_r = torch.matmul(kernel, x_v).transpose(-2, -3).flatten(2).contiguous()

        # 计算注意力权重
        # attention_weights = torch.sum(q_deformconv + k_deformconv, dim=-2)
        # attention_weights = attention_weights / torch.sqrt(torch.tensor(self.head_dim).float())

        # 对注意力权重进行softmax归一化
        # attention_weights = self.softmax(attention_weights)

        # 加权求和得到注意力输出
        # x_r = (attention_weights.unsqueeze(-2) * x_v).transpose(-2, -3).flatten(2).contiguous()  # torch.sum(attention_weights.unsqueeze(-1) * x_v, dim=-2)

        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def _prob_QK(query, q_deformconv, v, max_iter):  # (Q, K, v, sample_k, n_top):

    k_deformconv = q_deformconv
    kernel_1 = torch.matmul(query, k_deformconv.transpose(-1, -2).contiguous())
    # kernel_1 = torch.exp(-kernel_1 / 8)  # 1 / (1 + torch.exp(-0.2 * kernel_1))  # torch.sqrt(kernel_1 + 0.1**2)  # torch.sqrt(kernel_1)
    kernel_1 = F.softmax(kernel_1, dim=-1)
    kernel_2 = torch.matmul(q_deformconv, k_deformconv.transpose(-1, -2).contiguous())
    # kernel_2 = torch.exp(-kernel_2 / 8)  # 1 / (1 + torch.exp(-0.2 * kernel_2))  # torch.sqrt(kernel_2 + 0.1**2)  # torch.sqrt(kernel_2)
    kernel_2 = F.softmax(kernel_2, dim=-1)
    # kernel_1 = F.dropout(kernel_1, p=dropout_p, training=training)8
    # kernel_3 = kernel_1.transpose(-1, -2)
    kernel_3 = kernel_1.transpose(-1, -2)

    Q_K = torch.matmul(torch.matmul(kernel_1, newton_inv(kernel_2, max_iter)), torch.matmul(kernel_3, v))
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


def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def deform_attention(query, attn):  # (Q, K, v, sample_k, n_top):
    mask_flatten = []
    spatial_shapes = []

    bs, c, h, w = query.shape
    mask = torch.full((bs, h, w), False, device=query.device)
    spatial_shape = (h, w)
    spatial_shapes.append(spatial_shape)
    src_flatten = query.flatten(2).transpose(-1, -2)

    mask_flatten = mask.flatten(1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.stack([get_valid_ratio(mask)], 1)

    # 生成参考点坐标
    reference_points = get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
    attn_value = attn(src_flatten, reference_points, src_flatten, spatial_shapes, level_start_index, mask_flatten)

    return attn_value  # , index_sample  QK_top Q_K.transpose(1, 2).flatten(2)


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


def kerton_inv(kernel2, max_iter=5):
    bs, n_head, m_len, nd = kernel2.shape
    matNum = kernel2.reshape(bs * n_head, m_len * nd).cpu().detach().numpy()
    # bs, m_len, nd = kernel2.shape
    # matNum = kernel2.reshape(bs, m_len * nd).cpu().detach().numpy()
    Frob = np.linalg.norm(matNum, ord='fro')
    I = torch.eye(kernel2.size(-1), device=kernel2.device)
    A = kernel2 + 0.01 * I
    X = torch.from_numpy(np.broadcast_to(np.float32(2 / Frob)[..., None, None], shape=kernel2.shape)).to(device=kernel2.device)

    for i in range(max_iter):
        X = X * (3 * I - 3 * A @ X + (A @ X) ** 2)
    return X


def newton_inv(mat, iter=6):
    P = mat
    I = torch.eye(mat.size(-1), device=mat.device)
    alpha = 0.9 * 2 / torch.max(torch.sum(mat, dim=-1), -1)[0] ** 2
    P = P + 0.01 * I
    V = torch.from_numpy(np.broadcast_to(alpha[..., None, None].cpu().detach().numpy(), shape=P.shape)).to(device=mat.device) * P

    for i in range(iter):
        V = 2 * V - V @ P @ V
    return V


def find_matroid_basis(matrix):
    # 将矩阵转换为简化行阶梯形矩阵
    row_echelon_matrix = np.array(matrix.cpu(), dtype=float)
    m, n = row_echelon_matrix.shape
    for i in range(min(m, n)):
        # 找到第i列中第i行以下的最大值
        max_element = abs(row_echelon_matrix[i, i])
        max_row = i
        for j in range(i+1, m):
            if abs(row_echelon_matrix[j, i]) > max_element:
                max_element = abs(row_echelon_matrix[j, i])
                max_row = j
        # 如果该列全为0，则继续下一列
        if max_element == 0:
            continue
        # 将具有最大值的行与第i行交换
        row_echelon_matrix[[i, max_row]] = row_echelon_matrix[[max_row, i]]
        # 将主元归一化为1
        row_echelon_matrix[i] /= row_echelon_matrix[i, i]
        # 将主元所在列的其他元素消为0
        for j in range(m):
            if j != i:
                row_echelon_matrix[j] -= row_echelon_matrix[i] * row_echelon_matrix[j, i]
    # 找到主元所在的列，即拟阵基中向量的索引
    basis_indices = np.where(np.abs(row_echelon_matrix).sum(axis=0) == 1)[0]
    # 返回拟阵基中的向量
    return matrix[:, basis_indices]


if __name__ == '__main__':
    # 测试代码
    # matrix = np.array([[1, 2, 3, 4, 5],
    #                   [0, 1, 2, 3, 4],
    #                   [0, 0, 1, 2, 3],
    #                   [2, 4, 6, 8, 10],
    #                   [1, 0, 0, 0, 1]])
    matrix = np.random.randint(0, 10, (150, 100))
    basis = find_matroid_basis(matrix)
    print(basis)
    print(matrix)
    print(basis.shape)