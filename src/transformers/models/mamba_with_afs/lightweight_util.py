import torch
from torch import nn
from ..ibert.quant_modules import *

def asymmetric_linear_quantization_params(x, num_bits=8):
    # print(x.shape)
    x_transform = x.data.detach()
    x_min = x_transform.min().expand(1)
    x_max = x_transform.max().expand(1)
    edge = 2**(num_bits-1) - 1
    q_min, q_max = -edge, edge
    scale = (x_max - x_min) / (2 * edge)
    scale = torch.clip(scale, min=1e-5)
    zp = torch.round(q_min - (torch.min(x)) / scale)
    q_x = x / scale + zp
    q_x.clamp_(q_min, q_max).round_()
    return q_x, scale, zp

class OutlierFilter(nn.Module):
    def __init__(self, threshold, with_quan=False, num_bits=16, direct_quan=False):
        super().__init__()
        self.with_quan = with_quan
        self.num_bits = num_bits
        self.sym_quan_func = SymmetricQuantFunction.apply
        self.threshold = threshold
        self.direct_quan = direct_quan
        self.device = "cuda"
    
    def classify(self, input):
        # pdf method (deprecated)
        # total_num = input.numel()
        # outlier_num = int(total_num * 0.007)
        # flat_tensor = input.flatten()
        # sorted_tensor, _ = torch.sort(flat_tensor, descending=True)
        # threshold = sorted_tensor[-outlier_num]

        # mean = input.mean()
        # std = input.std(unbiased=False)
        # pdf_threshold = (1 / (std * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-((threshold - mean) ** 2) / (2 * std ** 2))
        # pdf = (1 / (std * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-((input - mean) ** 2) / (2 * std ** 2))
        # # print(pdf)

        # # outliers: label == 1    
        # label = torch.where(pdf < pdf_threshold, 1, 0)
        
        # iqr
        symmetried_A = torch.cat([input, -input], dim=1)
        x = symmetried_A.flatten().cpu().numpy()
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # outliers: label == 1    
        label = torch.where(input < lower_bound, 1, 0)

        return label

    def forward(self, A, discrete_time_step):
        if self.with_quan == False:
            return torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        else:
            if self.direct_quan == False:
                dt = discrete_time_step
                batch = dt.shape[0]
                intermediate_size = A.shape[0]
                ssm_state_size = A.shape[1]
                seq_len = dt.shape[-1]

                dt_transform = dt.data.detach()
                dt_min = dt_transform.min().expand(1)
                dt_max = dt_transform.max().expand(1)
                dt_scaling_factor = symmetric_linear_quantization_params(self.num_bits, dt_min, dt_max)
                dt_int = self.sym_quan_func(dt, self.num_bits, False, dt_scaling_factor)
                # compensate
                # dt_label = torch.where(dt < 0.000001, 1, 0)
                
                group_label = self.classify(A)
                result = torch.zeros(batch, intermediate_size, seq_len, ssm_state_size, dtype=torch.float).to(self.device)
                result_label = group_label[None, :, None, :].expand_as(result)
                # dt_label = dt_label[:, :, :, None].expand_as(result)
                # result_label = result_label 

                A_int = A.clone()
                A_int[group_label == 0], A_scaling_factor, Z = asymmetric_linear_quantization_params(A[group_label == 0], num_bits=self.num_bits)
                A_int[group_label == 0] = A_int[group_label == 0] - Z # (q - Z)
                A_int[group_label == 1] = -2 ** (self.num_bits - 1) # outlier encoding

                act_scaling_factor = A_scaling_factor * dt_scaling_factor

                result_mul = torch.exp(A_int[None, :, None, :] * dt_int[:, :, :, None] * act_scaling_factor)
                result[result_label == 0] = result_mul[result_label == 0]

                return result
            else:
                dt = discrete_time_step
                dt_transform = dt.data.detach()
                dt_min = dt_transform.min().expand(1)
                dt_max = dt_transform.max().expand(1)
                dt_scaling_factor = symmetric_linear_quantization_params(self.num_bits, dt_min, dt_max)
                dt_int = self.sym_quan_func(dt, self.num_bits, False, dt_scaling_factor)

                A_transform = A.data.detach()
                A_min = A_transform.min().expand(1)
                A_max = A_transform.max().expand(1)
                A_int, A_scaling_factor, Z = asymmetric_linear_quantization_params(A, num_bits=self.num_bits)
                A_int = A_int - Z

                act_scaling_factor = A_scaling_factor * dt_scaling_factor

                return torch.exp(A_int[None, :, None, :] * dt_int[:, :, :, None] * act_scaling_factor)
