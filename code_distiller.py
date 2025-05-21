import math
import torch
from vector_quantization import GRVQResult


class MultiGroupDistillation(torch.nn.Module):
    def __init__(self, 
                 n_group, 
                 group_in_dim,
                 group_out_dim):
        
        super().__init__()
        
        self.n_group = n_group
        self.group_in_dim = group_in_dim
        self.group_out_dim = group_out_dim

        zero_params = torch.empty(n_group * group_in_dim, group_out_dim)
        self.need_project: bool = (group_in_dim != group_out_dim)
        self.codebook_linear_out = torch.nn.Parameter(data=zero_params) if self.need_project else torch.nn.Identity()
        self.reset_parameters()
        
        self.mse_loss = torch.nn.MSELoss()
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.codebook_linear_out, a=math.sqrt(5))
        
    def forward(self, 
                student_vqresult: GRVQResult, 
                teacher_vqresult: GRVQResult,
                beta=0.0,
                gamma=2.0):
        
        stu_quantized = student_vqresult.quantized
        tea_quantized = teacher_vqresult.quantized
        quantized_distill_loss = self.mse_loss(stu_quantized, tea_quantized)
        
        stu_quantized_fup = student_vqresult.quantized_fup
        tea_quantized_fup = teacher_vqresult.quantized_fup
        if self.need_project:
            bsz = tea_quantized_fup.shape[0]
            seq_len = tea_quantized_fup.shape[1]
            # (bsz, seq_len, n_group * group_in_dim) -> (bsz, seq_len, n_group, group_in_dim) -> (bsz, n_group, seq_len, group_in_dim)
            tea_q_view = tea_quantized_fup.view(bsz, seq_len, self.n_group, self.group_in_dim).transpose(1, 2)
            # (n_group * group_in_dim, group_out_dim) -> (n_group, group_in_dim, group_out_dim)
            clo_view = self.codebook_linear_out.view(self.n_group, self.group_in_dim, self.group_out_dim)
            # (bsz, n_group, seq_len, group_out_dim) -> (bsz, seq_len, n_group, group_out_dim) -> (bsz, seq_len, n_group * group_out_dim)
            tea_linear_out = torch.matmul(tea_q_view, clo_view).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            quantized_fup_distill_loss = self.mse_loss(stu_quantized_fup, tea_linear_out)
        else:
            quantized_fup_distill_loss = self.mse_loss(stu_quantized_fup, tea_quantized_fup)
        
        distill_loss_all = beta * quantized_distill_loss + gamma * quantized_fup_distill_loss
        
        return distill_loss_all
