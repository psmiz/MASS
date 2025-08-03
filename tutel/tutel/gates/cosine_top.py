# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, max_expert_num=64, adaptive_experts=False, proj_dim=256, init_t=0.5, **options):
        super(CosineTopKGate, self).__init__()
        torch.manual_seed(1)
        self.top_k = 0
        self.expert_num = num_global_experts

        self.fp32_gate = fp32_gate
        self.enable_softmax_logits = False
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, max_expert_num)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)
        
        # Expert mask to control which experts are active
        self.adaptive_experts = adaptive_experts
        self.max_expert_num = max_expert_num
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=False))
        self.experts_mask[:num_global_experts] = 1.0
        
        self.adaptive_top_k = True
        self.threshold = options.get('rm_threshold', 1.0)
        
        self.avg_k = 0.0
        self.step_count = 0

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise', 'normalize_one_score_gate', 'rm_threshold'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            cosine_projector = self.cosine_projector.float()
            sim_matrix = self.sim_matrix.float()
        else:
            cosine_projector = self.cosine_projector
            sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        
        # Apply expert mask
        logits = logits.masked_fill(self.experts_mask.unsqueeze(0) == 0, -1e9)
        
        scores = F.softmax(logits, dim=-1, dtype=torch.float32) + 1e-14
        
        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        cum_scores = sorted_scores.cumsum(dim=-1)
        mask = (cum_scores - sorted_scores) < self.threshold
        top_k = mask.sum(dim=-1)
        
        # Clamp top_k to number of active experts
        active_experts = self.experts_mask.sum().int()
        top_k = torch.clamp(top_k, max=active_experts)
        
        avg_k = sum(top_k)/len(top_k)
        self.step_count += 1
        self.avg_k = (self.avg_k * (self.step_count - 1) + avg_k)/self.step_count
        
        # import pdb; pdb.set_trace()
        # print('Average Top K is {}, max is {}'.format(sum(top_k) / len(top_k), max(top_k)))
        return logits, top_k


Gate = CosineTopKGate