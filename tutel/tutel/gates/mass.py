# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class MASSGate(torch.nn.Module):
    """
    MASS-enabled Adaptive Gate with expert mask and dynamic expansion capabilities.
    Implements adaptive selection based on cumulative routing mass threshold.
    """
    
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, max_expert_num=None, **options):
        super().__init__()
        self.max_expert_num = max_expert_num or (num_global_experts * 2)
        self.initial_experts = num_global_experts
        
        try:
            self.wg = torch.nn.Linear(model_dim, self.max_expert_num, bias=False, 
                                    dtype=torch.float32 if fp32_gate else None)
        except:
            self.wg = torch.nn.Linear(model_dim, self.max_expert_num, bias=False)
        
        # Expert mask to control which experts are active
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(self.max_expert_num,)), requires_grad=False))
        self.experts_mask[:num_global_experts] = 1.0
        
        self.enable_softmax_logits = False
        self.threshold = options.get('rm_threshold', 1.0)

        self.avg_k = 0.0
        self.step_count = 0
        
        self.top_k = 0
        self.gate_noise = 0.0 
        self.fp32_gate = fp32_gate
        self.current_experts = num_global_experts
        self.capacity_factor = 0.0 
        self.adaptive_top_k = True
        
        print(f"MASS Adaptive Gate initialized: {num_global_experts} initial experts, max {self.max_expert_num}, threshold {self.threshold}")

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            wg = self.wg.float()
        else:
            wg = self.wg
        
        scores = torch.matmul(x, wg.weight.T)
        scores = scores.masked_fill(self.experts_mask.unsqueeze(0) == 0, -1e9)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32) + 1e-14

        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        cum_scores = sorted_scores.cumsum(dim=-1)
        mask = (cum_scores - sorted_scores) < self.threshold
        
        top_k = mask.sum(dim=-1)  
        active_experts = self.experts_mask.sum().int()
        top_k = torch.clamp(top_k, max=active_experts)
        
        avg_k = sum(top_k)/len(top_k)
        self.step_count += 1
        self.avg_k = (self.avg_k * (self.step_count - 1) + avg_k)/self.step_count
        
        # print('Average Top K is {}, max is {}'.format(sum(top_k) / len(top_k), max(top_k)))
        return scores.type_as(x), top_k

Gate = MASSGate
