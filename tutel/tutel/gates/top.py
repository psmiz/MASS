# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class LinearTopKGate(torch.nn.Module):

    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, max_expert_num=64, adaptive_experts=False, **options):
        super().__init__()
        torch.manual_seed(1)
        self.expert_num = num_global_experts

        try:
            self.wg = torch.nn.Linear(model_dim, max_expert_num, bias=False, dtype=torch.float32 if fp32_gate else None)
        except:
            self.wg = torch.nn.Linear(model_dim, max_expert_num, bias=False)
        self.fp32_gate = fp32_gate
        self.enable_softmax_logits = True
        
        # Expert mask to control which experts are active
        self.adaptive_experts = adaptive_experts
        self.max_expert_num = max_expert_num
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=False))
        self.experts_mask[:num_global_experts] = 1.0
        
        self.adaptive_top_k = True
        self.threshold = options.get('rm_threshold', 1.0)
        
        self.normalize_gate = options.get('normalize_one_score_gate', False)
        if self.normalize_gate:
            print('Gating module: Normalizing gate vectors')

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise', 'normalize_one_score_gate', 'rm_threshold'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.normalize_gate:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        if self.fp32_gate:
            x = x.float()
            wg = self.wg.float()
        else:
            wg = self.wg
        
        logits = wg(x)
        
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
        
        print('Average Top K is {}, max is {}'.format(sum(top_k) / len(top_k), max(top_k)))
        return logits, top_k

    
Gate = LinearTopKGate
