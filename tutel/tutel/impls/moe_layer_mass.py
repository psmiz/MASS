# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
import math
from collections import deque
from ..impls.moe_layer import MOELayer


class MoELayerMASS(MOELayer):
    """
    MASS-enabled MoE Layer that extends the base MOELayer with 
    Change Point Detection for dynamic expert expansion.
    """
    def __init__(self, *args, **kwargs):
        self.mass_config = kwargs.pop('mass_config', {})
        super().__init__(*args, **kwargs)
        
        # MASS configuration with defaults
        self.t_warmup = self.mass_config.get('warmup_steps', 50)
        self.window_size = self.mass_config.get('window_size', 200)
        self.p_limit = self.mass_config.get('p_threshold', 0.01)
        self.sim_thresh = self.mass_config.get('similarity_threshold', 0.001)
        self.expansion_patience = self.mass_config.get('expansion_patience', 3)
        self.enable_mass = self.mass_config.get('enable_mass', True)

        self.stop_expansion = False
        self.duplicated_pairs = []
        self.expansion_count = 0
        
        # self._init_mass_tracking()
        # self._register_gradient_hooks()
        
        self.mass_trace = {
            "mass_events": [],  # (step, expert_id, expansion_flag)
            "expansion_history": [],  # (step, source_expert_idx, new_expert_idx)
            "topk_history": []  # track average top-k over time
        }
        
    def _init_mass_tracking(self):
        """Initialize MASS tracking structures for all experts"""
        # self.grads_track = [[] for _ in range(self.num_global_experts)]
        self.grad_count = [0 for _ in range(self.num_global_experts)]
        self.grad_window = [deque(maxlen=self.window_size) for _ in range(self.num_global_experts)]
        self.z_window = [deque(maxlen=self.window_size) for _ in range(self.num_global_experts)]
        
        self._grad_hooks = []
        
    def _register_gradient_hooks(self):
        """Register gradient hooks on expert parameters"""
        if not self.enable_mass:
            return
        
        # Register hook on batched expert weights
        if hasattr(self.experts, 'batched_fc1_w'):
            def batched_hook(grad):
                if grad is not None and self.enable_mass:
                    # grad shape: [num_local_experts, hidden_size, model_dim]
                    for i in range(self.num_global_experts):
                        grad_norm = grad[i].norm().item()
                        self._update_mass_statistics(i, grad_norm)
            
            handle = self.experts.batched_fc1_w.register_hook(batched_hook)
            self._grad_hooks.append(handle)
            print(f"MASS: Registered gradient hook on batched_fc1_w")
        else:
            print("Warning: Could not find batched_fc1_w parameter for gradient hooking")
                
    def _remove_gradient_hooks(self):
        """Remove all gradient hooks"""
        for handle in self._grad_hooks:
            if handle is not None:
                handle.remove()
        self._grad_hooks.clear()
        
    def _reset_mass_statistics(self, expert_idx):
        """Reset MASS statistics"""
        self.grad_count[expert_idx] = 0

    def _remove_expert(self):
        """Remove an expert from the model"""
        self.expansion_count -= 1
        self._num_global_experts.data = torch.tensor(self.num_global_experts - 1)
        self.duplicated_pairs.pop()
        self.grad_count.pop()
        self.grad_window.pop()
        self.z_window.pop()
        print(f"MASS: Removed {self.num_global_experts+1}-th expert")

    def _update_mass_statistics(self, expert_idx, grad_norm):
        """Update MASS statistics for a specific expert"""
        if expert_idx >= len(self.grad_count):
            return
            
        self.grad_count[expert_idx] += 1
        self.grad_window[expert_idx].append(grad_norm)
        
        # Only start MASS computation after warmup period
        if self.grad_count[expert_idx] < self.t_warmup:
            return
            
        # Compute z-score for change point detection
        window_data = torch.tensor(list(self.grad_window[expert_idx]), device=self.experts.batched_fc1_w.device)
        mean_val = torch.mean(window_data)
        std_val = torch.std(window_data, unbiased=True)
        z_score = (grad_norm - mean_val) / std_val + 1e-12
        self.z_window[expert_idx].append(z_score.item())
        
    def _compute_p_value(self, expert_idx):
        """Compute p-value for change point detection"""
        s_tilde = sum(self.z_window[expert_idx]) / math.sqrt(self.window_size)
        dist = torch.distributions.Normal(0, 1)
        return 1 - dist.cdf(torch.tensor(s_tilde))

    def _compute_weight_grad_similarity(self, expert_idx):
        """Compute cosine similarity between expert weights and gradients"""
        try:
            if hasattr(self.experts, 'batched_fc1_w'):
                if expert_idx < self.experts.batched_fc1_w.size(0):
                    weight = self.experts.batched_fc1_w[expert_idx]
                    if self.experts.batched_fc1_w.grad is not None and expert_idx < self.experts.batched_fc1_w.grad.size(0):
                        grad = self.experts.batched_fc1_w.grad[expert_idx]
                        return F.cosine_similarity(weight.view(1, -1), grad.view(1, -1)).item()
            return 0.0
        except:
            return 0.0
            
    def _find_inactive_expert(self):
        """
        Find the next inactive expert slot.
        Returns the index of an inactive expert, or None if all are active.
        """
        gate = self.gates[0]
        if hasattr(gate, 'experts_mask'):
            inactive_experts = torch.where(gate.experts_mask == 0)[0]
            if len(inactive_experts) > 0:
                return inactive_experts[0].item()
        return None
        
    def _duplicate_experts_and_gate_weights(self, source_expert_idx, target_expert_idx):
        """
        Copy source expert parameters to target expert slot.
        Both experts already exist in the batched tensors.
        """
        try:
            with torch.no_grad():
                self.experts.batched_fc1_w[target_expert_idx].copy_(self.experts.batched_fc1_w[source_expert_idx])
                self.experts.batched_fc1_bias[target_expert_idx].copy_(self.experts.batched_fc1_bias[source_expert_idx])
                if hasattr(self.experts, 'batched_fc2_w'):
                    self.experts.batched_fc2_w[target_expert_idx].copy_(self.experts.batched_fc2_w[source_expert_idx])
                    self.experts.batched_fc2_bias[target_expert_idx].copy_(self.experts.batched_fc2_bias[source_expert_idx])

                gate = self.gates[0]
                if hasattr(gate, 'wg'):
                    gate.wg.weight[target_expert_idx].copy_(gate.wg.weight[source_expert_idx])
                gate.experts_mask[target_expert_idx] = 1.0
        
        except Exception as e:
            print(f"MASS: Error in copying experts and gate weights: {e}")
            
    def _edit_grad_align(self, source_expert_idx, new_expert_idx):
        """
        Apply gradient alignment to encourage expert divergence.
        Based on projection alignment from MASS AutoK.
        """
        try:
            if not hasattr(self.experts, 'batched_fc1_w') or self.experts.batched_fc1_w.grad is None:
                return

            with torch.no_grad():
                # Gradient Decomposition for Expert
                w = self.experts.batched_fc1_w[source_expert_idx].detach()
                g_old = self.experts.batched_fc1_w.grad[source_expert_idx].detach()
                
                dot_product = torch.sum(g_old * w, dim=1) 
                w_norm = torch.sum(w * w, dim=1) + 1e-12 
                proj_g = (dot_product.unsqueeze(1) / w_norm.unsqueeze(1)) * w  # shape: [hidden_size, model_dim]

                self.experts.batched_fc1_w.grad[source_expert_idx] = proj_g
                if new_expert_idx < self.experts.batched_fc1_w.grad.size(0):
                    self.experts.batched_fc1_w.grad[new_expert_idx] = g_old.clone()

                # Gradient Decomposition for Gate
                gate = self.gates[0]
                if hasattr(gate, 'wg') and gate.wg.weight.grad is not None:
                    wg = gate.wg.weight[source_expert_idx].detach()
                    gg_old = gate.wg.weight.grad[source_expert_idx].detach()
                    proj_gg = (torch.dot(gg_old, wg) / (torch.dot(wg, wg) + 1e-12)) * wg
                    gate.wg.weight.grad[source_expert_idx] = proj_gg
                    gate.wg.weight.grad[new_expert_idx] = gg_old.clone()

        except Exception as e:
            print(f"MASS: Error in gradient alignment: {e}")
            
    def _extend_mass_tracking(self, new_expert_idx):
        """Extend MASS tracking structures for new expert"""
        self.grad_count.append(0)
        self.grad_window.append(deque(maxlen=self.window_size))
        self.z_window.append(deque(maxlen=self.window_size))
            
    def _redundancy_regularization(self):
        """
        Compute redundancy regularization term to encourage expert divergence.
        Based on the MASS AutoK implementation.
        """
        if not self.duplicated_pairs:
            return torch.tensor(0.0, device=self.experts.batched_fc1_w.device)
        
        gate = self.gates[0]
        redundancy = torch.tensor(0.0, device=self.experts.batched_fc1_w.device)
        for source_idx, target_idx in self.duplicated_pairs:
            if hasattr(gate, 'wg'):
                w1 = F.normalize(gate.wg.weight[source_idx], dim=0)
                w2 = F.normalize(gate.wg.weight[target_idx], dim=0)
            similarity = torch.dot(w1, w2)
            redundancy += similarity ** 2
        
        return redundancy

    def get_redundancy_loss(self):
        return self._redundancy_regularization()
        
    def get_mass_info(self):
        """Return MASS statistics and state"""
        active_experts = self.num_global_experts
        
        # Calculate average top-k from recent history
        return {
            'expansion_count': self.expansion_count,
            'active_experts': active_experts,
            'max_experts': self.max_num_global_experts,
            'duplicated_pairs': self.duplicated_pairs,
            'stop_expansion': self.stop_expansion,
            'enable_mass': self.enable_mass,
            'mass_trace': self.mass_trace,
            'grad_counts': self.grad_count[:active_experts],
        }
        
    def disable_mass_tracking(self):
        """Disable MASS tracking"""
        self.enable_mass = False
        self._remove_gradient_hooks()
        print("MASS: Tracking disabled")

    def _expand_expert(self, source_expert_idx, target_expert_idx, step):
        """
        Expand experts by copying source expert to an inactive expert slot.
        All expert parameters already exist, we just copy weights and activate.
        """
        self._duplicate_experts_and_gate_weights(source_expert_idx, target_expert_idx)
        
        self._num_global_experts.data = torch.tensor(self.num_global_experts + 1)
        self.duplicated_pairs.append((source_expert_idx, target_expert_idx))
        self.expansion_count += 1
        
        self._reset_mass_statistics(source_expert_idx)
        self._extend_mass_tracking(target_expert_idx)
        self.mass_trace["expansion_history"].append((step, source_expert_idx, target_expert_idx))
            
    def expand_expert(self, info, step):
        """
        Check for expansion conditions and expand if necessary.
        This should be called during training loop after backward pass.
        """
        source_expert_idx, p_val, cos_sim = info
        target_expert_idx = self._find_inactive_expert()

        self._expand_expert(source_expert_idx, target_expert_idx, step)
        print(f"MASS expansion triggered: expert {source_expert_idx} -> expert {target_expert_idx}, p={p_val:.4f}, cos_sim={cos_sim:.4f}")
        self._edit_grad_align(source_expert_idx, target_expert_idx)
        return True

    def _check_expansion_conditions(self, step):
        """Check if any expert meets expansion conditions"""
        if self.stop_expansion or not self.enable_mass:
            return None
        
        for expert_idx in range(self.num_global_experts):
            if expert_idx >= len(self.grad_count) \
                or self.grad_count[expert_idx] < self.t_warmup \
                    or len(self.grad_window[expert_idx]) < self.window_size:
                continue
            
            p_val = self._compute_p_value(expert_idx)
            if p_val < self.p_limit:
                cos_sim = self._compute_weight_grad_similarity(expert_idx)
                
                if abs(cos_sim) < self.sim_thresh:
                    self.mass_trace["mass_events"].append((step, expert_idx, 1))
                    return (expert_idx, p_val, cos_sim)
                else:
                    self.mass_trace["mass_events"].append((step, expert_idx, 0))
                    
        return None
        

moe_layer_mass = MoELayerMASS