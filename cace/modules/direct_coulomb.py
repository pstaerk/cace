import torch
import torch.nn as nn
from typing import Dict


class CoulombPotential(nn.Module):
    def __init__(self,
                 feature_key: str = 'q',
                 output_key: str = 'coulomb_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 epsilon: float = 1e-9,
                 ):
        super().__init__()
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.compute_field = compute_field
        self.epsilon = epsilon
        self.model_outputs = [output_key]

        # Normalization factor matching Ewald module
        self.norm_factor = 1.0

        if self.compute_field:
            self.model_outputs.append(feature_key + '_field')

        self.required_derivatives = []

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)

        unique_batches = torch.unique(batch_now)

        results = []
        field_results = []

        for i in unique_batches:
            mask = batch_now == i
            r_now, q_now = r[mask], q[mask]

            pot, field = self.compute_coulomb_potential(r_now, q_now, self.compute_field)

            results.append(pot)
            field_results.append(field)

        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key + '_field'] = torch.cat(field_results, dim=0)

        return data

    def compute_coulomb_potential(self, r, q, compute_field=False):
        """
        Compute direct Coulomb sum for one configuration.

        Args:
            r: positions [N, 3]
            q: charges [N, n_q]
            compute_field: whether to compute electric field
            
        Returns:
            pot: potential energy [n_q]
            field: electric field [N, n_q]
        """
        N = r.shape[0]
        device = r.device
        
        # Compute pairwise distance matrix [N, N, 1]
        distances_matrix = torch.norm(
            r.unsqueeze(1) - r.unsqueeze(0), dim=-1
        ).unsqueeze(-1) + self.epsilon
        
        # Compute pairwise potentials [N, N, n_q]
        potentials = q.unsqueeze(1) * q.unsqueeze(0) / distances_matrix
        
        # Remove self-interaction
        mask = (1 - torch.eye(N, device=device)).unsqueeze(-1)
        potentials = mask * potentials
        
        # Sum over pairs and multiply by 0.5 to avoid double counting
        pot = potentials.sum(dim=[0, 1]) * 0.5 * self.norm_factor
        
        # Compute electric field if requested
        q_field = torch.zeros_like(q, dtype=q.dtype, device=device)
        if compute_field:
            # Field contribution from all other charges
            # Remove self-interaction with mask
            q_field = (q.unsqueeze(1) / distances_matrix * mask).sum(dim=0) * self.norm_factor
        
        return pot, q_field
