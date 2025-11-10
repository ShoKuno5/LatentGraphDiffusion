"""
ODE solvers for Latent Flow Matching.
Implements Euler and Heun (RK2) methods for solving the flow ODE.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from contextlib import contextmanager
import copy
from torch_geometric.data import Batch
from lgd.model.utils import num2batch, symmetrize
from torch_geometric.graphgym.config import cfg


def solve_flow(v_theta, z0, steps=20, method="heun", batch=None, verbose=True):
    """
    Solve the flow ODE from t=0 to t=1.
    
    Args:
        v_theta: Velocity field model (callable)
        z0: Initial latent (noise) - can be tuple of (nodes, edges, graph)
        steps: Number of integration steps
        method: "euler" or "heun" (RK2)
        batch: PyG batch object with graph structure info
        verbose: Show progress bar
    
    Returns:
        z1: Final latent at t=1
    """
    device = z0.device if torch.is_tensor(z0) else z0[0].device
    dt = 1.0 / steps
    t = torch.zeros(1, device=device)
    
    # Handle both single tensor and tuple inputs
    if isinstance(z0, (tuple, list)):
        z_nodes, z_edges = z0[0], z0[1]
        z_graph = z0[2] if len(z0) > 2 else None
    else:
        # Assume concatenated format like in LGD
        num_nodes = batch.num_nodes if batch is not None else z0.shape[0]
        z_nodes = z0[:num_nodes]
        z_edges = z0[num_nodes:]
        z_graph = None
    
    # Progress bar
    iterator = tqdm(range(steps), desc='Solving flow ODE', disable=not verbose)
    
    for step in iterator:
        # Current time (broadcast to batch size)
        if batch is not None:
            t_batch = t.expand(batch.num_graphs).to(device)
        else:
            t_batch = t.to(device)
        
        if method == "euler":
            # Euler method: z_{n+1} = z_n + dt * v(z_n, t_n)
            v_nodes, v_edges, v_graph = v_theta(z_nodes, z_edges, z_graph, t_batch, batch)
            z_nodes = z_nodes + dt * v_nodes
            z_edges = z_edges + dt * v_edges
            if z_graph is not None and v_graph is not None:
                z_graph = z_graph + dt * v_graph
                
        elif method == "heun":
            # Heun's method (RK2): 
            # k1 = v(z_n, t_n)
            # z_mid = z_n + dt * k1
            # k2 = v(z_mid, t_n + dt)
            # z_{n+1} = z_n + 0.5 * dt * (k1 + k2)
            
            # First stage
            k1_nodes, k1_edges, k1_graph = v_theta(z_nodes, z_edges, z_graph, t_batch, batch)
            
            # Midpoint
            z_mid_nodes = z_nodes + dt * k1_nodes
            z_mid_edges = z_edges + dt * k1_edges
            z_mid_graph = z_graph + dt * k1_graph if z_graph is not None else None
            
            # Second stage
            t_mid = (t + dt).expand(batch.num_graphs).to(device) if batch is not None else (t + dt).to(device)
            k2_nodes, k2_edges, k2_graph = v_theta(z_mid_nodes, z_mid_edges, z_mid_graph, t_mid, batch)
            
            # Update
            z_nodes = z_nodes + 0.5 * dt * (k1_nodes + k2_nodes)
            z_edges = z_edges + 0.5 * dt * (k1_edges + k2_edges)
            if z_graph is not None and k1_graph is not None:
                z_graph = z_graph + 0.5 * dt * (k1_graph + k2_graph)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Update time
        t = t + dt
    
    # Return in same format as input
    if isinstance(z0, (tuple, list)):
        return (z_nodes, z_edges, z_graph) if z_graph is not None else (z_nodes, z_edges)
    else:
        return torch.cat([z_nodes, z_edges], dim=0)


class FlowSampler(nn.Module):
    """
    Flow sampler for generation with various ODE solvers.
    Mirrors the DDIMSampler interface from LGD.
    """
    
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.schedule = schedule
        
    @torch.no_grad()
    def sample(self, batch, steps, batch_size, shape, cond=None, 
               verbose=False, eta=0., method="heun", 
               temperature=1.0, return_intermediates=False, **kwargs):
        """
        Sample from the flow model.
        
        Args:
            batch: PyG batch object with graph structure
            steps: Number of ODE integration steps
            batch_size: Batch size (unused, kept for compatibility)
            shape: Shape of latent tensors
            cond: Conditioning information (for conditional generation)
            verbose: Show progress
            eta: Unused (kept for DDIM compatibility)
            method: ODE solver method ("euler" or "heun")
            temperature: Noise temperature for initial sampling
            return_intermediates: Whether to return intermediate states
        
        Returns:
            samples: Generated latents at t=1
        """
        device = self.model.device if hasattr(self.model, 'device') else batch.x.device
        
        # Sample initial noise z0
        if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[0], tuple):
            # New format: tuple of (shape_nodes, shape_edges) 
            shape_nodes, shape_edges = shape
            z0_nodes = torch.randn(shape_nodes, device=device) * temperature
            z0_edges = torch.randn(shape_edges, device=device) * temperature
            z0_graph = None
            if hasattr(batch, 'num_graphs') and self.model.use_graph_latent:
                z0_graph = torch.randn((batch.num_graphs, shape_nodes[-1]), device=device) * temperature
            z0 = (z0_nodes, z0_edges, z0_graph) if z0_graph is not None else (z0_nodes, z0_edges)
        elif isinstance(shape, (tuple, list)) and len(shape) == 2:
            # Legacy format: single shape provided
            z0 = torch.randn(shape, device=device) * temperature
        else:
            # Fallback: separate shapes for nodes/edges/graph
            z0_nodes = torch.randn((batch.num_nodes, shape[-1]), device=device) * temperature
            num_edges = batch.edge_index.shape[1]
            z0_edges = torch.randn((num_edges, shape[-1]), device=device) * temperature
            z0_graph = None
            if hasattr(batch, 'num_graphs') and self.model.use_graph_latent:
                z0_graph = torch.randn((batch.num_graphs, shape[-1]), device=device) * temperature
            z0 = (z0_nodes, z0_edges, z0_graph) if z0_graph is not None else (z0_nodes, z0_edges)

        if getattr(self.model, 'objective', None) == "gaussian_cfm":
            dtype = z0[0].dtype if isinstance(z0, tuple) else z0.dtype
            t0_graph = torch.zeros(batch.num_graphs, device=device, dtype=dtype)
            node_sigma = self.model.get_alpha_sigma(t0_graph[batch.batch].unsqueeze(-1))[1]
            edge_batch = batch.batch[batch.edge_index[0]]
            edge_sigma = self.model.get_alpha_sigma(t0_graph[edge_batch].unsqueeze(-1))[1]
            graph_sigma = self.model.get_alpha_sigma(t0_graph.unsqueeze(-1))[1]

            if isinstance(z0, tuple):
                z_nodes, z_edges, *maybe_graph = z0
                z_nodes = z_nodes * node_sigma
                z_edges = z_edges * edge_sigma
                if maybe_graph:
                    z_graph = maybe_graph[0]
                    if z_graph is not None:
                        z_graph = z_graph * graph_sigma
                    z0 = (z_nodes, z_edges, z_graph)
                else:
                    z0 = (z_nodes, z_edges)
            else:
                num_nodes = batch.num_nodes
                num_edges = batch.edge_index.shape[1]
                z_nodes = z0[:num_nodes] * node_sigma
                z_edges = z0[num_nodes:num_nodes + num_edges] * edge_sigma
                if self.model.use_graph_latent:
                    z_graph_flat = z0[num_nodes + num_edges:]
                    if z_graph_flat.numel() > 0:
                        z_graph = z_graph_flat.view(batch.num_graphs, -1) * graph_sigma
                        z0 = torch.cat(
                            [z_nodes, z_edges, z_graph.view(-1, z_graph.shape[-1])],
                            dim=0
                        )
                    else:
                        z0 = torch.cat([z_nodes, z_edges], dim=0)
                else:
                    z0 = torch.cat([z_nodes, z_edges], dim=0)
        
        # Force undirected if needed
        if hasattr(self.model, 'force_undirected') and self.model.force_undirected:
            if isinstance(z0, tuple):
                z0_edges_sym = symmetrize(batch.edge_index, batch.batch, z0[1])
                z0 = (z0[0], z0_edges_sym) + z0[2:] if len(z0) > 2 else (z0[0], z0_edges_sym)
            else:
                num_nodes = batch.num_nodes
                z0_edges = z0[num_nodes:]
                z0_edges_sym = symmetrize(batch.edge_index, batch.batch, z0_edges)
                z0 = torch.cat([z0[:num_nodes], z0_edges_sym], dim=0)
        
        # Pre-create a template to avoid deep copies in the ODE loop
        template = copy.copy(batch)  # Shallow copy is enough for structure
        template.edge_index = batch.edge_index  # Share graph topology
        
        # Define velocity function wrapper
        def v_theta_wrapper(z_nodes, z_edges, z_graph, t, batch_data):
            # Reuse template, only mutate the fields we need
            template.x = z_nodes
            template.edge_attr = z_edges
            if z_graph is not None:
                template.graph_attr = z_graph
            
            # Set conditioning if provided
            if cond is not None:
                template.c = cond
            
            # Forward through model
            return self.model.forward_velocity(template, t)
        
        # Solve ODE
        intermediates = []
        if return_intermediates:
            intermediates.append(z0)
        
        samples = solve_flow(
            v_theta_wrapper, 
            z0, 
            steps=steps, 
            method=method, 
            batch=batch,
            verbose=verbose
        )
        
        if return_intermediates:
            intermediates.append(samples)
            return samples, intermediates
        
        return samples
    
    @torch.no_grad()
    def sample_with_guidance(self, batch, steps, guidance_scale=1.0, 
                           cond=None, uncond=None, **kwargs):
        """
        Classifier-free guided sampling.
        
        Args:
            batch: PyG batch object
            steps: Number of ODE steps
            guidance_scale: Guidance weight (1.0 = no guidance)
            cond: Conditional information
            uncond: Unconditional information (or None)
        """
        if guidance_scale == 1.0 or uncond is None:
            # No guidance, regular sampling
            return self.sample(batch, steps, cond=cond, **kwargs)
        
        # Guided sampling: v = v_uncond + s * (v_cond - v_uncond)
        def guided_velocity(z_nodes, z_edges, z_graph, t, batch_data):
            # Get conditional velocity
            batch_cond = copy.deepcopy(batch_data)
            batch_cond.c = cond
            v_cond = self.model.forward_velocity(batch_cond, t)
            
            # Get unconditional velocity
            batch_uncond = copy.deepcopy(batch_data)
            batch_uncond.c = uncond
            v_uncond = self.model.forward_velocity(batch_uncond, t)
            
            # Apply guidance
            v_guided_nodes = v_uncond[0] + guidance_scale * (v_cond[0] - v_uncond[0])
            v_guided_edges = v_uncond[1] + guidance_scale * (v_cond[1] - v_uncond[1])
            v_guided_graph = None
            if len(v_cond) > 2:
                v_guided_graph = v_uncond[2] + guidance_scale * (v_cond[2] - v_uncond[2])
            
            return v_guided_nodes, v_guided_edges, v_guided_graph
        
        # Sample with guided velocity
        z0 = torch.randn((batch.num_nodes + batch.edge_index.shape[1], 
                         self.model.hid_dim), device=batch.x.device)
        
        return solve_flow(guided_velocity, z0, steps=steps, 
                         batch=batch, **kwargs)
