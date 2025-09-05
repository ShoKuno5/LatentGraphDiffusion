"""
Latent Flow Matching for Graph Generation.
Main training module parallel to lgd/ddpm/LGD.py but using flow matching.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import logging
import networkx as nx

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from lgd.ddpm.ema import LitEma
from lgd.model.utils import (
    exists, default, mean_flat, count_params, 
    num2batch, symmetrize
)
from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder
from lgd.model.SyntheticGraphTransformerEncoder import GraphTransformerSyntheticEncoder
from lgd.model.DenoisingTransformer import DenoisingTransformer
from .sampler import FlowSampler, solve_flow


def disabled_train(self, mode=True):
    """Overwrite model.train to freeze training mode."""
    return self


class VelocityWrapper(nn.Module):
    """
    Wrapper for the denoising network to output velocities.
    Reuses the DenoisingTransformer from LGD.
    """
    def __init__(self, denoiser_model, conditioning_key=None):
        super().__init__()
        self.denoiser = denoiser_model
        self.conditioning_key = conditioning_key
        
    def forward(self, batch, t, c=None):
        """
        Forward pass predicting velocity.
        The denoiser outputs a "denoised" version which we interpret as velocity.
        """
        # Feed continuous time directly; the denoiser embeds time internally
        output = self.denoiser(batch, t, c)
        
        return output


class LatentFlow(pl.LightningModule):
    """
    Latent Flow Matching model for graph generation.
    Implements rectified flow and Gaussian CFM objectives.
    """
    
    def __init__(self,
                 first_stage_config,  # Path to pretrained encoder checkpoint
                 denoiser_config=None,  # Config for velocity network
                 objective="rectified",  # "rectified" or "gaussian_cfm"
                 cond_stage_config=None,  # Conditioning config
                 cond_stage_key="unconditional",
                 first_stage_trainable=False,
                 cond_stage_trainable=False,
                 conditioning_key=None,
                 hid_dim=128,
                 node_factor=1.0,
                 edge_factor=1.0,
                 graph_factor=1.0,
                 use_graph_latent=False,
                 force_undirected=False,
                 use_ema=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor="val/loss",
                 scheduler_config=None,
                 # Gaussian CFM specific params
                 alpha_fn="linear",  # Schedule for mu_t
                 sigma_fn="constant",  # Schedule for sigma_t
                 sigma_min=0.01,
                 sigma_max=1.0,
                 # Training params
                 learning_rate=1e-4,
                 weight_decay=0.0,
                 train_mode='sample',
                 *args, **kwargs):
        super().__init__()
        
        self.objective = objective
        assert objective in ["rectified", "gaussian_cfm"]
        
        self.hid_dim = hid_dim
        self.node_factor = cfg.flow.get("node_factor", node_factor)
        self.edge_factor = cfg.flow.get("edge_factor", edge_factor)
        self.graph_factor = cfg.flow.get("graph_factor", graph_factor)
        self.use_graph_latent = use_graph_latent
        self.force_undirected = cfg.flow.get('force_undirected', force_undirected)
        self.first_stage_trainable = first_stage_trainable
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.conditioning_key = conditioning_key
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_mode = train_mode
        
        # Setup schedules for Gaussian CFM
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Load first stage encoder/decoder
        self.instantiate_first_stage(first_stage_config)
        
        # Setup conditioning if needed
        if cond_stage_config is not None:
            self.instantiate_cond_stage(cond_stage_config)
        else:
            self.cond_stage_model = None
        
        # Build velocity network (reuse DenoisingTransformer)
        self.model = self.build_velocity_network(denoiser_config)
        count_params(self.model, verbose=True)
        
        # EMA
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        # Scheduler config
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        
        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
    
    def build_velocity_network(self, config):
        """Build the velocity prediction network."""
        # Use the same DenoisingTransformer as LGD
        denoiser = DenoisingTransformer(cfg)
        # Wrap it to output velocities
        return VelocityWrapper(denoiser, self.conditioning_key)
    
    def instantiate_first_stage(self, config):
        """Load pretrained encoder/decoder."""
        model = eval(cfg.encoder.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.encoder)
        self.first_stage_model = model
        
        # Load checkpoint
        sd = torch.load(config, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        if "model_state" in list(sd.keys()):
            sd = sd["model_state"]
        
        state_dict = {}
        for k, v in sd.items():
            new_k = k[6:] if k.startswith('model.') else k
            state_dict[new_k] = v
        
        missing, unexpected = self.first_stage_model.load_state_dict(state_dict, strict=False)
        logging.info(f"Restored encoder from {config}")
        logging.info(f"Missing Keys: {len(missing)}, Unexpected Keys: {len(unexpected)}")
        
        if not self.first_stage_trainable:
            self.first_stage_model = self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        
        # Infer and validate latent dimension from first stage
        if hasattr(self.first_stage_model, 'out_dim'):
            encoder_latent_dim = self.first_stage_model.out_dim
        elif hasattr(cfg, 'encoder') and hasattr(cfg.encoder, 'out_dim'):
            encoder_latent_dim = cfg.encoder.out_dim
        else:
            encoder_latent_dim = self.hid_dim  # fallback
        
        if self.hid_dim != encoder_latent_dim:
            logging.warning(f"hid_dim={self.hid_dim} != encoder latent dim={encoder_latent_dim}, using encoder dim")
            self.hid_dim = encoder_latent_dim
    
    def instantiate_cond_stage(self, config):
        """Setup conditioning model."""
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            self.cond_stage_model = None
        else:
            # Load conditioning model
            model = eval(cfg.cond.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.cond)
            self.cond_stage_model = model
            sd = torch.load(config, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            if "model_state" in list(sd.keys()):
                sd = sd["model_state"]
            
            state_dict = {}
            for k, v in sd.items():
                new_k = k[6:] if k.startswith('model.') else k
                state_dict[new_k] = v
            
            missing, unexpected = self.cond_stage_model.load_state_dict(state_dict, strict=False)
            logging.info(f"Restored cond model from {config}")
            
            if not self.cond_stage_trainable:
                self.cond_stage_model = self.cond_stage_model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
    
    @torch.no_grad()
    def encode_first_stage(self, batch, label=None, prefix=None):
        """Encode graph to latent space."""
        if hasattr(self.first_stage_model, 'encode'):
            return self.first_stage_model.encode(batch, prefix=prefix, label=label)
        else:
            return self.first_stage_model(batch, prefix=prefix, label=label)
    
    @torch.no_grad()
    def decode_first_stage(self, batch_z):
        """Decode from latent space to graph space."""
        return self.first_stage_model.decode(batch_z)
    
    def get_alpha_sigma(self, t):
        """Get alpha_t and sigma_t for Gaussian CFM."""
        if self.objective == "rectified":
            # For rectified flow: linear interpolation
            return t, torch.zeros_like(t)
        
        # Gaussian CFM schedules
        if self.alpha_fn == "linear":
            alpha_t = t
        elif self.alpha_fn == "cosine":
            alpha_t = torch.sin(t * np.pi / 2)
        else:
            alpha_t = t
        
        if self.sigma_fn == "constant":
            sigma_t = self.sigma_min * torch.ones_like(t)
        elif self.sigma_fn == "linear":
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        else:
            sigma_t = self.sigma_min * torch.ones_like(t)
        
        return alpha_t, sigma_t
    
    def get_velocity_target(self, z0, z1, t, zt=None):
        """
        Compute target velocity for training.
        
        For rectified flow: u = z1 - z0
        For Gaussian CFM: u = d_alpha/dt * z1 + (d_sigma/dt / sigma) * (zt - mu_t)
        """
        if self.objective == "rectified":
            return z1 - z0
        
        # Gaussian CFM
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        
        # Compute derivatives
        eps = 1e-5
        alpha_t_plus = self.get_alpha_sigma(t + eps)[0]
        sigma_t_plus = self.get_alpha_sigma(t + eps)[1]
        
        d_alpha_dt = (alpha_t_plus - alpha_t) / eps
        d_sigma_dt = (sigma_t_plus - sigma_t) / eps
        
        # Target velocity (Eq. (*) from instructions)
        mu_t = alpha_t * z1
        u = d_alpha_dt * z1 + (d_sigma_dt / (sigma_t + 1e-8)) * (zt - mu_t)
        
        return u
    
    def sample_zt(self, z0, z1, t):
        """
        Sample zt along the path.
        
        For rectified flow: zt = (1-t) * z0 + t * z1
        For Gaussian CFM: zt ~ N(alpha_t * z1, sigma_t^2 * I)
        """
        if self.objective == "rectified":
            return (1 - t) * z0 + t * z1
        
        # Gaussian CFM
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        mu_t = alpha_t * z1
        noise = torch.randn_like(z1)
        return mu_t + sigma_t * noise
    
    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, force_c_encode=False, return_original_cond=False):
        """Prepare input batch with encoding."""
        batch = batch.to(self.device)
        batch.x_0 = batch.x.clone().detach()
        batch.edge_attr_0 = batch.edge_attr.clone().detach()
        
        # Encode to latent space
        input_label = batch.y.clone().detach() if cfg.train.pretrain.input_target else None
        if input_label is not None and cfg.dataset.format == 'PyG-QM9':
            input_label = (input_label - batch.y_mean) / batch.y_std
        
        batch_z = copy.deepcopy(batch)
        batch_z = self.encode_first_stage(batch_z, label=input_label)
        
        if not self.first_stage_trainable:
            batch_z.x = batch_z.x.detach()
            batch_z.edge_attr = batch_z.edge_attr.detach()
            if hasattr(batch_z, 'graph_attr'):
                batch_z.graph_attr = batch_z.graph_attr.detach()
        
        # Create concatenated latent representation
        batch.x_start = torch.cat([batch_z.x, batch_z.edge_attr], dim=0)
        if hasattr(batch_z, 'graph_attr') and self.use_graph_latent:
            batch.graph_start = batch_z.graph_attr.clone().detach()
        
        # Setup batch indices
        batch_num_node = getattr(batch, 'num_node_per_graph', 
                                 torch.tensor([batch.num_nodes // batch.num_graphs] * batch.num_graphs,
                                            dtype=torch.long, device=batch.x.device))
        batch_idx = torch.cat([batch.batch, num2batch(batch_num_node ** 2)], dim=0)
        batch.batch_idx = batch_idx
        
        # Add decoded versions for compatibility with inference
        if return_first_stage_outputs:
            # Store reconstructed versions (for flow, these are just the originals since we don't reconstruct during inference)
            batch.x_rec = batch.x_0.clone()
            batch.edge_attr_rec = batch.edge_attr_0.clone()
            # Ensure graph_start exists for inference compatibility
            if not hasattr(batch, 'graph_start'):
                batch.graph_start = None
            # Handle graph_attr_rec safely
            if hasattr(batch, 'graph_start') and batch.graph_start is not None:
                batch.graph_attr_rec = batch.graph_start.clone()
            else:
                batch.graph_attr_rec = None
        
        return batch
    
    def forward_velocity(self, batch, t):
        """
        Forward pass to predict velocity.
        Returns (v_nodes, v_edges, v_graph).
        """
        # Call the velocity network
        batch_out = self.model(batch, t, getattr(batch, 'c', None))
        
        # Split output into nodes and edges
        num_nodes = batch.num_nodes
        v_nodes = batch_out.x
        v_edges = batch_out.edge_attr
        v_graph = getattr(batch_out, 'graph_attr', None) if self.use_graph_latent else None
        
        return v_nodes, v_edges, v_graph
    
    def training_step(self, batch, batch_idx):
        """Main training step."""
        # Prepare input
        batch = self.get_input(batch)
        
        # Get latent representations
        z1 = batch.x_start.clone().detach()
        num_nodes = batch.num_nodes
        z1_nodes = z1[:num_nodes]
        z1_edges = z1[num_nodes:]
        z1_graph = getattr(batch, 'graph_start', None)
        
        # Sample noise z0
        z0_nodes = torch.randn_like(z1_nodes)
        z0_edges = torch.randn_like(z1_edges)
        z0_graph = torch.randn_like(z1_graph) if z1_graph is not None else None
        
        # Force undirected if needed
        if self.force_undirected:
            z0_edges = symmetrize(batch.edge_index, batch.batch, z0_edges)
            z1_edges = symmetrize(batch.edge_index, batch.batch, z1_edges)
        
        # Sample time uniformly
        t = torch.rand(batch.num_graphs, device=self.device)
        
        # Broadcast time to nodes and edges
        t_nodes = t[batch.batch]
        # Get num_node_per_graph with fallback
        batch_num_node = getattr(batch, 'num_node_per_graph',
                                 torch.tensor([batch.num_nodes // batch.num_graphs] * batch.num_graphs,
                                            dtype=torch.long, device=batch.x.device))
        batch_edge_idx = num2batch(batch_num_node ** 2)
        t_edges = t[batch_edge_idx]
        
        # Sample zt along the path
        zt_nodes = self.sample_zt(z0_nodes, z1_nodes, t_nodes.unsqueeze(-1))
        zt_edges = self.sample_zt(z0_edges, z1_edges, t_edges.unsqueeze(-1))
        zt_graph = self.sample_zt(z0_graph, z1_graph, t.unsqueeze(-1)) if z1_graph is not None else None
        
        # Get target velocity
        u_nodes = self.get_velocity_target(z0_nodes, z1_nodes, t_nodes.unsqueeze(-1), zt_nodes)
        u_edges = self.get_velocity_target(z0_edges, z1_edges, t_edges.unsqueeze(-1), zt_edges)
        u_graph = self.get_velocity_target(z0_graph, z1_graph, t.unsqueeze(-1), zt_graph) if z1_graph is not None else None
        
        # Prepare batch for velocity network
        batch_flow = copy.deepcopy(batch)
        batch_flow.x = zt_nodes
        batch_flow.edge_attr = zt_edges
        if zt_graph is not None:
            batch_flow.graph_attr = zt_graph
        
        # Predict velocity
        v_nodes, v_edges, v_graph = self.forward_velocity(batch_flow, t)
        
        # Compute loss
        loss_nodes = F.mse_loss(v_nodes, u_nodes)
        loss_edges = F.mse_loss(v_edges, u_edges)

        # Optional time weighting
        t_weight_mode = cfg.flow.get('t_weight', 'none') if hasattr(cfg, 'flow') else 'none'
        if t_weight_mode and t_weight_mode != 'none':
            if t_weight_mode == 't(1-t)':
                wn = torch.mean(t_nodes * (1.0 - t_nodes))
                we = torch.mean(t_edges * (1.0 - t_edges))
                loss_nodes = loss_nodes * wn
                loss_edges = loss_edges * we
            # add other weighting schemes here if needed

        loss = self.node_factor * loss_nodes + self.edge_factor * loss_edges
        
        loss_graph_val = torch.zeros(1, device=self.device)
        if v_graph is not None and u_graph is not None:
            loss_graph_val = F.mse_loss(v_graph, u_graph)
            loss = loss + self.graph_factor * loss_graph_val
            self.log("train/loss_graph", loss_graph_val, prog_bar=False)
        
        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_nodes", loss_nodes, prog_bar=False)
        self.log("train/loss_edges", loss_edges, prog_bar=False)
        
        # Return values expected by train_diffusion mode
        # (loss, loss_task, pred, loss_node, loss_edge, loss_graph, loss_encoder)
        # For flow matching, we don't have a separate task loss or encoder loss
        loss_task = torch.zeros(1, device=self.device)
        loss_encoder = torch.zeros(1, device=self.device)
        
        # For pred, we can return the velocity prediction or a dummy value
        # Since we're doing unconditional generation, return dummy predictions
        pred = torch.zeros_like(batch.y) if hasattr(batch, 'y') else torch.zeros(1, device=self.device)
        
        return loss, loss_task, pred, loss_nodes, loss_edges, loss_graph_val, loss_encoder
    
    def validation_step(self, batch, batch_idx):
        """Validation step using proper inference."""
        with torch.no_grad():
            # Use inference method for validation (not training_step)
            loss_graph, _ = self.inference(batch, ddim_steps=20, sample=True)
            
            # Log validation loss
            self.log("val/loss", loss_graph, prog_bar=True)
            
            return loss_graph
    
    def test_step(self, batch, batch_idx):
        """Test step using proper inference."""
        with torch.no_grad():
            # Use inference method for testing (not training_step)
            loss_graph, _ = self.inference(batch, ddim_steps=20, sample=True)
            
            # Log test loss
            self.log("test/loss", loss_graph, prog_bar=True)
            
            return loss_graph
    
    def configure_optimizers(self):
        """Setup optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.use_scheduler:
            scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch),
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        
        return optimizer
    
    @torch.no_grad()
    def inference(self, batch, sample=True, ddim_steps=None, ddim_eta=0., use_ddpm_steps=False, return_keys=None,
                  quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                  plot_diffusion_rows=True, visualize=False, **kwargs):
        """
        Inference method parallel to diffusion version for compatibility.
        For flow matching, ddim_steps controls the number of ODE integration steps.
        """
        # Flow matching uses ODE steps instead of DDIM
        flow_steps = ddim_steps if ddim_steps is not None else 20
        
        # Follow exact same pattern as diffusion inference
        N = batch.num_graphs
        batch = self.get_input(batch, return_first_stage_outputs=True, 
                               force_c_encode=True, return_original_cond=True)
        
        # Extract same components as diffusion (for consistency) - use safe attribute access
        graph_start = getattr(batch, 'graph_start', None)
        graph_attr_rec = getattr(batch, 'graph_attr_rec', None)
        z, c, x, xrec, xc = (batch.x_start, graph_start), batch.get('c', None), (batch.x_0, batch.edge_attr_0), (batch.x_rec, batch.edge_attr_rec, graph_attr_rec), batch.get('xc', None)
        batch_idx = batch.batch_idx
        
        # Flow sampling instead of diffusion sampling
        samples = self.sample_flow_for_inference(batch=batch, cond=c, batch_size=N, 
                                                 batch_idx=batch_idx, steps=flow_steps)
        
        # Use exact same decoding as diffusion
        node_decode, edge_decode, graph_decode = self.decode_first_stage(samples)
        
        # Same denormalization as diffusion
        if cfg.dataset.format == 'PyG-QM9':
            graph_decode = graph_decode * batch.get('y_std', 1.) + batch.get('y_mean', 0.)
        
        # Same loss computation as diffusion
        loss_graph, graph_decode = compute_loss(graph_decode, batch.y.clone().detach())
        
        # Same generation mode handling as diffusion
        if cfg.train.mode in ['qm9_unconditional', 'qm9_conditional']:
            generated_mol = []
            accumulated_node, accumulated_edge = 0, 0
            for i in range(batch.num_graphs):
                num_nodes_i = batch.num_node_per_graph[i] if hasattr(batch, 'num_node_per_graph') else batch.num_nodes // batch.num_graphs
                generated_mol.append((
                    torch.argmax(node_decode[accumulated_node: accumulated_node + num_nodes_i], dim=1, keepdim=False),
                    torch.argmax(edge_decode[accumulated_edge: accumulated_edge + num_nodes_i ** 2], dim=1, keepdim=False).reshape(num_nodes_i, num_nodes_i),
                    graph_decode[i].unsqueeze(0)
                ))
                accumulated_node += num_nodes_i
                accumulated_edge += num_nodes_i ** 2
            graph_decode = generated_mol
        
        elif cfg.train.mode in ['generic_generation']:
            generic_graphs = []
            accumulated_node, accumulated_edge = 0, 0
            for i in range(batch.num_graphs):
                num_nodes_i = batch.num_node_per_graph[i] if hasattr(batch, 'num_node_per_graph') else batch.num_nodes // batch.num_graphs
                adj = torch.argmax(edge_decode[accumulated_edge: accumulated_edge + num_nodes_i ** 2], dim=1, keepdim=False)\
                      .reshape(num_nodes_i, num_nodes_i).detach().cpu().numpy()
                G = nx.from_numpy_array(adj)
                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))
                if G.number_of_nodes() < 1:
                    G.add_node(1)
                generic_graphs.append(G)

                accumulated_node += num_nodes_i
                accumulated_edge += num_nodes_i ** 2
            graph_decode = generic_graphs
        
        return loss_graph, graph_decode
    
    @torch.no_grad()
    def sample_flow_for_inference(self, batch, cond, batch_size, batch_idx, steps):
        """
        Flow sampling that returns a batch object compatible with decode_first_stage.
        This replaces the diffusion sample_log method.
        """
        # Sample latents using flow ODE
        hid = self.hid_dim
        device = self.device
        
        # Compute shapes from batch structure  
        if hasattr(batch, 'num_node_per_graph'):
            n_per_g = batch.num_node_per_graph
            E_dense = int((n_per_g * n_per_g).sum().item())
        else:
            E_dense = batch.edge_index.shape[1]
        
        N = batch.num_nodes
        shape_nodes = (N, hid)
        shape_edges = (E_dense, hid)
        
        # Create sampler
        sampler = FlowSampler(self)
        
        # Sample using flow ODE
        samples = sampler.sample(
            batch, 
            steps=steps,
            batch_size=batch_size,
            shape=(shape_nodes, shape_edges),
            method="heun",
            verbose=False,
            cond=cond
        )
        
        # Convert samples to batch format expected by decoder
        if isinstance(samples, tuple):
            z_nodes, z_edges = samples[0], samples[1] 
            z_graph = samples[2] if len(samples) > 2 else None
        else:
            # Handle concatenated format
            z_nodes = samples[:N]
            z_edges = samples[N:]
            z_graph = None
            
        # Create batch object for decoder (same format as diffusion sample_log output)
        samples_batch = copy.deepcopy(batch)
        samples_batch.x = z_nodes
        samples_batch.edge_attr = z_edges
        
        # Handle graph attributes - ensure they exist for decoder
        if z_graph is not None and self.use_graph_latent:
            samples_batch.graph_attr = z_graph
        else:
            # Create dummy graph_attr to prevent decoder errors
            samples_batch.graph_attr = torch.zeros(batch_size, hid, device=device)
            
        return samples_batch
    
    @torch.no_grad()
    def sample(self, batch, steps=20, method="heun", verbose=True, **kwargs):
        """Sample from the model."""
        sampler = FlowSampler(self)
        hid = self.hid_dim
        
        # Compute shapes deterministically from graph structure
        if hasattr(batch, 'num_node_per_graph'):
            # Dense edge latents: sum of n_i^2 per graph
            n_per_g = batch.num_node_per_graph
            E_dense = int((n_per_g * n_per_g).sum().item())
        else:
            # Sparse edge latents: use actual edge count (fallback)
            E_dense = batch.edge_index.shape[1]
        
        N = batch.num_nodes
        shape_nodes = (N, hid)
        shape_edges = (E_dense, hid)
        
        samples = sampler.sample(
            batch, 
            steps=steps,
            batch_size=batch.num_graphs,
            shape=(shape_nodes, shape_edges),  # Pass tuple of shapes
            method=method,
            verbose=verbose,
            **kwargs
        )
        return samples
    
    @contextmanager
    def ema_scope(self, context=None):
        """EMA scope for sampling."""
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        """Update EMA after each training batch."""
        if self.use_ema:
            self.model_ema(self.model)


class LatentFlowInductive(LatentFlow):
    """
    Inductive version of Latent Flow for handling unseen graphs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized LatentFlowInductive model")
    
    # Override methods as needed for inductive setting
    pass
