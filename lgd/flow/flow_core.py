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

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from lgd.ddpm.ema import LitEma
from lgd.model.utils import (
    exists, default, mean_flat, count_params, 
    num2batch, symmetrize
)
from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder
from lgd.model.SyntheticGraphTransformerEncoder import GraphTransformerSyntheticEncoder as SyntheticGraphTransformerEncoder
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
        # The denoiser expects timesteps in [0, num_timesteps]
        # For flow, t is in [0, 1], so we scale for compatibility
        # TODO: Consider feeding continuous t directly if denoiser supports it
        t_scaled = t * 999.0  # Map [0,1] to [0,999] for compatibility, keep as float
        
        # Call denoiser (it will handle conditioning internally)
        output = self.denoiser(batch, t_scaled, c)
        
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
    def get_input(self, batch, return_first_stage_outputs=False):
        """Prepare input batch with encoding."""
        assert hasattr(batch, 'num_node_per_graph'), \
            "Expected `batch.num_node_per_graph` to be set by dataset transforms. " \
            "Ensure virtual node/edge preprocessing is enabled for dense edge layout."
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
        #batch_idx = torch.cat([batch.batch, num2batch(batch_num_node ** 2)], dim=0)
        #batch.batch_idx = batch_idx
        
        edge_batch = batch.batch[batch.edge_index[0]]
        batch.batch_idx = torch.cat([batch.batch, edge_batch], dim=0)
        
        return batch
    
    def forward_velocity(self, batch, t):
        """
        Forward pass to predict velocity.
        Returns (v_nodes, v_edges, v_graph).
        """
        assert hasattr(batch, 'num_node_per_graph'), \
            "Expected `batch.num_node_per_graph` in forward pass. " \
            "It is required by the denoiser for dense edge indexing."
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
        # training_step 内
        # エッジが属するグラフIDを source ノード側から作る
        edge_batch = batch.batch[batch.edge_index[0]]  # shape: [num_edges]
        t_edges = t[edge_batch]                        # shape: [num_edges]

        
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
        loss = self.node_factor * loss_nodes + self.edge_factor * loss_edges
        
        loss_graph_val = torch.zeros(1, device=self.device)
        if v_graph is not None and u_graph is not None:
            loss_graph_val = F.mse_loss(v_graph, u_graph)
            loss = loss + self.graph_factor * loss_graph_val
            # self.log("train/loss_graph", loss_graph_val, prog_bar=False)
        
        # Logging
        # self.log("train/loss", loss, prog_bar=True)
        # self.log("train/loss_nodes", loss_nodes, prog_bar=False)
        # self.log("train/loss_edges", loss_edges, prog_bar=False)

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
        """Validation step."""
        with torch.no_grad():
            # Same as training but without gradients
            result = self.training_step(batch, batch_idx)
            # Unpack the result tuple
            loss = result[0] if isinstance(result, tuple) else result
            
            # Log validation loss
            # self.log("val/loss", loss, prog_bar=True)
            
            return loss
    
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

    @torch.no_grad()
    def inference(self, batch, ddim_steps=None, ddim_eta=0.0, use_ddpm_steps=False, **kwargs):
        """
        Evaluation for flow matching with decoding to graph properties.

        Returns:
            (loss_graph, graph_pred)
            - loss_graph: property loss (e.g., L1) for logging/selection
            - graph_pred: decoded graph-level predictions for downstream metrics
        """
        # Prepare encoded inputs (sets x_start, graph_start, etc.)
        batch = self.get_input(batch)

        # Sample final latent z(1) via ODE solver
        steps = getattr(cfg, 'flow', {}).get('nfe', 20)
        solver = getattr(cfg, 'flow', {}).get('solver', 'heun')
        with self.ema_scope("Inference"):
            samples = self.sample(batch, steps=steps, method=solver, verbose=False)

        # Build a decode batch with sampled latents
        batch_dec = copy.deepcopy(batch)
        if isinstance(samples, (tuple, list)):
            z_nodes, z_edges = samples[0], samples[1]
            z_graph = samples[2] if len(samples) > 2 else None
            batch_dec.x = z_nodes
            batch_dec.edge_attr = z_edges
            if z_graph is not None:
                batch_dec.graph_attr = z_graph
            elif hasattr(batch, 'graph_start'):
                # Fallback: reuse encoder graph latent if graph latent not modeled
                batch_dec.graph_attr = batch.graph_start
        else:
            # Legacy concatenated format (nodes first then edges)
            num_nodes = batch.num_nodes
            batch_dec.x = samples[:num_nodes]
            batch_dec.edge_attr = samples[num_nodes:]
            if hasattr(batch, 'graph_start'):
                batch_dec.graph_attr = batch.graph_start

        # Decode to node/edge/graph predictions
        node_dec, edge_dec, graph_dec = self.decode_first_stage(batch_dec)

        # Unnormalize targets for certain datasets (match diffusion behavior)
        if cfg.dataset.format == 'PyG-QM9':
            graph_dec = graph_dec * batch.get('y_std', 1.) + batch.get('y_mean', 0.)

        # Compute property loss against ground-truth
        true = batch.y.clone().detach() if hasattr(batch, 'y') else graph_dec.detach()
        loss_graph, _ = compute_loss(graph_dec, true)

        return loss_graph, graph_dec


class LatentFlowInductive(LatentFlow):
    """
    Inductive version of Latent Flow for handling unseen graphs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized LatentFlowInductive model")
    
    # Override methods as needed for inductive setting
    pass
