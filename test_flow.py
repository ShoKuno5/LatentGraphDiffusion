#!/usr/bin/env python
"""
Minimal test script for flow matching implementation.
Tests the core functionality without requiring full training.
"""

import os
import sys
import torch
import copy
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/workspace' if os.path.exists('/workspace') else '.')

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric not found: {e}")
        return False
    
    try:
        from torch_geometric.graphgym.config import cfg
        print("✓ GraphGym config imported")
    except ImportError as e:
        print(f"✗ GraphGym config import failed: {e}")
        return False
    
    try:
        from lgd.flow.flow_core import LatentFlow, VelocityWrapper
        print("✓ LatentFlow imported")
    except ImportError as e:
        print(f"✗ LatentFlow import failed: {e}")
        return False
    
    try:
        from lgd.flow.sampler import FlowSampler, solve_flow
        print("✓ FlowSampler imported")
    except ImportError as e:
        print(f"✗ FlowSampler import failed: {e}")
        return False
    
    try:
        from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder
        print("✓ GraphTransformerEncoder imported")
    except ImportError as e:
        print(f"✗ GraphTransformerEncoder import failed: {e}")
        return False
    
    try:
        from lgd.model.DenoisingTransformer import DenoisingTransformer
        print("✓ DenoisingTransformer imported")
    except ImportError as e:
        print(f"✗ DenoisingTransformer import failed: {e}")
        return False
    
    print("✓ All imports successful!")
    return True


def test_model_creation():
    """Test if LatentFlow model can be created."""
    print("\n" + "=" * 50)
    print("TESTING MODEL CREATION")
    print("=" * 50)
    
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from lgd.flow.flow_core import LatentFlow
    
    # Load config
    cfg_path = 'cfg/zinc-flow_rf.yaml'
    if not os.path.exists(cfg_path):
        print(f"✗ Config file not found: {cfg_path}")
        return False
    
    # Setup minimal config
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    
    # Load the flow config
    with open(cfg_path, 'r') as f:
        import yaml
        flow_cfg = yaml.safe_load(f)
    
    # Apply config manually to avoid full pipeline
    cfg.flow = flow_cfg.get('flow', {})
    cfg.encoder = flow_cfg.get('encoder', {})
    cfg.dt = flow_cfg.get('dt', {})
    cfg.dataset = flow_cfg.get('dataset', {})
    cfg.optim = flow_cfg.get('optim', {})
    cfg.train = flow_cfg.get('train', {})
    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Config loaded from: {cfg_path}")
    print(f"Flow objective: {cfg.flow.get('objective', 'unknown')}")
    print(f"Encoder checkpoint: {cfg.flow.get('first_stage_config', 'unknown')}")
    
    # Check if encoder checkpoint exists (but don't require it for test)
    encoder_path = cfg.flow.get('first_stage_config', '')
    if os.path.exists(encoder_path):
        print(f"✓ Encoder checkpoint found: {encoder_path}")
    else:
        print(f"⚠ Encoder checkpoint not found: {encoder_path}")
        print("  (Will create dummy checkpoint for testing)")
        
        # Create a dummy checkpoint for testing
        dummy_encoder = create_dummy_encoder()
        if dummy_encoder:
            cfg.flow['first_stage_config'] = dummy_encoder
            print(f"✓ Created dummy encoder at: {dummy_encoder}")
    
    # Try to create the model
    try:
        model = LatentFlow(
            first_stage_config=cfg.flow.get('first_stage_config', 'dummy.ckpt'),
            objective=cfg.flow.get('objective', 'rectified'),
            cond_stage_config=cfg.flow.get('cond_stage_config', '__is_unconditional__'),
            cond_stage_key=cfg.flow.get('cond_stage_key', 'unconditional'),
            first_stage_trainable=cfg.flow.get('first_stage_trainable', False),
            cond_stage_trainable=cfg.flow.get('cond_stage_trainable', False),
            conditioning_key=cfg.flow.get('conditioning_key', None),
            hid_dim=cfg.flow.get('hid_dim', 4),
            node_factor=cfg.flow.get('node_factor', 1.0),
            edge_factor=cfg.flow.get('edge_factor', 1.0),
            graph_factor=cfg.flow.get('graph_factor', 1.0),
            use_graph_latent=cfg.flow.get('use_graph_latent', False),
            force_undirected=cfg.flow.get('force_undirected', False),
            use_ema=cfg.flow.get('ema', False),
        )
        print("✓ LatentFlow model created successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_encoder():
    """Create a dummy encoder checkpoint for testing."""
    from torch_geometric.graphgym.config import cfg
    from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder
    
    # Create dummy encoder
    try:
        encoder = GraphTransformerEncoder(cfg=cfg.encoder)
        
        # Save dummy checkpoint
        dummy_path = 'test_encoder.ckpt'
        torch.save({
            'state_dict': encoder.state_dict(),
            'model_state': encoder.state_dict(),
        }, dummy_path)
        
        return dummy_path
    except Exception as e:
        print(f"Could not create dummy encoder: {e}")
        return None


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\n" + "=" * 50)
    print("TESTING FORWARD PASS")
    print("=" * 50)
    
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.data import Data, Batch
    from lgd.flow.flow_core import LatentFlow
    
    # Create a minimal batch
    num_nodes = 10
    num_edges = 20
    num_graphs = 2
    
    # Create simple graph data
    x = torch.randn(num_nodes, 4)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connectivity
    edge_attr = torch.randn(num_edges, 4)  # Edge features
    batch_idx = torch.tensor([0]*5 + [1]*5)  # Graph assignment
    
    # Create PyG batch
    data_list = []
    for i in range(num_graphs):
        mask = (batch_idx == i)
        node_indices = torch.where(mask)[0]
        
        # Get edges for this graph
        edge_mask = (edge_index[0] < len(node_indices)) & (edge_index[1] < len(node_indices))
        
        data = Data(
            x=x[mask],
            edge_index=edge_index[:, edge_mask[:num_edges//2]],
            edge_attr=edge_attr[edge_mask[:num_edges//2]],
            y=torch.tensor([0.5])  # Dummy target
        )
        data.num_nodes = mask.sum().item()
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list)
    print(f"Created batch with {batch.num_nodes} nodes, {batch.edge_index.shape[1]} edges")
    
    # Try forward pass
    try:
        # Create model (reuse from previous test or create new)
        model = create_test_model()
        if model is None:
            print("✗ Could not create model for forward pass test")
            return False
        
        model.eval()
        
        # Test training step (which includes encoding, sampling, loss)
        with torch.no_grad():
            # Mock training_step but simplified
            batch = model.get_input(batch)
            
            # Get latents
            z1 = batch.x_start[:batch.num_nodes]
            z0 = torch.randn_like(z1)
            
            # Sample time
            t = torch.rand(batch.num_graphs)
            
            # Interpolate
            t_expanded = t[batch.batch].unsqueeze(-1)
            z_t = (1 - t_expanded) * z0 + t_expanded * z1
            
            print(f"✓ Latent interpolation successful")
            print(f"  z_t shape: {z_t.shape}")
            
            # Test velocity prediction
            batch_test = copy.deepcopy(batch)
            batch_test.x = z_t
            v_pred = model.forward_velocity(batch_test, t)
            
            print(f"✓ Velocity prediction successful")
            print(f"  v_nodes shape: {v_pred[0].shape}")
            print(f"  v_edges shape: {v_pred[1].shape}")
            
        print("✓ Forward pass test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_model():
    """Create a test model instance."""
    from torch_geometric.graphgym.config import cfg
    from lgd.flow.flow_core import LatentFlow
    
    try:
        # Use dummy checkpoint if real one doesn't exist
        encoder_path = cfg.flow.get('first_stage_config', '')
        if not os.path.exists(encoder_path):
            encoder_path = create_dummy_encoder() or 'dummy.ckpt'
            # Create empty checkpoint file if needed
            if not os.path.exists(encoder_path):
                torch.save({'state_dict': {}}, encoder_path)
        
        model = LatentFlow(
            first_stage_config=encoder_path,
            objective='rectified',
            cond_stage_config='__is_unconditional__',
            cond_stage_key='unconditional',
            hid_dim=4,
            use_ema=False,
        )
        return model
    except Exception as e:
        print(f"Model creation error: {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 50)
    print("LATENT FLOW MATCHING TEST SUITE")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Track test results
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    
    if results['imports']:
        results['model_creation'] = test_model_creation()
        
        if results['model_creation']:
            results['forward_pass'] = test_forward_pass()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("Flow matching implementation is ready for training.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please fix the issues above before running full training.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())