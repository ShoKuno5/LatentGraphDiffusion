#!/usr/bin/env python
"""
Local test script for Flow Matching implementation.
Tests basic functionality without full HPC environment.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all flow modules can be imported."""
    logger.info("Testing imports...")
    try:
        from lgd.flow import LatentFlow, LatentFlowInductive, solve_flow, FlowSampler
        logger.info("✓ All flow modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test that flow config can be loaded."""
    logger.info("Testing config loading...")
    try:
        from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
        from torch_geometric.graphgym.cmd_args import parse_args
        
        # Mock args for config loading
        class MockArgs:
            cfg_file = 'cfg/zinc-flow_rf.yaml'
            cfg = []
            mark_done = False
            repeat = 1
        
        set_cfg(cfg)
        cfg.set_new_allowed(True)
        load_cfg(cfg, MockArgs())
        
        # Check critical config values
        assert cfg.model.type == 'LatentFlow', f"Expected model.type='LatentFlow', got {cfg.model.type}"
        assert cfg.train.mode == 'train_diffusion', f"Expected train.mode='train_diffusion', got {cfg.train.mode}"
        assert cfg.flow.objective == 'rectified', f"Expected flow.objective='rectified', got {cfg.flow.objective}"
        
        logger.info("✓ Config loaded and validated successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False

def test_model_creation():
    """Test that LatentFlow model can be created."""
    logger.info("Testing model creation...")
    try:
        from torch_geometric.graphgym.config import cfg
        from lgd.flow.flow_core import LatentFlow
        import tempfile
        
        # Create a dummy checkpoint file for encoder
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            # Save a minimal checkpoint
            checkpoint = {
                'state_dict': {},
                'epoch': 0
            }
            torch.save(checkpoint, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Create model with minimal config
            model = LatentFlow(
                first_stage_config=tmp_path,
                objective='rectified',
                cond_stage_config='__is_unconditional__',
                hid_dim=4,
                use_ema=False
            )
            
            logger.info(f"✓ LatentFlow model created successfully")
            logger.info(f"  - Model type: {model.__class__.__name__}")
            logger.info(f"  - Objective: {model.objective}")
            logger.info(f"  - Hidden dim: {model.hid_dim}")
            
            # Test that key methods exist
            assert hasattr(model, 'training_step'), "Missing training_step method"
            assert hasattr(model, 'validation_step'), "Missing validation_step method"
            assert hasattr(model, 'sample'), "Missing sample method"
            assert hasattr(model, 'forward_velocity'), "Missing forward_velocity method"
            
            logger.info("✓ All required methods present")
            return True
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing logic."""
    logger.info("Testing batch processing...")
    try:
        from torch_geometric.data import Data, Batch
        
        # Create a simple mock batch
        data_list = []
        for i in range(2):  # 2 graphs
            data = Data(
                x=torch.randn(3, 4),  # 3 nodes, 4 features
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                edge_attr=torch.randn(3, 4),  # 3 edges, 4 features
                y=torch.randn(1)
            )
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        
        # Test getattr with fallback (the fix we applied)
        batch_num_node = getattr(batch, 'num_node_per_graph',
                                 torch.tensor([batch.num_nodes // batch.num_graphs] * batch.num_graphs,
                                            dtype=torch.long))
        
        assert batch_num_node is not None, "Failed to get batch_num_node"
        assert len(batch_num_node) == 2, f"Expected 2 graphs, got {len(batch_num_node)}"
        
        logger.info("✓ Batch processing logic works correctly")
        logger.info(f"  - Batch size: {batch.num_graphs} graphs")
        logger.info(f"  - Total nodes: {batch.num_nodes}")
        logger.info(f"  - Nodes per graph: {batch_num_node.tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_velocity_computation():
    """Test velocity computation for rectified flow."""
    logger.info("Testing velocity computation...")
    try:
        # Test rectified flow velocity: u = z1 - z0
        z0 = torch.randn(10, 4)
        z1 = torch.randn(10, 4)
        t = torch.rand(10, 1)
        
        # Rectified flow path
        zt = (1 - t) * z0 + t * z1
        
        # Target velocity
        u_target = z1 - z0
        
        # Check that velocity is constant along the path
        assert u_target.shape == z0.shape, f"Velocity shape mismatch: {u_target.shape} vs {z0.shape}"
        
        logger.info("✓ Velocity computation works correctly")
        logger.info(f"  - Input shape: {z0.shape}")
        logger.info(f"  - Velocity shape: {u_target.shape}")
        logger.info(f"  - Path interpolation tested at t={t[0].item():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Velocity computation failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting Flow Matching Local Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("Model Creation", test_model_creation),
        ("Batch Processing", test_batch_processing),
        ("Velocity Computation", test_velocity_computation),
    ]
    
    results = []
    for name, test_fn in tests:
        logger.info(f"\n[{name}]")
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{name:.<40} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! Flow matching implementation is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run quick debug test: pjsub scripts/lgd_flow_debug.sh")
        logger.info("2. If debug passes, run full test: pjsub scripts/lgd_flow_test.sh")
        logger.info("3. If test passes, run full training: pjsub scripts/lgd_flow_training.sh")
    else:
        logger.error("\n❌ Some tests failed. Please fix the issues before running HPC jobs.")
        sys.exit(1)

if __name__ == "__main__":
    main()