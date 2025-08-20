#!/usr/bin/env python3
"""Quick test of mock client and GRPO trainer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_parallm.rl_training.atropos_client import MockAtroposClient
from mlx_parallm.rl_training.types import ScoredDataGroup

def test_mock_client():
    """Test that mock client generates proper ScoredDataGroups."""
    print("Testing MockAtroposClient...")
    
    client = MockAtroposClient()
    batch = list(client.fetch(batch_size=2))
    
    print(f"\nGenerated {len(batch)} ScoredDataGroups")
    
    for i, group in enumerate(batch):
        print(f"\nGroup {i+1}:")
        print(f"  Number of trajectories: {len(group['tokens'])}")
        print(f"  Scores: {group['scores']}")
        
        for j, (tokens, mask, score) in enumerate(zip(group['tokens'], group['masks'], group['scores'])):
            print(f"  Trajectory {j+1}:")
            print(f"    Tokens: {tokens}")
            print(f"    Mask: {mask}")
            print(f"    Score: {score}")
            print(f"    Response tokens: {sum(mask)} tokens")
    
    # Verify structure
    assert len(batch) == 2, "Should have 2 groups"
    for group in batch:
        assert isinstance(group, dict), "Should be a dict (TypedDict)"
        assert 'tokens' in group, "Must have tokens"
        assert 'masks' in group, "Must have masks"
        assert 'scores' in group, "Must have scores"
        assert len(group['tokens']) == 3, "Each group should have 3 trajectories"
        assert len(group['masks']) == 3, "Each group should have 3 masks"
        assert len(group['scores']) == 3, "Each group should have 3 scores"
        
        # Check that masks and tokens align
        for tokens, mask in zip(group['tokens'], group['masks']):
            assert len(tokens) == len(mask), "Tokens and mask must have same length"
    
    print("\n✓ All checks passed!")
    return True

if __name__ == "__main__":
    success = test_mock_client()
    sys.exit(0 if success else 1)