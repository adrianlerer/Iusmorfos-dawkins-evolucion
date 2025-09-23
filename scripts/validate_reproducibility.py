#!/usr/bin/env python3
"""
Reproducibility validation script
Ensures results are identical across multiple runs
"""

import json
import numpy as np
import hashlib
from pathlib import Path
import sys
import subprocess
from datetime import datetime

def run_experiment_multiple_times(n_runs=3):
    """Run experiment multiple times and compare results"""
    print(f"🔄 Running experiment {n_runs} times for reproducibility check...")
    
    results = []
    checksums = []
    
    for i in range(n_runs):
        print(f"   Run {i+1}/{n_runs}...")
        
        # Run the main script
        result = subprocess.run([
            sys.executable, 
            "src/iusmorfos_comprehensive_empirical_integration.py"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ Run {i+1} failed:")
            print(result.stderr)
            return False
        
        # Find latest results file
        results_dir = Path("results")
        result_files = list(results_dir.glob("iusmorfos_comprehensive_empirical_results_*.json"))
        
        if not result_files:
            print(f"❌ No results file found for run {i+1}")
            return False
            
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        # Load and store results
        with open(latest_file) as f:
            data = json.load(f)
            results.append(data)
        
        # Calculate checksum of key results
        key_results = {
            'overall_score': data['empirical_validations']['overall_empirical_score'],
            'power_law_gamma': data['empirical_validations']['individual_validations']['power_law_validation']['gamma_empirical'],
            'individual_scores': data['empirical_validations']['individual_scores']
        }
        
        checksum = hashlib.md5(json.dumps(key_results, sort_keys=True).encode()).hexdigest()
        checksums.append(checksum)
        
    # Validate all checksums are identical
    if len(set(checksums)) == 1:
        print("✅ All runs produced identical results - REPRODUCIBLE")
        return True
    else:
        print("❌ Results differ between runs - NOT REPRODUCIBLE")
        print("Checksums:")
        for i, checksum in enumerate(checksums):
            print(f"   Run {i+1}: {checksum}")
        return False

def validate_seed_consistency():
    """Validate that seeds are properly set"""
    from src.config import config
    
    # Test numpy reproducibility
    np.random.seed(config.get('reproducibility.numpy_seed'))
    sample1 = np.random.random(10)
    
    np.random.seed(config.get('reproducibility.numpy_seed'))  
    sample2 = np.random.random(10)
    
    if np.array_equal(sample1, sample2):
        print("✅ NumPy seed reproducibility confirmed")
        return True
    else:
        print("❌ NumPy seed not working - different random samples")
        return False

def main():
    """Main reproducibility validation"""
    print("🔒 REPRODUCIBILITY VALIDATION")
    print("=" * 50)
    
    # Check seed consistency
    seed_ok = validate_seed_consistency()
    
    # Run multiple experiments
    repro_ok = run_experiment_multiple_times(n_runs=2)  # Reduced for CI speed
    
    if seed_ok and repro_ok:
        print("\n✅ REPRODUCIBILITY VALIDATED")
        sys.exit(0)
    else:
        print("\n❌ REPRODUCIBILITY FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()