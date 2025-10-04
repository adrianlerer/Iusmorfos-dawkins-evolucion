#!/usr/bin/env python3
"""
Simple test to debug Kahneman framework step by step
"""

import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_imports():
    """Test individual imports first"""
    print("Testing imports...")
    
    try:
        from kahneman_prediction_correction import KahnemanPredictionCorrector
        print("✓ KahnemanPredictionCorrector imported")
    except Exception as e:
        print(f"✗ KahnemanPredictionCorrector import failed: {e}")
        return False
        
    try:
        from kahneman_bias_detector import KahnemanBiasDetector
        print("✓ KahnemanBiasDetector imported")
    except Exception as e:
        print(f"✗ KahnemanBiasDetector import failed: {e}")
        return False
        
    try:
        from universal_legal_taxonomy import UniversalLegalTaxonomy
        print("✓ UniversalLegalTaxonomy imported")
    except Exception as e:
        print(f"✗ UniversalLegalTaxonomy import failed: {e}")
        return False
        
    try:
        from normative_trajectory_analyzer import NormativeTrajectoryAnalyzer
        print("✓ NormativeTrajectoryAnalyzer imported")
    except Exception as e:
        print(f"✗ NormativeTrajectoryAnalyzer import failed: {e}")
        return False
        
    try:
        from kahneman_enhanced_framework import KahnemanEnhancedFramework
        print("✓ KahnemanEnhancedFramework imported")
    except Exception as e:
        print(f"✗ KahnemanEnhancedFramework import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_basic_initialization():
    """Test basic framework initialization"""
    print("\nTesting basic initialization...")
    
    try:
        from kahneman_enhanced_framework import KahnemanEnhancedFramework
        
        framework = KahnemanEnhancedFramework(
            jurisdiction="argentina",
            legal_tradition="civil_law"
        )
        print("✓ KahnemanEnhancedFramework initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual Kahneman components"""
    print("\nTesting individual components...")
    
    try:
        from kahneman_prediction_correction import KahnemanPredictionCorrector, BaseRateData
        corrector = KahnemanPredictionCorrector()
        print("✓ Prediction corrector works")
    except Exception as e:
        print(f"✗ Prediction corrector failed: {e}")
        return False
    
    try:
        from kahneman_bias_detector import KahnemanBiasDetector, PredictionInput
        detector = KahnemanBiasDetector()
        print("✓ Bias detector works")
    except Exception as e:
        print(f"✗ Bias detector failed: {e}")
        return False
        
    return True

def main():
    print("🔍 SIMPLE KAHNEMAN DEBUG TEST")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        return False
        
    # Test individual components
    if not test_individual_components():
        return False
        
    # Test basic initialization
    if not test_basic_initialization():
        return False
    
    print("\n🎉 All basic tests passed!")
    print("Ready to try more complex operations")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)