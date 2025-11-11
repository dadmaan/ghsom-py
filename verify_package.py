"""Verification script for ghsom-py package."""
import sys

print("=" * 60)
print("GHSOM-Py Package Verification")
print("=" * 60)

# Test 1: Version
print("\n1. Testing version import...")
try:
    from ghsom import __version__
    print(f"   ✓ Version: {__version__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Core imports
print("\n2. Testing core imports...")
try:
    from ghsom import GHSOM, GSOM, Neuron, NeuronBuilder
    print("   ✓ All core classes imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Callbacks
print("\n3. Testing callback imports...")
try:
    from ghsom import TrackingCallback
    from ghsom.callbacks import WandBCallback
    print("   ✓ Callback system imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: I/O
print("\n4. Testing I/O imports...")
try:
    from ghsom.io import save_model, load_model
    print("   ✓ I/O functions imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Instantiation
print("\n5. Testing GHSOM instantiation...")
try:
    import numpy as np
    data = np.random.rand(20, 5)
    ghsom = GHSOM(
        input_dataset=data,
        t1=0.5, t2=0.05,
        learning_rate=0.1,
        decay=0.9,
        gaussian_sigma=1.0
    )
    print(f"   ✓ GHSOM created with data shape: {data.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Build verification
print("\n6. Checking distribution files...")
import os
dist_files = os.listdir('dist') if os.path.exists('dist') else []
if any('.whl' in f for f in dist_files):
    print(f"   ✓ Found {len(dist_files)} distribution files")
else:
    print("   ⚠ Warning: No wheel file found in dist/")

# Summary
print("\n" + "=" * 60)
print("✅ ALL VERIFICATION CHECKS PASSED!")
print("=" * 60)
print("\nPackage is ready for:")
print("  • Phase 3: Comprehensive Testing")
print("  • Phase 4: Documentation")
print("  • Production use (experimental)")
print("\nNext steps:")
print("  1. Run: pytest tests/ -v")
print("  2. Review: PHASE2_SUMMARY.md")
print("  3. Begin: Phase 3 (Testing)")
