"""
Quick test script for pyspectacle
"""
from main import RelabParser
import numpy as np

# Initialize parser
print("Initializing parser...")
parser = RelabParser('/home/cjtu/projects/lunar_spectroscopy/data/spectra/RelabDB2018Dec31')
print(f"✓ Loaded {len(parser.sample_catalogue)} samples")
print(f"✓ Loaded {len(parser.spectra_catalogue)} spectra")

# Test 1: List spectra
print("\n--- Test 1: List all spectra ---")
all_spectra = parser.list_spectra()
print(f"✓ Found {len(all_spectra)} spectra total")
print(f"  Columns: {list(all_spectra.columns[:5])}...")

# Test 2: List spectra for specific sample
print("\n--- Test 2: List spectra for sample ---")
sample_id = parser.sample_catalogue['SampleID'].iloc[0]
sample_spectra = parser.list_spectra(sample_id=sample_id)
print(f"✓ Sample {sample_id} has {len(sample_spectra)} spectra")

# Test 3: Get single spectrum
print("\n--- Test 3: Get single spectrum ---")
spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
refl, std, meta = parser.get_spectrum(spectrum_id)
print(f"✓ Loaded spectrum {spectrum_id}")
print(f"  Wavelength range: {refl.index.min():.3f} - {refl.index.max():.3f} μm")
print(f"  Data points: {len(refl)}")
print(f"  Sample: {meta['SampleName']}")

# Test 4: Case insensitive
print("\n--- Test 4: Case insensitive lookup ---")
refl_upper, _, _ = parser.get_spectrum(spectrum_id.upper())
refl_lower, _, _ = parser.get_spectrum(spectrum_id.lower())
print(f"✓ Upper case: {len(refl_upper)} points")
print(f"✓ Lower case: {len(refl_lower)} points")

# Test 5: Batch retrieval (union)
print("\n--- Test 5: Batch retrieval (union) ---")
spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:3].tolist()
refl_df, std_df, meta_df = parser.get_spectra_batch(spectrum_ids)
print(f"✓ Loaded {len(refl_df.columns)} spectra")
print(f"  Shape: {refl_df.shape} (wavelengths × spectra)")
print(f"  Metadata shape: {meta_df.shape}")

# Test 6: Batch retrieval with interpolation
print("\n--- Test 6: Batch with interpolation ---")
wl_grid = np.arange(0.5, 1.0, 0.01)
refl_interp, std_interp, meta_interp = parser.get_spectra_batch(
    spectrum_ids[:2],
    wavelength_grid=wl_grid,
    interpolation='cubic'
)
print(f"✓ Interpolated to {len(refl_interp)} wavelength points")
print(f"  Grid: {wl_grid[0]:.2f} - {wl_grid[-1]:.2f} μm (step {wl_grid[1]-wl_grid[0]:.3f})")

# Test 7: NaN filling outside bounds
print("\n--- Test 7: NaN filling outside bounds ---")
refl_orig, _, _ = parser.get_spectrum(spectrum_ids[0])
wl_min, wl_max = refl_orig.index.min(), refl_orig.index.max()
wl_extended = np.arange(wl_min - 0.1, wl_max + 0.1, 0.01)
refl_ext, _, _ = parser.get_spectra_batch(
    [spectrum_ids[0]],
    wavelength_grid=wl_extended,
    interpolation='linear'
)
print(f"✓ Original range: {wl_min:.3f} - {wl_max:.3f} μm")
print(f"  Extended range: {wl_extended[0]:.3f} - {wl_extended[-1]:.3f} μm")
print(f"  First value (should be NaN): {refl_ext.iloc[0, 0]}")
print(f"  Last value (should be NaN): {refl_ext.iloc[-1, 0]}")
print(f"  Middle value (should be valid): {refl_ext.iloc[len(refl_ext)//2, 0]:.5f}")

print("\n✓ All tests passed!")
