"""
Tests for pyspectacle

Test coverage:
- Catalogue loading
- Spectrum listing with filters
- Single spectrum retrieval
- Batch retrieval with union wavelengths
- Batch retrieval with interpolation
- Case-insensitive spectrum ID lookup
- Metadata joining
- Missing file handling
- NaN filling outside interpolation bounds
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from main import RelabParser


# Fixture for parser instance
@pytest.fixture
def parser():
    """Create parser instance for testing."""
    # db_path = Path("/home/cjtu/projects/lunar_spectroscopy/data/spectra/RelabDB2018Dec31")
    db_path = Path("/mnt/c/Users/cjtai/Downloads/RelabDatabase2024Dec31")
    if not db_path.exists():
        pytest.skip(f"Database not found at {db_path}")
    return RelabParser(db_path)


class TestRelabParser:
    """Test suite for RelabParser class."""
    
    def test_init_loads_catalogues(self, parser):
        """Test that catalogues are loaded on initialization."""
        assert parser.sample_catalogue is not None
        assert parser.spectra_catalogue is not None
        assert len(parser.sample_catalogue) > 0
        assert len(parser.spectra_catalogue) > 0
        assert 'SampleID' in parser.sample_catalogue.columns
        assert 'SpectrumID' in parser.spectra_catalogue.columns
    
    def test_list_spectra_no_filter(self, parser):
        """Test listing all spectra without filters."""
        df = parser.list_spectra()
        assert len(df) > 0
        # Should have columns from both catalogues
        assert 'SpectrumID' in df.columns
        assert 'SampleID' in df.columns
        assert 'SampleName' in df.columns
    
    def test_list_spectra_by_sample_id(self, parser):
        """Test filtering by Sample ID."""
        # Get first sample ID from catalogue
        sample_id = parser.sample_catalogue['SampleID'].iloc[0]
        df = parser.list_spectra(sample_id=sample_id)
        assert len(df) > 0
        assert all(df['SampleID'] == sample_id)
    
    def test_list_spectra_with_filters(self, parser):
        """Test filtering with keyword arguments."""
        # Filter by Source if available
        if 'Source' in parser.sample_catalogue.columns:
            df = parser.list_spectra(Source='Earth')
            if len(df) > 0:
                assert all(df['Source'] == 'Earth')
    
    def test_get_spectrum_returns_tuple(self, parser):
        """Test that get_spectrum returns 3-tuple."""
        # Get first spectrum ID
        spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
        result = parser.get_spectrum(spectrum_id)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        refl, std, meta = result
        assert isinstance(refl, pd.Series)
        assert isinstance(std, pd.Series)
        assert isinstance(meta, pd.Series)
    
    def test_get_spectrum_data_structure(self, parser):
        """Test that spectrum data has correct structure."""
        spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
        refl, std, meta = parser.get_spectrum(spectrum_id)
        
        # Reflectance and std_dev should have wavelength as index
        assert refl.index.name is None or 'wavelength' in str(refl.index.name).lower()
        assert len(refl) > 0
        assert len(std) == len(refl)
        
        # Values should be numeric
        assert np.issubdtype(refl.dtype, np.number)
        
        # Metadata should contain fields from both catalogues
        assert 'SampleID' in meta.index
        assert 'SpectrumID' in meta.index
    
    def test_get_spectrum_case_insensitive(self, parser):
        """Test case-insensitive spectrum ID lookup."""
        # Get a spectrum ID
        spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
        
        # Try different cases
        refl1, _, _ = parser.get_spectrum(spectrum_id.upper())
        refl2, _, _ = parser.get_spectrum(spectrum_id.lower())
        refl3, _, _ = parser.get_spectrum(spectrum_id)
        
        # All should return same data
        assert len(refl1) == len(refl2) == len(refl3)
    
    def test_get_spectrum_invalid_id(self, parser):
        """Test that invalid spectrum ID raises error."""
        with pytest.raises(ValueError):
            parser.get_spectrum('INVALID_SPECTRUM_ID_12345')
    
    def test_get_spectra_batch_returns_tuple(self, parser):
        """Test that get_spectra_batch returns 3-tuple."""
        # Get first 3 spectrum IDs
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:3].tolist()
        result = parser.get_spectra_batch(spectrum_ids)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        refl_df, std_df, meta_df = result
        assert isinstance(refl_df, pd.DataFrame)
        assert isinstance(std_df, pd.DataFrame)
        assert isinstance(meta_df, pd.DataFrame)
    
    def test_get_spectra_batch_wide_format(self, parser):
        """Test that batch retrieval returns wide format."""
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:3].tolist()
        refl_df, std_df, meta_df = parser.get_spectra_batch(spectrum_ids)
        
        # Wavelength should be index
        assert 'wavelength' in str(refl_df.index.name).lower() or refl_df.index.name is None
        
        # Spectrum IDs should be columns
        assert len(refl_df.columns) <= len(spectrum_ids)
        
        # Same structure for std_dev
        assert refl_df.shape == std_df.shape
        
        # Metadata should have spectrum IDs as index
        assert len(meta_df) <= len(spectrum_ids)
    
    def test_get_spectra_batch_union_wavelengths(self, parser):
        """Test batch retrieval with union of wavelengths."""
        # Get spectra that might have different wavelength ranges
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:3].tolist()
        refl_df, std_df, meta_df = parser.get_spectra_batch(spectrum_ids)
        
        # Should have NaN where data is missing for some spectra
        # (unless all spectra have same wavelength grid)
        assert refl_df.shape[0] > 0  # Has wavelength points
        assert refl_df.shape[1] > 0  # Has spectrum columns
    
    def test_get_spectra_batch_with_interpolation(self, parser):
        """Test batch retrieval with custom wavelength grid."""
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:2].tolist()
        
        # Create custom wavelength grid
        wl_grid = np.arange(0.5, 1.0, 0.01)  # 0.5-1.0 μm with 10nm steps
        
        refl_df, std_df, meta_df = parser.get_spectra_batch(
            spectrum_ids,
            wavelength_grid=wl_grid,
            interpolation='linear'
        )
        
        # Check that wavelength grid matches
        assert len(refl_df) == len(wl_grid)
        np.testing.assert_array_almost_equal(refl_df.index.values, wl_grid)
        
        # Same for std_dev
        assert std_df.shape == refl_df.shape
    
    def test_interpolation_methods(self, parser):
        """Test different interpolation methods."""
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:1].tolist()
        wl_grid = np.arange(0.5, 1.0, 0.01)
        
        for method in ['linear', 'cubic', 'nearest']:
            refl_df, _, _ = parser.get_spectra_batch(
                spectrum_ids,
                wavelength_grid=wl_grid,
                interpolation=method
            )
            assert len(refl_df) == len(wl_grid)
    
    def test_interpolation_fills_nan_outside_bounds(self, parser):
        """Test that interpolation fills NaN outside original wavelength bounds."""
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:1].tolist()
        
        # Get original spectrum to find its bounds
        refl_orig, _, _ = parser.get_spectrum(spectrum_ids[0])
        wl_min = refl_orig.index.min()
        wl_max = refl_orig.index.max()
        
        # Create grid that extends beyond original bounds
        wl_grid = np.arange(wl_min - 0.1, wl_max + 0.1, 0.01)
        
        refl_df, _, _ = parser.get_spectra_batch(
            spectrum_ids,
            wavelength_grid=wl_grid,
            interpolation='linear'
        )
        
        # First and last values should be NaN (outside bounds)
        assert np.isnan(refl_df.iloc[0, 0])
        assert np.isnan(refl_df.iloc[-1, 0])
        
        # Middle values should not all be NaN
        assert not refl_df.iloc[len(refl_df)//2].isna().all()
    
    def test_metadata_contains_all_catalogue_fields(self, parser):
        """Test that metadata includes all Sample_Catalogue columns."""
        spectrum_ids = parser.spectra_catalogue['SpectrumID'].iloc[:2].tolist()
        _, _, meta_df = parser.get_spectra_batch(spectrum_ids)
        
        # Should have sample catalogue columns
        sample_cols = ['SampleID', 'SampleName', 'PI', 'Source', 'SubType']
        for col in sample_cols:
            if col in parser.sample_catalogue.columns:
                assert col in meta_df.columns
        
        # Should have spectra catalogue columns
        spectra_cols = ['SpectrumID', 'Date', 'SpecCode']
        for col in spectra_cols:
            if col in parser.spectra_catalogue.columns:
                assert col in meta_df.columns
    
    def test_missing_file_warning(self, parser):
        """Test that missing files generate warnings but don't crash."""
        # Create list with one invalid ID
        valid_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
        spectrum_ids = [valid_id, 'INVALID_ID_999']
        
        # Should warn but still return valid spectrum
        with pytest.warns(UserWarning):
            refl_df, std_df, meta_df = parser.get_spectra_batch(spectrum_ids)
        
        # Should have at least one valid spectrum
        assert len(refl_df.columns) >= 1
        assert len(meta_df) >= 1
    
    def test_parse_txt_file_format(self, parser):
        """Test parsing of .txt file format."""
        spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0]
        refl, std, _ = parser.get_spectrum(spectrum_id)
        
        # Should have positive wavelength values
        assert all(refl.index > 0)
        
        # Wavelengths should be in increasing order
        assert all(refl.index[1:] >= refl.index[:-1])
        
        # Reflectance values should be reasonable (0-1 range typically)
        # Some might be > 1 due to calibration, but check general range
        assert refl.min() >= 0
        assert refl.max() < 10  # Sanity check
    
    def test_construct_file_path(self, parser):
        """Test file path construction from spectrum ID."""
        spectrum_id = parser.spectra_catalogue['SpectrumID'].iloc[0].lower()
        file_path = parser._construct_file_path(spectrum_id)
        
        # File should exist
        assert file_path.exists()
        assert file_path.suffix == '.txt'
        
        # Path should follow pattern: data/{pi_code}/{subcode}/{spectrum_id}.txt
        assert file_path.parent.parent.parent.name == 'data'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
