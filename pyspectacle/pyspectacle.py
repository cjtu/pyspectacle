"""
pyspectacle - Spectroscopy Data Browser

Parser and interactive browser for spectroscopy databases.
Provides methods to query catalogues and retrieve spectral data as pandas DataFrames.
"""

import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets


class DatabaseParser:
    """Abstract base class for spectroscopy database parsers.
    
    This class serves as a placeholder for future database parser implementations.
    Subclasses should implement methods for loading, querying, and retrieving
    spectral data from specific database formats.
    """
    pass


class RelabParser(DatabaseParser):
    """Parser for RELAB spectroscopy database.
    
    The parser loads Sample_Catalogue and Spectra_Catalogue on initialization
    and provides methods to list available spectra and retrieve spectral data
    with optional metadata joining.
    
    Parameters
    ----------
    db_path : str or Path
        Path to RelabDB2018Dec31 root directory
        
    Attributes
    ----------
    sample_catalogue : pd.DataFrame
        Loaded Sample_Catalogue with all sample metadata
    spectra_catalogue : pd.DataFrame
        Loaded Spectra_Catalogue with all spectrum metadata
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize parser and load catalogues.
        
        Parameters
        ----------
        db_path : str or Path
            Path to RELAB database root directory (e.g., RelabDB2018Dec31)
        """
        self.db_path = Path(db_path)
        self.data_path = self.db_path / "data"
        self.catalogue_path = self.db_path / "catalogues"
        print(f"Loading RELAB database from: {self.db_path.resolve()}")

        # Load catalogues with Latin-1 encoding (handles special characters)
        self.sample_catalogue = pd.read_csv(
            self.catalogue_path / "Sample_Catalogue.txt",
            sep="\t",
            low_memory=False,
            encoding='latin-1'
        )
        
        self.spectra_catalogue = pd.read_csv(
            self.catalogue_path / "Spectra_Catalogue.txt",
            sep="\t",
            low_memory=False,
            encoding='latin-1'
        )
        
        # Strip whitespace from all string columns
        for df in [self.sample_catalogue, self.spectra_catalogue]:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip()
        
        # Convert SpectrumID to lowercase for case-insensitive lookup
        self.spectra_catalogue['SpectrumID_lower'] = (
            self.spectra_catalogue['SpectrumID'].str.lower()
        )

        print(f"✓ Loaded {len(self.sample_catalogue):,} samples and {len(self.spectra_catalogue):,} spectra")
    
    def list_spectra(
        self,
        sample_id: Optional[str] = None,
        **filters
    ) -> pd.DataFrame:
        """List available spectra with metadata.
        
        Query the Spectra_Catalogue with optional filters and join with
        Sample_Catalogue to include all sample metadata.
        
        Parameters
        ----------
        sample_id : str, optional
            Filter by specific Sample ID
        **filters : dict
            Additional filters as column=value pairs
            (e.g., Source='Moon-Ret', SubType='Basalt')
            Filters support umbrella matching: 'Apollo 11' matches all values
            starting with 'Apollo 11'
            
        Returns
        -------
        pd.DataFrame
            Filtered spectra with joined metadata from both catalogues
            
        Examples
        --------
        >>> parser = RelabParser('data/spectra/RelabDB2018Dec31')
        >>> # List all lunar samples
        >>> lunar = parser.list_spectra(Source='Moon-Ret')
        >>> # List spectra for specific sample
        >>> sample_spectra = parser.list_spectra(sample_id='DD-MDD-035')
        """
        df = self.spectra_catalogue.copy()
        
        # Apply sample_id filter
        if sample_id is not None:
            df = df[df['SampleID'] == sample_id]
        
        # Apply additional filters with umbrella matching
        for key, value in filters.items():
            if key in df.columns:
                # Special handling for Atmosphere: exact match unless 'Ambient'
                if key == 'Atmosphere' and value != 'Ambient':
                    df = df[df[key] == value]
                else:
                    # Use umbrella matching: check if column value starts with filter value
                    df = df[df[key].str.startswith(value, na=False)]
            elif key in self.sample_catalogue.columns:
                # Filter will be applied after join
                pass
            else:
                warnings.warn(f"Filter key '{key}' not found in catalogues")
        
        # Join with Sample_Catalogue
        df = df.merge(
            self.sample_catalogue,
            on='SampleID',
            how='left'
        )
        
        # Apply filters that are in Sample_Catalogue with umbrella matching
        for key, value in filters.items():
            if key in self.sample_catalogue.columns and key not in self.spectra_catalogue.columns:
                # Special handling for Atmosphere: exact match unless 'Ambient'
                if key == 'Atmosphere' and value != 'Ambient':
                    df = df[df[key] == value]
                else:
                    df = df[df[key].str.startswith(value, na=False)]
        return df
    
    def get_spectrum(
        self,
        spectrum_id: str
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Retrieve a single spectrum with metadata.
        
        Parameters
        ----------
        spectrum_id : str
            Spectrum ID (case-insensitive)
            
        Returns
        -------
        reflectance : pd.Series
            Reflectance values indexed by wavelength (microns)
        std_dev : pd.Series
            Standard deviation values indexed by wavelength (microns)
            May be all NaN if not available
        metadata : pd.Series
            Metadata from joined Sample and Spectra catalogues
            
        Examples
        --------
        >>> parser = RelabParser('data/spectra/RelabDB2018Dec31')
        >>> refl, std, meta = parser.get_spectrum('C0AT02')
        >>> print(f"Sample: {meta['SampleName']}")
        >>> print(f"Wavelength range: {refl.index.min():.2f}-{refl.index.max():.2f} μm")
        """
        # Normalize to lowercase
        spectrum_id_lower = spectrum_id.lower()
        
        # Get file path
        file_path = self._construct_file_path(spectrum_id_lower)
        
        # Parse spectrum file
        wavelengths, reflectance, std_dev = self._parse_txt_file(file_path)
        
        # Create Series with wavelength as index
        refl_series = pd.Series(reflectance, index=wavelengths, name=spectrum_id)
        std_series = pd.Series(std_dev, index=wavelengths, name=spectrum_id)
        
        # Get metadata
        metadata = self._get_metadata(spectrum_id_lower)
        
        return refl_series, std_series, metadata
    
    def get_spectra_batch(
        self,
        spectrum_ids: List[str],
        wavelength_grid: Optional[np.ndarray] = None,
        interpolation: str = 'cubic'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve multiple spectra in wide format.
        
        Parameters
        ----------
        spectrum_ids : list of str
            List of Spectrum IDs (case-insensitive)
        wavelength_grid : np.ndarray, optional
            Target wavelength grid for interpolation (in microns).
            If None, uses union of all wavelengths with NaN for missing values.
            Example: np.arange(0.3, 3.0, 0.001) for 1nm resolution
        interpolation : str, default 'cubic'
            Interpolation method: 'linear', 'cubic', 'nearest'
            Only used if wavelength_grid is provided
            
        Returns
        -------
        reflectance_df : pd.DataFrame
            Wide format DataFrame with wavelength as index (rows),
            spectrum IDs as columns, reflectance values
        std_dev_df : pd.DataFrame
            Wide format DataFrame with wavelength as index (rows),
            spectrum IDs as columns, standard deviation values
        metadata_df : pd.DataFrame
            DataFrame with spectrum IDs as index, all catalogue
            metadata columns
            
        Examples
        --------
        >>> parser = RelabParser('data/spectra/RelabDB2018Dec31')
        >>> ids = ['C0AT02', 'C0AT03', 'C0AT04']
        >>> refl, std, meta = parser.get_spectra_batch(ids)
        >>> # With custom wavelength grid (1nm resolution)
        >>> wl_grid = np.arange(0.5, 2.5, 0.001)
        >>> refl, std, meta = parser.get_spectra_batch(ids, wavelength_grid=wl_grid)
        """
        # Normalize to lowercase
        spectrum_ids_lower = [sid.lower() for sid in spectrum_ids]
        
        # Storage for spectra
        spectra_dict = {}
        std_dict = {}
        metadata_list = []
        
        # Load all spectra
        for spectrum_id in spectrum_ids_lower:
            try:
                refl, std, meta = self.get_spectrum(spectrum_id)
                spectra_dict[spectrum_id] = refl
                std_dict[spectrum_id] = std
                metadata_list.append(meta)
            except Exception as e:
                warnings.warn(f"Failed to load spectrum '{spectrum_id}': {e}")
                continue
        
        if not spectra_dict:
            raise ValueError("No spectra could be loaded")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.index = list(spectra_dict.keys())
        metadata_df.index.name = 'SpectrumID'
        
        # Handle wavelength grid
        if wavelength_grid is not None:
            # Interpolate all spectra to target grid
            reflectance_df = pd.DataFrame(index=wavelength_grid)
            std_dev_df = pd.DataFrame(index=wavelength_grid)
            
            for spectrum_id, refl_series in spectra_dict.items():
                reflectance_df[spectrum_id] = self._interpolate_spectrum(
                    refl_series.index.values,
                    refl_series.values,
                    wavelength_grid,
                    interpolation
                )
                
                std_series = std_dict[spectrum_id]
                std_dev_df[spectrum_id] = self._interpolate_spectrum(
                    std_series.index.values,
                    std_series.values,
                    wavelength_grid,
                    interpolation
                )
        else:
            # Use union of all wavelengths
            # Combine all DataFrames with outer join (union of indices)
            reflectance_df = pd.DataFrame(spectra_dict)
            std_dev_df = pd.DataFrame(std_dict)
        
        # Set index name
        reflectance_df.index.name = 'wavelength_um'
        std_dev_df.index.name = 'wavelength_um'
        
        return reflectance_df, std_dev_df, metadata_df
    
    def _parse_txt_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse RELAB .txt spectrum file.
        
        Parameters
        ----------
        file_path : Path
            Path to .txt spectrum file
            
        Returns
        -------
        wavelengths : np.ndarray
            Wavelength values in microns
        reflectance : np.ndarray
            Reflectance values
        std_dev : np.ndarray
            Standard deviation values (NaN if not available)
        """
        # Read file, skip first 2 header lines
        df = pd.read_csv(file_path, sep='\t', skiprows=2, header=None)
        
        # Extract columns
        wavelengths = df[0].values
        reflectance = df[1].values
        
        # Standard deviation may or may not be present
        if df.shape[1] >= 3:
            std_dev = df[2].values
        else:
            std_dev = np.full_like(reflectance, np.nan)
        
        return wavelengths, reflectance, std_dev
    
    def _construct_file_path(self, spectrum_id_lower: str) -> Path:
        """Construct file path for a spectrum.
        
        Parameters
        ----------
        spectrum_id_lower : str
            Spectrum ID in lowercase
            
        Returns
        -------
        Path
            Full path to spectrum .txt file
        """
        # Look up spectrum in catalogue
        mask = self.spectra_catalogue['SpectrumID_lower'] == spectrum_id_lower
        
        if not mask.any():
            raise ValueError(f"Spectrum ID '{spectrum_id_lower}' not found in catalogue")
        
        row = self.spectra_catalogue[mask].iloc[0]
        sample_id = row['SampleID']
        
        # Extract PI code and subcode from Sample ID
        # Sample ID format: XY-PII-NNN[-ABC]
        # XY = subcode (first 2 letters)
        # PII = PI initials (characters after first -)
        parts = sample_id.split('-')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid Sample ID format: {sample_id}")
        
        subcode = parts[0].lower()  # First 2 letters
        pi_code = parts[1].lower()  # PI initials
        
        # Construct path: data/{pi_code}/{subcode}/{spectrum_id}.txt
        file_path = self.data_path / pi_code / subcode / f"{spectrum_id_lower}.txt"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Spectrum file not found: {file_path}")
        
        return file_path
    
    def _get_metadata(self, spectrum_id_lower: str) -> pd.Series:
        """Get metadata for a spectrum by joining catalogues.
        
        Parameters
        ----------
        spectrum_id_lower : str
            Spectrum ID in lowercase
            
        Returns
        -------
        pd.Series
            Metadata from joined Sample and Spectra catalogues
        """
        # Look up in Spectra_Catalogue
        mask = self.spectra_catalogue['SpectrumID_lower'] == spectrum_id_lower
        
        if not mask.any():
            raise ValueError(f"Spectrum ID '{spectrum_id_lower}' not found in catalogue")
        
        spectra_row = self.spectra_catalogue[mask].iloc[0]
        sample_id = spectra_row['SampleID']
        
        # Look up in Sample_Catalogue
        sample_mask = self.sample_catalogue['SampleID'] == sample_id
        
        if sample_mask.any():
            sample_row = self.sample_catalogue[sample_mask].iloc[0]
            # Merge both rows
            metadata = pd.concat([spectra_row, sample_row])
            # Remove duplicate SampleID
            metadata = metadata[~metadata.index.duplicated(keep='first')]
        else:
            metadata = spectra_row
        
        return metadata
    
    def _interpolate_spectrum(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        target_grid: np.ndarray,
        method: str = 'cubic'
    ) -> np.ndarray:
        """Interpolate spectrum to target wavelength grid.
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Original wavelength values
        values : np.ndarray
            Original spectrum values
        target_grid : np.ndarray
            Target wavelength grid
        method : str
            Interpolation method: 'linear', 'cubic', 'nearest'
            
        Returns
        -------
        np.ndarray
            Interpolated values on target grid, NaN outside original bounds
        """
        # Remove NaN values from input
        valid_mask = ~np.isnan(values)
        valid_wavelengths = wavelengths[valid_mask]
        valid_values = values[valid_mask]
        
        if len(valid_wavelengths) == 0:
            # All NaN input, return NaN output
            return np.full_like(target_grid, np.nan)
        
        # Create interpolator with bounds_error=False, fill_value=nan
        # This ensures no extrapolation outside original wavelength range
        interpolator = interp1d(
            valid_wavelengths,
            valid_values,
            kind=method,
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Interpolate to target grid
        interpolated_values = interpolator(target_grid)
        
        return interpolated_values


class SpectraBrowser:
    """Interactive GUI for viewing and exporting spectra.
    
    Provides an interactive widget-based interface for filtering, selecting,
    plotting, and exporting spectral data with metadata from spectroscopy databases.
    
    Features:
    - Filter spectra by Source, SubType, Origin, PI, Particulate
    - Search by Spectrum or Sample ID
    - Lock spectra to keep them plotted across filter changes
    - Configurable normalization wavelength
    - Dynamic plot updates without manual refresh
    - Wavelength range controls with presets
    - Export to CSV with customizable filename
    
    Parameters
    ----------
    parser : RelabParser
        Initialized RelabParser instance with loaded database
        
    Examples
    --------
    >>> from pyspectacle import RelabParser, SpectraBrowser
    >>> parser = RelabParser('data/spectra/RelabDB2018Dec31')
    >>> browser = SpectraBrowser(parser)
    >>> browser.display()
    """
    
    def __init__(self, parser: 'RelabParser'):
        """Initialize the interactive viewer.
        
        Parameters
        ----------
        parser : RelabParser
            Initialized RelabParser instance
        """
        self.parser = parser
        self.current_spectra = None
        self.filtered_df = None
        self.locked_spectra = set()  # Store locked spectrum IDs
        self.locked_data = {}  # Store locked spectrum data: {id: (refl, std, meta)}
        self.current_figure = None
        self.create_widgets()
        self.create_layout()
        self.update_filter()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Search
        self.search_id = widgets.Text(
            description='Search ID:',
            placeholder='Spectrum or Sample ID',
            style={'description_width': '120px'}
        )
        # Add observer for dynamic search
        self.search_id.observe(self.on_search_change, names='value')
        
        # Get all unique filter values
        all_sources = sorted(self.parser.sample_catalogue['Source'].dropna().unique().tolist())
        
        # Filter out broken/empty sources
        broken_sources = ['Eath', 'Other-Qet', 'Other-Vet', 'TE']
        valid_sources = [src for src in all_sources if src not in broken_sources]
        
        # Validate sources by checking if they return any results
        validated_sources = []
        for source in valid_sources:
            try:
                test_df = self.parser.list_spectra(Source=source)
                if len(test_df) > 0:
                    validated_sources.append(source)
            except Exception:
                pass
        
        sources = ['All'] + validated_sources
        subtypes = ['All'] + sorted(self.parser.sample_catalogue['SubType'].dropna().unique().tolist())
        origins = ['All'] + sorted(self.parser.sample_catalogue['Origin'].dropna().unique().tolist())
        
        # Get GeneralType1, Type1, and Atmosphere if columns exist
        if 'GeneralType1' in self.parser.sample_catalogue.columns:
            general_type1s = ['All'] + sorted(self.parser.sample_catalogue['GeneralType1'].dropna().unique().tolist())
        else:
            general_type1s = ['All']
        
        if 'Type1' in self.parser.sample_catalogue.columns:
            type1s = ['All'] + sorted(self.parser.sample_catalogue['Type1'].dropna().unique().tolist())
        else:
            type1s = ['All']
        
        if 'Atmosphere' in self.parser.sample_catalogue.columns:
            atmospheres = ['All'] + sorted(self.parser.sample_catalogue['Atmosphere'].dropna().unique().tolist())
        else:
            atmospheres = ['All']
        
        # Store original options for reset
        self.all_filter_options = {
            'Source': sources,
            'SubType': subtypes,
            'Origin': origins,
            'GeneralType1': general_type1s,
            'Type1': type1s,
            'Atmosphere': atmospheres
        }
        
        # Filters
        self.filter_source = widgets.Dropdown(
            options=sources, value='All', description='Source:',
            style={'description_width': '120px'}
        )
        self.filter_subtype = widgets.Dropdown(
            options=subtypes, value='All', description='SubType:',
            style={'description_width': '120px'}
        )
        self.filter_origin = widgets.Dropdown(
            options=origins, value='All', description='Origin:',
            style={'description_width': '120px'}
        )
        self.filter_general_type1 = widgets.Dropdown(
            options=general_type1s, value='All', description='GeneralType1:',
            style={'description_width': '120px'}
        )
        self.filter_type1 = widgets.Dropdown(
            options=type1s, value='All', description='Type1:',
            style={'description_width': '120px'}
        )
        self.filter_atmosphere = widgets.Dropdown(
            options=atmospheres, value='All', description='Atmosphere:',
            style={'description_width': '120px'}
        )
        
        # Add observers - Source is special and clears other filters
        self.filter_source.observe(self.on_source_change, names='value')
        
        # Other filters cascade normally
        self.filter_subtype.observe(self.on_filter_change, names='value')
        self.filter_origin.observe(self.on_filter_change, names='value')
        self.filter_general_type1.observe(self.on_filter_change, names='value')
        self.filter_type1.observe(self.on_filter_change, names='value')
        self.filter_atmosphere.observe(self.on_filter_change, names='value')
        
        self.clear_filters_button = widgets.Button(description='Clear All Filters', button_style='warning', icon='times')
        self.clear_filters_button.on_click(lambda b: self.clear_filters())
        
        # Selection
        self.spectrum_select = widgets.SelectMultiple(
            options=[], description='Spectra:', rows=10,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%')
        )
        # Add observer for dynamic updates
        self.spectrum_select.observe(self.on_selection_change, names='value')
        
        self.select_info = widgets.HTML(value='<i>No spectra filtered</i>')
        
        self.clear_selection_button = widgets.Button(
            description='Clear Selection', button_style='', icon='eraser',
            layout=widgets.Layout(width='auto')
        )
        self.clear_selection_button.on_click(lambda b: self.clear_selection())
        
        # Lock/Unlock buttons
        self.lock_button = widgets.Button(
            description='Lock Selected', button_style='info', icon='lock',
            layout=widgets.Layout(width='auto')
        )
        self.lock_button.on_click(lambda b: self.lock_selected())
        
        self.unlock_button = widgets.Button(
            description='Unlock All', button_style='warning', icon='unlock',
            layout=widgets.Layout(width='auto')
        )
        self.unlock_button.on_click(lambda b: self.unlock_all())
        
        # Individual unlock controls
        self.locked_list_output = widgets.Output(layout=widgets.Layout(max_height='150px', overflow_y='auto'))
        self.locked_info = widgets.HTML(value='<i>No locked spectra</i>')
        
        # Plot options
        self.show_errors = widgets.Checkbox(value=False, description='Show Error Bars')
        self.show_errors.observe(self.on_plot_option_change, names='value')
        
        self.offset_spectra = widgets.Checkbox(value=False, description='Offset Spectra (+0.5)')
        self.offset_spectra.observe(self.on_plot_option_change, names='value')
        
        # Normalization with configurable wavelength
        self.normalize = widgets.Checkbox(value=False, description='Normalize at:')
        self.normalize.observe(self.on_plot_option_change, names='value')
        
        self.normalize_wl = widgets.FloatText(
            value=0.75, min=0.0, max=50.0, step=0.01,
            description='',
            layout=widgets.Layout(width='100px')
        )
        self.normalize_wl.observe(self.on_plot_option_change, names='value')
        
        # Wavelength controls - text boxes and slider
        self.wl_min_text = widgets.FloatText(
            value=0.3, min=0.0, max=50.0, step=0.1,
            description='Min (μm):',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='150px')
        )
        self.wl_max_text = widgets.FloatText(
            value=3.0, min=0.0, max=50.0, step=0.1,
            description='Max (μm):',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='150px')
        )
        self.wl_range = widgets.FloatRangeSlider(
            value=[0.3, 3.0], min=0.3, max=50.0, step=0.1,
            description='Range:',
            continuous_update=False,
            style={'description_width': '70px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Add observers for dynamic updates
        self.wl_range.observe(self.on_plot_option_change, names='value')
        
        # Link text boxes and slider bidirectionally
        def update_slider_from_text(change):
            self.wl_range.value = [self.wl_min_text.value, self.wl_max_text.value]
        
        def update_text_from_slider(change):
            self.wl_min_text.value, self.wl_max_text.value = self.wl_range.value
        
        self.wl_min_text.observe(update_slider_from_text, names='value')
        self.wl_max_text.observe(update_slider_from_text, names='value')
        self.wl_range.observe(update_text_from_slider, names='value')
        
        # Wavelength preset buttons
        self.wl_preset_vnir = widgets.Button(description='VNIR (0.3-1.0)', layout=widgets.Layout(width='auto'))
        self.wl_preset_swir = widgets.Button(description='SWIR (1.0-3.0)', layout=widgets.Layout(width='auto'))
        self.wl_preset_vswir = widgets.Button(description='VSWIR (0.3-3.0)', layout=widgets.Layout(width='auto'))
        self.wl_preset_full = widgets.Button(description='Full (0.3-50)', layout=widgets.Layout(width='auto'))
        
        self.wl_preset_vnir.on_click(lambda b: self.set_wavelength_range(0.3, 1.0))
        self.wl_preset_swir.on_click(lambda b: self.set_wavelength_range(1.0, 3.0))
        self.wl_preset_vswir.on_click(lambda b: self.set_wavelength_range(0.3, 3.0))
        self.wl_preset_full.on_click(lambda b: self.set_wavelength_range(0.3, 50.0))
        
        # Metadata column selection - curated list of useful columns
        metadata_column_options = [
            'Atmosphere', 'AzimuthAngle', 'DetectAngle', 'GeneralType1', 'GeneralType2',
            'MinSize', 'MaxSize', 'Origin', 'Particulate', 'PhaseAngle', 'Resolution',
            'SampleID', 'SampleName', 'Source', 'SourceAngle', 'SpecCode', 'SpectrumID',
            'Start', 'Stop', 'SubType', 'Temperature', 'Text', 'Texture', 'Type1', 'Type2'
        ]
        
        # Filter to only include columns that exist in the catalogues
        available_cols = set(list(self.parser.sample_catalogue.columns) + 
                           list(self.parser.spectra_catalogue.columns))
        metadata_column_options = [col for col in metadata_column_options if col in available_cols]
        
        default_columns = ['SampleID', 'SpectrumID', 'SampleName', 'Source', 'Origin', 'Type1', 'SubType', 'GeneralType1']
        
        self.metadata_columns = widgets.SelectMultiple(
            options=metadata_column_options,
            value=[col for col in default_columns if col in metadata_column_options],
            description='Columns:',
            rows=8,
            style={'description_width': '70px'},
            layout=widgets.Layout(width='100%')
        )
        # Add observer for dynamic metadata update
        self.metadata_columns.observe(self.on_metadata_columns_change, names='value')
        
        # Restore default metadata columns button
        self.restore_defaults_button = widgets.Button(
            description='Restore Defaults',
            button_style='info',
            icon='undo',
            layout=widgets.Layout(width='auto')
        )
        self.restore_defaults_button.on_click(lambda b: self.restore_default_columns())
        self.default_columns = [col for col in default_columns if col in metadata_column_options]
        
        # Export controls
        self.export_filename = widgets.Text(
            value='spectra_export',
            description='Filename:',
            placeholder='Base filename (no extension)',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='100%')
        )
        self.export_button = widgets.Button(description='Export CSV', button_style='info', icon='download')
        self.export_button.on_click(lambda b: self.export_data())
        self.export_status = widgets.HTML(value='')
        
        # Save plot button
        self.save_plot_button = widgets.Button(description='Save Plot', button_style='info', icon='image')
        self.save_plot_button.on_click(lambda b: self.save_plot())
        
        # Wavelength range warning
        self.wl_warning = widgets.HTML(value='')
        
        # Outputs
        self.metadata_output = widgets.Output()
        self.plot_output = widgets.Output()
    
    def create_layout(self):
        """Create the main layout structure."""
        left_panel = widgets.VBox([
            widgets.HTML('<h3>Search & Filter</h3>'),
            self.search_id,
            self.filter_source,
            self.filter_subtype,
            self.filter_origin,
            self.filter_general_type1,
            self.filter_type1,
            self.filter_atmosphere,
            self.clear_filters_button,
            self.select_info,
            widgets.HTML('<h3>Select Spectra</h3>'),
            self.spectrum_select,
            widgets.HTML('<i>Hold Ctrl/Cmd for multiple</i>'),
            self.clear_selection_button,
            widgets.HTML('<h3>Lock Spectra</h3>'),
            widgets.HBox([self.lock_button, self.unlock_button]),
            self.locked_info,
            self.locked_list_output,
            widgets.HTML('<h3>Plot Options</h3>'),
            self.show_errors,
            self.offset_spectra,
            widgets.HBox([self.normalize, self.normalize_wl, widgets.HTML('<i>μm</i>')]),
            widgets.HTML('<b>Wavelength Range:</b>'),
            widgets.HBox([self.wl_min_text, self.wl_max_text]),
            self.wl_range,
            self.wl_warning,
            widgets.HTML('<i>Presets:</i>'),
            widgets.HBox([self.wl_preset_vnir, self.wl_preset_swir]),
            widgets.HBox([self.wl_preset_vswir, self.wl_preset_full]),
            widgets.HTML('<h3>Export</h3>'),
            self.export_filename,
            widgets.HBox([self.export_button, self.save_plot_button]),
            self.export_status
        ], layout=widgets.Layout(width='420px', padding='10px'))
        
        right_panel = widgets.VBox([
            widgets.HTML('<h3>Metadata Display</h3>'),
            widgets.HTML('<i>Select columns to display (hold Ctrl/Cmd for multiple):</i>'),
            self.metadata_columns,
            self.restore_defaults_button,
            self.metadata_output,
            widgets.HTML('<h3>Spectrum Plot</h3>'),
            self.plot_output
        ], layout=widgets.Layout(flex='1', padding='10px'))
        
        self.main_layout = widgets.HBox([left_panel, right_panel])
    
    def on_source_change(self, change):
        """Handle Source filter changes - clear other filters and update options."""
        # Store flag to prevent recursive updates
        if hasattr(self, '_updating_filters') and self._updating_filters:
            return
        
        self._updating_filters = True
        
        try:
            # Temporarily unobserve other filters
            for widget in [self.filter_subtype, self.filter_origin, self.filter_general_type1,
                           self.filter_type1, self.filter_atmosphere]:
                widget.unobserve(self.on_filter_change, names='value')
            
            # Reset other filters to 'All'
            self.filter_subtype.value = 'All'
            self.filter_origin.value = 'All'
            self.filter_general_type1.value = 'All'
            self.filter_type1.value = 'All'
            self.filter_atmosphere.value = 'All'
            
            # Update available options based on selected source
            if self.filter_source.value != 'All':
                filters = {'Source': self.filter_source.value}
                temp_df = self.parser.list_spectra(**filters)
                
                # Update options for each filter
                for filter_name, filter_widget in [
                    ('SubType', self.filter_subtype),
                    ('Origin', self.filter_origin),
                    ('GeneralType1', self.filter_general_type1),
                    ('Type1', self.filter_type1),
                    ('Atmosphere', self.filter_atmosphere)
                ]:
                    if filter_name in temp_df.columns:
                        available_values = ['All'] + sorted(temp_df[filter_name].dropna().unique().tolist())
                        filter_widget.options = available_values
            else:
                # Reset to original options when 'All' is selected
                self.filter_subtype.options = self.all_filter_options['SubType']
                self.filter_origin.options = self.all_filter_options['Origin']
                self.filter_general_type1.options = self.all_filter_options['GeneralType1']
                self.filter_type1.options = self.all_filter_options['Type1']
                self.filter_atmosphere.options = self.all_filter_options['Atmosphere']
            
            # Re-observe other filters
            for widget in [self.filter_subtype, self.filter_origin, self.filter_general_type1,
                           self.filter_type1, self.filter_atmosphere]:
                widget.observe(self.on_filter_change, names='value')
            
            # Update spectra list
            self.update_filter()
        finally:
            self._updating_filters = False
    
    def on_filter_change(self, change):
        """Update filter options when any filter changes (cascading filters)."""
        # Prevent recursive updates
        if hasattr(self, '_updating_filters') and self._updating_filters:
            return
        
        self._updating_filters = True
        
        try:
            # Get current filter values
            filters = {}
            if self.filter_source.value != 'All':
                filters['Source'] = self.filter_source.value
            if self.filter_subtype.value != 'All':
                filters['SubType'] = self.filter_subtype.value
            if self.filter_origin.value != 'All':
                filters['Origin'] = self.filter_origin.value
            if 'GeneralType1' in self.parser.sample_catalogue.columns and self.filter_general_type1.value != 'All':
                filters['GeneralType1'] = self.filter_general_type1.value
            if 'Type1' in self.parser.sample_catalogue.columns and self.filter_type1.value != 'All':
                filters['Type1'] = self.filter_type1.value
            if 'Atmosphere' in self.parser.sample_catalogue.columns and self.filter_atmosphere.value != 'All':
                filters['Atmosphere'] = self.filter_atmosphere.value
            
            # Get filtered data
            if filters:
                temp_df = self.parser.list_spectra(**filters)
            else:
                temp_df = self.parser.list_spectra()
            
            # Update options for filters that come after the changed one
            # Determine which filter changed
            changed_widget = change['owner']
            filter_order = [
                ('Source', self.filter_source),
                ('SubType', self.filter_subtype),
                ('Origin', self.filter_origin),
                ('GeneralType1', self.filter_general_type1),
                ('Type1', self.filter_type1),
                ('Atmosphere', self.filter_atmosphere)
            ]
            
            # Find index of changed filter
            changed_idx = next((i for i, (name, widget) in enumerate(filter_order) if widget == changed_widget), -1)
            
            # Update options for subsequent filters
            for i, (filter_name, filter_widget) in enumerate(filter_order):
                if i > changed_idx and filter_name in temp_df.columns:
                    # Temporarily unobserve
                    filter_widget.unobserve(self.on_filter_change, names='value')
                    
                    available_values = ['All'] + sorted(temp_df[filter_name].dropna().unique().tolist())
                    current_value = filter_widget.value
                    
                    # Update options
                    filter_widget.options = available_values
                    
                    # Restore value if still available
                    if current_value not in available_values:
                        filter_widget.value = 'All'
                    
                    # Re-observe
                    filter_widget.observe(self.on_filter_change, names='value')
            
            # Update spectra list
            self.update_filter()
        finally:
            self._updating_filters = False
    
    def on_search_change(self, change):
        """Update filtered spectra when search text changes."""
        self.update_filter()
    
    def clear_filters(self):
        """Reset all filters to 'All'."""
        # Set flag to prevent cascading updates
        self._updating_filters = True
        
        try:
            # Unobserve all filters
            self.filter_source.unobserve(self.on_source_change, names='value')
            for widget in [self.filter_subtype, self.filter_origin, self.filter_general_type1,
                           self.filter_type1, self.filter_atmosphere]:
                widget.unobserve(self.on_filter_change, names='value')
            
            # Reset all to original options and 'All'
            self.filter_source.options = self.all_filter_options['Source']
            self.filter_source.value = 'All'
            self.filter_subtype.options = self.all_filter_options['SubType']
            self.filter_subtype.value = 'All'
            self.filter_origin.options = self.all_filter_options['Origin']
            self.filter_origin.value = 'All'
            self.filter_general_type1.options = self.all_filter_options['GeneralType1']
            self.filter_general_type1.value = 'All'
            self.filter_type1.options = self.all_filter_options['Type1']
            self.filter_type1.value = 'All'
            self.filter_atmosphere.options = self.all_filter_options['Atmosphere']
            self.filter_atmosphere.value = 'All'
            
            # Clear search
            self.search_id.unobserve(self.on_search_change, names='value')
            self.search_id.value = ''
            self.search_id.observe(self.on_search_change, names='value')
            
            # Re-observe with correct handlers
            self.filter_source.observe(self.on_source_change, names='value')
            for widget in [self.filter_subtype, self.filter_origin, self.filter_general_type1,
                           self.filter_type1, self.filter_atmosphere]:
                widget.observe(self.on_filter_change, names='value')
        finally:
            self._updating_filters = False
        
        # Update the filtered data
        self.update_filter()
    
    def clear_selection(self):
        """Clear all selected spectra."""
        self.spectrum_select.value = []
    
    def restore_default_columns(self):
        """Restore metadata display to default columns."""
        self.metadata_columns.value = tuple(self.default_columns)
        self.update_metadata_display()

    def lock_selected(self):
        """Lock currently selected spectra to keep them plotted."""
        selected = list(self.spectrum_select.value)
        if not selected:
            with self.locked_list_output:
                print("⚠ No spectra selected to lock")
            return
        
        # Load and store locked spectra data
        for spectrum_id in selected:
            if spectrum_id not in self.locked_spectra:
                try:
                    refl, std, meta = self.parser.get_spectrum(spectrum_id)
                    self.locked_data[spectrum_id] = (refl, std, meta)
                    self.locked_spectra.add(spectrum_id)
                except Exception as e:
                    warnings.warn(f"Failed to lock spectrum '{spectrum_id}': {e}")
        
        self.update_locked_info()
        self.update_plot()
    
    def unlock_all(self):
        """Unlock all locked spectra."""
        self.locked_spectra.clear()
        self.locked_data.clear()
        self.update_locked_info()
        self.update_plot()
    
    def unlock_single(self, spectrum_id: str):
        """Unlock a single spectrum.
        
        Parameters
        ----------
        spectrum_id : str
            Spectrum ID to unlock
        """
        if spectrum_id in self.locked_spectra:
            self.locked_spectra.remove(spectrum_id)
            self.locked_data.pop(spectrum_id, None)
            self.update_locked_info()
            self.update_plot()
    
    def update_locked_info(self):
        """Update the locked spectra info display with individual unlock buttons."""
        if self.locked_spectra:
            count = len(self.locked_spectra)
            self.locked_info.value = f'<b>{count} locked spectrum/spectra</b>'
            
            # Create individual unlock buttons
            with self.locked_list_output:
                self.locked_list_output.clear_output()
                for spectrum_id in sorted(self.locked_spectra):
                    # Create unlock button for this spectrum
                    unlock_btn = widgets.Button(
                        description='✕',
                        button_style='',
                        tooltip=f'Unlock {spectrum_id}',
                        layout=widgets.Layout(width='30px', height='28px')
                    )
                    # Use lambda with default argument to capture spectrum_id
                    unlock_btn.on_click(lambda b, sid=spectrum_id: self.unlock_single(sid))
                    
                    label = widgets.HTML(
                        value=f'<code>{spectrum_id}</code>',
                        layout=widgets.Layout(flex='1')
                    )
                    
                    row = widgets.HBox([unlock_btn, label], layout=widgets.Layout(margin='2px 0'))
                    display(row)
        else:
            self.locked_info.value = '<i>No locked spectra</i>'
            with self.locked_list_output:
                self.locked_list_output.clear_output()
    
    def set_wavelength_range(self, wl_min: float, wl_max: float):
        """Set wavelength range from preset buttons.
        
        Parameters
        ----------
        wl_min : float
            Minimum wavelength in microns
        wl_max : float
            Maximum wavelength in microns
        """
        self.wl_min_text.value = wl_min
        self.wl_max_text.value = wl_max
    
    def update_filter(self):
        """Update filtered spectra based on current filter settings."""
        filters = {}
        if self.filter_source.value != 'All':
            filters['Source'] = self.filter_source.value
        if self.filter_subtype.value != 'All':
            filters['SubType'] = self.filter_subtype.value
        if self.filter_origin.value != 'All':
            filters['Origin'] = self.filter_origin.value
        if 'GeneralType1' in self.parser.sample_catalogue.columns and self.filter_general_type1.value != 'All':
            filters['GeneralType1'] = self.filter_general_type1.value
        if 'Type1' in self.parser.sample_catalogue.columns and self.filter_type1.value != 'All':
            filters['Type1'] = self.filter_type1.value
        if 'Atmosphere' in self.parser.sample_catalogue.columns and self.filter_atmosphere.value != 'All':
            filters['Atmosphere'] = self.filter_atmosphere.value
        
        search_text = self.search_id.value.strip()
        if search_text:
            spectrum_match = self.parser.spectra_catalogue[
                self.parser.spectra_catalogue['SpectrumID'].str.contains(search_text, case=False, na=False)]
            sample_match = self.parser.sample_catalogue[
                self.parser.sample_catalogue['SampleID'].str.contains(search_text, case=False, na=False)]
            
            if len(spectrum_match) > 0:
                self.filtered_df = self.parser.list_spectra(**filters)
                self.filtered_df = self.filtered_df[self.filtered_df['SpectrumID'].isin(spectrum_match['SpectrumID'])]
            elif len(sample_match) > 0:
                sample_ids = sample_match['SampleID'].tolist()
                self.filtered_df = pd.concat([self.parser.list_spectra(sample_id=sid, **filters) for sid in sample_ids])
            else:
                self.filtered_df = pd.DataFrame()
        else:
            self.filtered_df = self.parser.list_spectra(**filters)
        
        if len(self.filtered_df) > 0:
            options = [
                (
                    f"{row['SpectrumID']} - {str(row['SampleName'])[:50] if pd.notnull(row['SampleName']) else 'Unknown'}",
                    row['SpectrumID']
                )
                for _, row in self.filtered_df.iterrows()
            ]
            self.spectrum_select.options = options
            self.select_info.value = f'<b>{len(self.filtered_df)} spectra found</b>'
        else:
            self.spectrum_select.options = []
            self.select_info.value = '<i>No spectra match filters</i>'
    
    def update_plot(self):
        """Update the plot with current selection and locked spectra."""
        # Combine selected and locked spectrum IDs
        selected = list(self.spectrum_select.value)
        all_spectrum_ids = list(set(selected + list(self.locked_spectra)))
        
        if not all_spectrum_ids:
            # Clear both outputs when no spectra
            with self.plot_output:
                self.plot_output.clear_output()
            with self.metadata_output:
                self.metadata_output.clear_output()
            self.wl_warning.value = ''
            return
        
        # Use clear_output(wait=True) to prevent duplication
        self.plot_output.clear_output(wait=True)
        
        with self.plot_output:
            try:
                # Load spectra data
                refl_dict = {}
                std_dict = {}
                meta_list = []
                
                for spectrum_id in all_spectrum_ids:
                    # Use locked data if available, otherwise load
                    if spectrum_id in self.locked_data:
                        refl, std, meta = self.locked_data[spectrum_id]
                    else:
                        refl, std, meta = self.parser.get_spectrum(spectrum_id)
                    
                    refl_dict[spectrum_id] = refl
                    std_dict[spectrum_id] = std
                    meta_list.append(meta)
                
                # Create DataFrames
                refl_df = pd.DataFrame(refl_dict)
                std_df = pd.DataFrame(std_dict)
                meta_df = pd.DataFrame(meta_list)
                meta_df.index = list(refl_dict.keys())
                meta_df.index.name = 'SpectrumID'
                
                # Add lock status column to metadata (use emoji for table)
                meta_df['Locked'] = meta_df.index.map(lambda x: '🔒' if x in self.locked_spectra else '')
                
                self.current_spectra = (refl_df, std_df, meta_df)
                
                # Update wavelength slider bounds based on actual data
                wl_data_min = refl_df.index.min()
                wl_data_max = refl_df.index.max()
                self.wl_range.min = max(0.0, wl_data_min - 0.1)
                self.wl_range.max = min(50.0, wl_data_max + 0.1)
                
                # Update metadata display
                self.update_metadata_display()
                
                # Close any existing figures to prevent duplication
                plt.close('all')
                
                # Check wavelength range validity
                wl_min, wl_max = self.wl_range.value
                out_of_range_spectra = []
                
                for col in refl_df.columns:
                    col_wl_min = refl_df[col].dropna().index.min()
                    col_wl_max = refl_df[col].dropna().index.max()
                    
                    # Check if requested range is outside spectrum range
                    if wl_min > col_wl_max or wl_max < col_wl_min:
                        out_of_range_spectra.append(col)
                
                # Show warning if spectra are out of range
                if out_of_range_spectra:
                    spectra_list = ', '.join(out_of_range_spectra[:3])
                    if len(out_of_range_spectra) > 3:
                        spectra_list += f' and {len(out_of_range_spectra) - 3} more'
                    self.wl_warning.value = (
                        f'<div style="color: orange; padding: 5px; border: 1px solid orange; '
                        f'border-radius: 3px; margin: 5px 0;">'
                        f'⚠ Out of range: {spectra_list}</div>'
                    )
                else:
                    self.wl_warning.value = ''
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 7))
                colors = plt.cm.tab10(np.linspace(0, 1, len(all_spectrum_ids)))
                
                for i, col in enumerate(refl_df.columns):
                    wl = refl_df.index.values
                    refl = refl_df[col].values
                    std = std_df[col].values if self.show_errors.value else None
                    
                    mask = (wl >= wl_min) & (wl <= wl_max)
                    wl, refl = wl[mask], refl[mask]
                    if std is not None:
                        std = std[mask]
                    
                    # Skip if no valid data in range
                    if len(wl) == 0 or np.all(np.isnan(refl)):
                        continue
                    
                    if self.normalize.value:
                        norm_wl = self.normalize_wl.value
                        idx_ref = np.argmin(np.abs(wl - norm_wl))
                        if idx_ref < len(refl) and not np.isnan(refl[idx_ref]) and refl[idx_ref] != 0:
                            refl = refl / refl[idx_ref]
                            if std is not None:
                                std = std / refl[idx_ref]
                    
                    offset = i * 0.5 if self.offset_spectra.value else 0
                    refl = refl + offset
                    
                    # Don't add lock indicator to plot label
                    label = f"{col}: {meta_df.loc[col, 'SampleName'][:30]}"
                    
                    if self.show_errors.value and std is not None and not np.all(np.isnan(std)):
                        ax.errorbar(wl, refl, yerr=std, label=label, color=colors[i],
                                   alpha=0.7, capsize=2, errorevery=50)
                    else:
                        ax.plot(wl, refl, label=label, color=colors[i], linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Wavelength (μm)', fontsize=12, fontweight='bold')
                if self.normalize.value:
                    ylabel = f'Reflectance (normalized at {self.normalize_wl.value} μm)'
                else:
                    ylabel = 'Reflectance'
                ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                ax.set_title('Spectra', fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(alpha=0.3)
                
                # Only set xlim if there's valid data to plot
                if len(ax.lines) > 0 or len(ax.collections) > 0:
                    ax.set_xlim(self.wl_range.value)
                
                plt.tight_layout()
                
                # Store figure for saving
                self.current_figure = fig
                
                plt.show()
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def update_metadata_display(self):
        """Update metadata table with selected columns."""
        if self.current_spectra is None:
            return
        
        _, _, meta_df = self.current_spectra
        
        # Display metadata with selected columns
        selected_cols = [col for col in self.metadata_columns.value if col in meta_df.columns]
        
        # Always include lock status if it exists
        if 'Locked' in meta_df.columns:
            display_cols = ['Locked'] + [c for c in selected_cols if c != 'Locked']
        else:
            display_cols = selected_cols
        
        with self.metadata_output:
            self.metadata_output.clear_output()
            if display_cols:
                display(meta_df[display_cols])
            else:
                print("⚠ No valid columns selected")
    
    def save_plot(self):
        """Save the current plot to file."""
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            with self.plot_output:
                print("⚠ No plot to save. Plot spectra first.")
            return
        
        try:
            basename = self.export_filename.value.strip() or 'spectra_export'
            filename = f'{basename}_plot.png'
            self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
            
            with self.plot_output:
                print(f"✓ Plot saved to '{filename}'")
        except Exception as e:
            with self.plot_output:
                print(f"❌ Error saving plot: {e}")
    
    def export_data(self):
        """Export current spectra and metadata to CSV files."""
        if self.current_spectra is None:
            self.export_status.value = '<span style="color: orange;">⚠ No spectra plotted yet</span>'
            return
        
        refl_df, std_df, meta_df = self.current_spectra
        
        # Get selected metadata columns (excluding lock status for export)
        selected_cols = [col for col in self.metadata_columns.value if col in meta_df.columns]
        if selected_cols:
            meta_export = meta_df[selected_cols]
        else:
            meta_export = meta_df.drop(columns=['Locked'], errors='ignore')
        
        # Use custom basename
        basename = self.export_filename.value.strip() or 'spectra_export'
        
        try:
            refl_filename = f'{basename}_reflectance.csv'
            std_filename = f'{basename}_std_dev.csv'
            meta_filename = f'{basename}_metadata.csv'
            
            refl_df.to_csv(refl_filename)
            std_df.to_csv(std_filename)
            meta_export.to_csv(meta_filename)
            
            self.export_status.value = f'''
                <div style="color: green; font-weight: bold; padding: 5px; border: 2px solid green; border-radius: 5px; margin-top: 5px;">
                ✓ Successfully exported:<br>
                • {refl_filename}<br>
                • {std_filename}<br>
                • {meta_filename}
                </div>
            '''
            
            # Clear status after 10 seconds
            import threading
            def clear_status():
                import time
                time.sleep(10)
                self.export_status.value = ''
            
            threading.Thread(target=clear_status, daemon=True).start()
            
        except Exception as e:
            self.export_status.value = f'<span style="color: red;">❌ Error: {e}</span>'
    
    def display(self):
        """Display the interactive viewer widget."""
        display(self.main_layout)
    
    def on_selection_change(self, change: dict) -> None:
        """Handle spectrum selection changes."""
        # Update the plot when selection changes
        self.update_plot()
    
    def on_plot_option_change(self, change: dict) -> None:
        """Handle plot option changes (normalize, error bars, etc.)."""
        self.update_plot()

    def on_metadata_columns_change(self, change: dict) -> None:
        """Handle plot option changes (normalize, error bars, etc.)."""
        self.update_plot()