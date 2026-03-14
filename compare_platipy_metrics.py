"""
Comparison Test Script: PlatiPy vs Custom Spatial Overlap Metrics

This script downloads the PlatiPy test dataset and computes metrics using both:
1. PlatiPy's built-in metric functions
2. Custom implementation from app/utils/spatial_overlap_metrics.py

The results are compared in a tabular format for quality assurance.
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import sys
import os

# Add the project root to the path to import custom modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import PlatiPy functions
try:
    from platipy.imaging.tests.data import download_and_extract_zip_file
    from platipy.imaging.label.comparison import (
        compute_metric_dsc,
        compute_metric_hd,
        compute_metric_masd,
        compute_surface_dsc,
        compute_surface_metrics,
        compute_volume_metrics,
        compute_metric_total_apl,
        compute_metric_mean_apl,
        compute_metric_sensitivity,
        compute_metric_specificity,
    )
except ImportError:
    print("PlatiPy not installed. Installing...")
    os.system("pip install platipy")
    from platipy.imaging.tests.data import download_and_extract_zip_file
    from platipy.imaging.label.comparison import (
        compute_metric_dsc,
        compute_metric_hd,
        compute_metric_masd,
        compute_surface_dsc,
        compute_surface_metrics,
        compute_volume_metrics,
        compute_metric_total_apl,
        compute_metric_mean_apl,
        compute_metric_sensitivity,
        compute_metric_specificity,
    )

# Import custom metrics
from app.utils.spatial_overlap_metrics import (
    dice_similarity,
    jaccard_similarity,
    hausdorff_distance_95,
    mean_surface_distance,
    surface_dsc,
    added_path_length,
    volume_overlap_error,
    variation_of_information,
    cosine_similarity,
    mean_distance_to_conformity,
    overcontouring_mean_distance_to_conformity,
    undercontouring_mean_distance_to_conformity,
)


def download_test_data():
    """Download PlatiPy test dataset."""
    data_path = Path("./platipy_test_data/contour_comparison_sample")
    test_data_zip_url = "https://zenodo.org/record/7519243/files/platipy_contour_comparison_testdata.zip?download=1"
    
    # Only download if not already present
    if not data_path.exists() or len(list(data_path.glob("*/*.nii.gz"))) == 0:
        print("Downloading PlatiPy test dataset...")
        download_and_extract_zip_file(test_data_zip_url, data_path)
        print(f"Data downloaded to: {data_path}")
    else:
        print(f"Using existing test data at: {data_path}")
    
    return data_path


def load_test_structures(data_path):
    """Load the test structures from the PlatiPy dataset."""
    structure_names = ["ESOPHAGUS", "HEART", "LUNG_L", "LUNG_R", "SPINALCORD"]
    
    manual_structures = {
        s: sitk.ReadImage(str(data_path.joinpath("manual", f"{s}.nii.gz")))
        for s in structure_names
    }
    
    auto_structures = {
        s: sitk.ReadImage(str(data_path.joinpath("auto", f"{s}.nii.gz")))
        for s in structure_names
    }
    
    return manual_structures, auto_structures, structure_names


def compute_platipy_metrics(label_a, label_b):
    """Compute metrics using PlatiPy functions."""
    metrics = {}
    
    # DSC (Dice Similarity Coefficient)
    metrics['DSC'] = compute_metric_dsc(label_a, label_b)
    
    # Hausdorff Distance (max)
    metrics['HD_max'] = compute_metric_hd(label_a, label_b)
    
    # Mean Absolute Surface Distance
    metrics['MASD'] = compute_metric_masd(label_a, label_b)
    
    # Surface DSC with tau=3mm (default)
    metrics['SurfaceDSC_3mm'] = compute_surface_dsc(label_a, label_b, tau=3)
    
    # Added Path Length
    try:
        metrics['APL_total'] = compute_metric_total_apl(label_a, label_b, distance_threshold_mm=3)
        metrics['APL_mean'] = compute_metric_mean_apl(label_a, label_b, distance_threshold_mm=3)
    except Exception as e:
        metrics['APL_total'] = None
        metrics['APL_mean'] = None
        print(f"  Warning: APL computation failed: {e}")
    
    # Surface metrics (includes HD95, MSD, etc.)
    try:
        surface_metrics = compute_surface_metrics(label_a, label_b, verbose=False)
        metrics['HD95'] = surface_metrics.get('hausdorffDistance95', None)
        metrics['MSD'] = surface_metrics.get('meanSurfaceDistance', None)
        metrics['MedianSD'] = surface_metrics.get('medianSurfaceDistance', None)
    except Exception as e:
        metrics['HD95'] = None
        metrics['MSD'] = None
        metrics['MedianSD'] = None
        print(f"  Warning: Surface metrics computation failed: {e}")
    
    # Volume metrics - PlatiPy doesn't compute Jaccard or VOE directly
    # volumeOverlap returns volume in cm³, not Jaccard coefficient
    # fractionOverlap is the actual Jaccard, but we don't compare it since custom has separate Jaccard
    try:
        volume_metrics = compute_volume_metrics(label_a, label_b)
        # Store for reference but don't use in comparison
        metrics['fractionOverlap'] = volume_metrics.get('fractionOverlap', None)
    except Exception as e:
        metrics['fractionOverlap'] = None
        print(f"  Warning: Volume metrics computation failed: {e}")
    
    return metrics


def compute_custom_metrics(label_a, label_b):
    """Compute metrics using custom implementation."""
    # Convert SimpleITK images to numpy arrays
    volume_a = sitk.GetArrayFromImage(label_a)
    volume_b = sitk.GetArrayFromImage(label_b)
    spacing = label_a.GetSpacing()
    
    metrics = {}
    
    # DSC (Dice Similarity Coefficient)
    metrics['DSC'] = dice_similarity(volume_a, volume_b)
    
    # Jaccard Similarity
    metrics['Jaccard'] = jaccard_similarity(volume_a, volume_b)
    
    # Hausdorff Distance 95th percentile
    metrics['HD95'] = hausdorff_distance_95(volume_a, volume_b)
    
    # Mean Surface Distance
    metrics['MSD'] = mean_surface_distance(volume_a, volume_b)
    
    # Surface DSC with tau=3mm
    metrics['SurfaceDSC_3mm'] = surface_dsc(volume_a, volume_b, tau=3.0, spacing=spacing)
    
    # Added Path Length
    try:
        metrics['APL_total'] = added_path_length(volume_a, volume_b, distance_threshold_mm=3, spacing=spacing)
    except Exception as e:
        metrics['APL_total'] = None
        print(f"  Warning: Custom APL computation failed: {e}")
    
    # Volume Overlap Error
    metrics['VOE'] = volume_overlap_error(volume_a, volume_b)
    
    # Variation of Information
    try:
        metrics['VI'] = variation_of_information(volume_a, volume_b)
    except Exception as e:
        metrics['VI'] = None
        print(f"  Warning: VI computation failed: {e}")
    
    # Cosine Similarity
    metrics['Cosine'] = cosine_similarity(volume_a, volume_b)
    
    # Mean Distance to Conformity
    try:
        metrics['MDC'] = mean_distance_to_conformity(volume_a, volume_b, spacing=spacing)
        metrics['OMDC'] = overcontouring_mean_distance_to_conformity(volume_a, volume_b, spacing=spacing)
        metrics['UMDC'] = undercontouring_mean_distance_to_conformity(volume_a, volume_b, spacing=spacing)
    except Exception as e:
        metrics['MDC'] = None
        metrics['OMDC'] = None
        metrics['UMDC'] = None
        print(f"  Warning: MDC computation failed: {e}")
    
    return metrics


def compare_metrics(structure_name, manual_label, auto_label):
    """Compare metrics from both implementations for a single structure pair."""
    print(f"\nProcessing: {structure_name}")
    
    # Compute metrics using both methods
    platipy_metrics = compute_platipy_metrics(manual_label, auto_label)
    custom_metrics = compute_custom_metrics(manual_label, auto_label)
    
    # Create comparison dictionary
    comparison = {
        'Structure': structure_name,
    }
    
    # Common metrics that can be directly compared
    # Note: Jaccard and VOE removed - PlatiPy doesn't compute these
    common_metrics = {
        'DSC': ('DSC', 'DSC'),
        'HD95': ('HD95', 'HD95'),
        'MSD': ('MSD', 'MSD'),
        'SurfaceDSC_3mm': ('SurfaceDSC_3mm', 'SurfaceDSC_3mm'),
        'APL_total': ('APL_total', 'APL_total'),
    }
    
    for metric_name, (platipy_key, custom_key) in common_metrics.items():
        platipy_val = platipy_metrics.get(platipy_key)
        custom_val = custom_metrics.get(custom_key)
        
        comparison[f'{metric_name}_PlatiPy'] = platipy_val
        comparison[f'{metric_name}_Custom'] = custom_val
        
        # Calculate difference if both values exist
        if platipy_val is not None and custom_val is not None:
            if not (np.isinf(platipy_val) or np.isinf(custom_val)):
                diff = abs(platipy_val - custom_val)
                comparison[f'{metric_name}_Diff'] = diff
            else:
                comparison[f'{metric_name}_Diff'] = None
        else:
            comparison[f'{metric_name}_Diff'] = None
    
    # Add custom-only metrics (not available in PlatiPy)
    comparison['Jaccard_Custom'] = custom_metrics.get('Jaccard')
    comparison['VOE_Custom'] = custom_metrics.get('VOE')
    comparison['VI_Custom'] = custom_metrics.get('VI')
    comparison['Cosine_Custom'] = custom_metrics.get('Cosine')
    comparison['MDC_Custom'] = custom_metrics.get('MDC')
    comparison['OMDC_Custom'] = custom_metrics.get('OMDC')
    comparison['UMDC_Custom'] = custom_metrics.get('UMDC')
    
    # Add PlatiPy-only metrics (not available in custom)
    comparison['HD_max_PlatiPy'] = platipy_metrics.get('HD_max')
    comparison['MASD_PlatiPy'] = platipy_metrics.get('MASD')
    comparison['APL_mean_PlatiPy'] = platipy_metrics.get('APL_mean')
    comparison['fractionOverlap_PlatiPy'] = platipy_metrics.get('fractionOverlap')  # This is PlatiPy's Jaccard
    
    return comparison


def create_summary_table(results):
    """Create a summary table showing key metrics comparison."""
    summary_data = []
    
    for result in results:
        structure = result['Structure']
        
        # Key metrics to summarize
        summary_row = {
            'Structure': structure,
            'DSC_PlatiPy': result.get('DSC_PlatiPy'),
            'DSC_Custom': result.get('DSC_Custom'),
            'DSC_Diff': result.get('DSC_Diff'),
            'HD95_PlatiPy': result.get('HD95_PlatiPy'),
            'HD95_Custom': result.get('HD95_Custom'),
            'HD95_Diff': result.get('HD95_Diff'),
            'SurfaceDSC_PlatiPy': result.get('SurfaceDSC_3mm_PlatiPy'),
            'SurfaceDSC_Custom': result.get('SurfaceDSC_3mm_Custom'),
            'SurfaceDSC_Diff': result.get('SurfaceDSC_3mm_Diff'),
        }
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def main():
    """Main function to run the comparison."""
    print("=" * 80)
    print("PlatiPy vs Custom Metrics Comparison")
    print("=" * 80)
    
    # Download and load test data
    data_path = download_test_data()
    manual_structures, auto_structures, structure_names = load_test_structures(data_path)
    
    print(f"\nLoaded {len(structure_names)} structure pairs: {', '.join(structure_names)}")
    
    # Compare metrics for each structure
    results = []
    for structure_name in structure_names:
        comparison = compare_metrics(
            structure_name,
            manual_structures[structure_name],
            auto_structures[structure_name]
        )
        results.append(comparison)
    
    # Create full DataFrame
    df_full = pd.DataFrame(results)
    
    # Create summary table
    df_summary = create_summary_table(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("SUMMARY: Key Metrics Comparison")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: All Metrics")
    print("=" * 80)
    
    # Display in sections for better readability
    print("\n--- Overlap Metrics ---")
    overlap_cols = ['Structure', 'DSC_PlatiPy', 'DSC_Custom', 'DSC_Diff']
    print(df_full[overlap_cols].to_string(index=False))
    
    print("\n--- Distance Metrics ---")
    distance_cols = ['Structure', 'HD95_PlatiPy', 'HD95_Custom', 'HD95_Diff',
                     'MSD_PlatiPy', 'MSD_Custom', 'MSD_Diff']
    print(df_full[distance_cols].to_string(index=False))
    
    print("\n--- Surface Metrics ---")
    surface_cols = ['Structure', 'SurfaceDSC_3mm_PlatiPy', 'SurfaceDSC_3mm_Custom', 'SurfaceDSC_3mm_Diff']
    print(df_full[surface_cols].to_string(index=False))
    
    print("\n--- Added Path Length ---")
    apl_cols = ['Structure', 'APL_total_PlatiPy', 'APL_total_Custom', 'APL_total_Diff', 'APL_mean_PlatiPy']
    print(df_full[apl_cols].to_string(index=False))
    
    print("\n--- Custom-Only Metrics ---")
    custom_only_cols = ['Structure', 'Jaccard_Custom', 'VOE_Custom', 'VI_Custom', 'Cosine_Custom', 'MDC_Custom', 'OMDC_Custom', 'UMDC_Custom']
    print(df_full[custom_only_cols].to_string(index=False))
    
    print("\n--- PlatiPy-Only Metrics ---")
    platipy_only_cols = ['Structure', 'HD_max_PlatiPy', 'MASD_PlatiPy', 'fractionOverlap_PlatiPy']
    print(df_full[platipy_only_cols].to_string(index=False))
    
    # Save to CSV
    output_file = "platipy_custom_metrics_comparison.csv"
    df_full.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Full results saved to: {output_file}")
    print(f"{'=' * 80}")
    
    # Calculate and display statistics on differences
    print("\n" + "=" * 80)
    print("DIFFERENCE STATISTICS")
    print("=" * 80)
    
    diff_cols = [col for col in df_full.columns if col.endswith('_Diff')]
    for col in diff_cols:
        metric_name = col.replace('_Diff', '')
        values = df_full[col].dropna()
        if len(values) > 0:
            print(f"\n{metric_name}:")
            print(f"  Mean Difference: {values.mean():.6f}")
            print(f"  Max Difference:  {values.max():.6f}")
            print(f"  Min Difference:  {values.min():.6f}")
            print(f"  Std Deviation:   {values.std():.6f}")
    
    return df_full


if __name__ == "__main__":
    df_results = main()
