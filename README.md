# COMET - Contour Metrics

COMET (Contour Metrics) is an open source application designed to provide a graphical user interface for computation of spatial overlap metrics between structures delineated in radiotherapy. This application is built using Django and provides users with a web interface to:
1. Upload a set of DICOM files with images and structureset files. 
2. Automatically processes the DICOM data to extract information on the regions of interest in the structureset files.
3. Converts the dicom structureset regions of interest into compressed nifti files.
4. Computes STAPLE contours from a set of contours. 
5. Computes spatial overlap metrics between arbitrary pair of contours as well as between the STAPLE contour.
6. Allows users to download the metrics for further downstream analysis.
7. **Visualize NIfTI images and ROI overlays** using two methods:
   - **WebGL Viewer (niivue)**: GPU-accelerated, instant loading, interactive navigation
   - **Matplotlib Viewer**: Traditional static image generation


## Key Features

### 🚀 Fast WebGL Visualization
- **Instant loading**: View NIfTI volumes in 2-5 seconds (vs 1-2 minutes with traditional methods)
- **GPU-accelerated**: Uses WebGL for smooth, real-time rendering
- **Interactive navigation**: Scroll through slices, adjust windowing on-the-fly
- **No pre-rendering**: Loads NIfTI files directly in browser, no PNG generation needed
- See [docs/NIIVUE_VISUALIZATION.md](docs/NIIVUE_VISUALIZATION.md) for details

#### Using the WebGL Viewer

1. **Access the Viewer**: From the NIfTI list page, click the "WebGL" button next to any image series
2. **Select ROIs**: Choose which structures to visualize from the left sidebar
3. **Load Visualization**: Click "Load Visualization" to render the volumes
4. **Navigation Modes**:
   - **Pan/Zoom Mode**: Scroll to zoom in/out, drag to pan around the image
   - **Slice Navigation Mode**: Scroll to navigate through slices, drag to adjust window/level
   - Toggle between modes using the "Mode" button in the sidebar
5. **Window/Level Adjustment**:
   - Use preset buttons (Soft Tissue, Brain, Bone, Lung, Liver) for quick adjustments
   - Fine-tune with manual sliders for Center and Width
6. **Slice Type**: Switch between Axial, Coronal, Sagittal, Multi-planar, or Mosaic views
7. **Overlay Controls**: Below the canvas, adjust individual overlay opacity or toggle structures on/off
8. **STAPLE Contours**: Automatically displayed in warm gold/orange tones when "Include STAPLE" is checked

#### Performance Considerations & Limitations

- **Optimal Performance**: Best with 1-10 overlays. Loading is fast and interactive.
- **Moderate Load (10-20 overlays)**: Still performs well, may take 5-10 seconds to load.
- **Heavy Load (20+ overlays)**: 
  - Loading time increases significantly (10-30 seconds)
  - May experience slower interaction and rendering
  - Browser memory usage increases
  - Recommended to select only the ROIs you need to visualize
- **Very Heavy Load (50+ overlays)**:
  - Not recommended - may cause browser slowdown or crashes
  - Consider using the Matplotlib viewer for static visualization instead
  - If needed, visualize in smaller batches
- **GPU Requirements**: Requires WebGL-capable graphics card. Older/integrated GPUs may have reduced performance.

### 📊 Spatial Overlap Metrics
- Compute Dice coefficient, Jaccard index, and other overlap metrics
- Compare multiple observer contours
- STAPLE consensus contour generation
- Export results for downstream analysis

# Installation

A docker image will be provided for running the application locally. Further instructions will be provided in the future here.