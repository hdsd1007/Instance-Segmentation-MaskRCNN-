# Instance Segmentation with Mask R-CNN

A video instance segmentation project using PyTorch's pre-trained Mask R-CNN model. This implementation performs object detection and pixel-level segmentation on video streams, identifying and masking individual objects with colored overlays and bounding boxes.

## 🎯 Features

- **Video Processing**: Process videos frame by frame with instance segmentation
- **80 COCO Classes**: Detect and segment 80 different object classes from the COCO dataset
- **Colored Masks**: Each detected object gets a unique colored overlay with transparency
- **Bounding Box Annotations**: Green bounding boxes with confidence scores and class labels
- **GPU Acceleration**: CUDA support for faster inference
- **Letterbox Resizing**: Maintains aspect ratio while resizing for consistent processing

## 🚀 Demo

### Before vs After Processing

<table>
  <tr>
    <td align="center"><b>Original Video</b></td>
    <td align="center"><b>With Instance Segmentation</b></td>
  </tr>
  <tr>
    <td><img src="Sample Video/NightLife-Original.gif" width="300"/></td>
    <td><img src="Sample Video/NightLife_output_DistinctMask.gif" width="300"/></td>
  </tr>
</table>

### Before [Random Masks] vs After [Distinct Masks]

<table>
  <tr>
    <td align="center"><b>Random Mask</b></td>
    <td align="center"><b>With Distinct</b></td>
  </tr>
  <tr>
    <td><img src="Sample Video/NightLife2_output_RandomMask.gif" width="300"/></td>
    <td><img src="Sample Video/NightLife2_output_DistinctMask.gif" width="300"/></td>
  </tr>
</table>

## 📋 Requirements

### Dependencies
```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended) or CPU
- Minimum 4GB RAM (8GB+ recommended for GPU)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/instance-segmentation-maskrcnn.git
cd instance-segmentation-maskrcnn
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision opencv-python numpy Pillow
```

## 📁 Project Structure

```
instance-segmentation-maskrcnn/
│
├── InstanceSegmentation_MaskRCNN.ipynb    # Main Jupyter notebook
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── sample_videos/                         # Sample input videos
├── video_dir_input/                      # Input videos directory
└── video_dir_output/                     # Output videos directory
```

## 🎮 Usage

### Basic Usage

1. **Place your input video** in the `video_dir_input/` directory
2. **Update the video filename** in the notebook:
```python
INPUT_VIDEO = os.path.join(video_dir_input, 'your_video.mp4')
OUTPUT_VIDEO = os.path.join(video_dir_output, 'your_video_output.mp4')
```

3. **Configure parameters** (optional):
```python
CONF_THRESHOLD = 0.5    # Minimum confidence threshold
MASK_THRES = 0.5        # Minimum mask threshold
max_long_side = 1280    # Maximum resolution for processing
```

4. **Run the notebook** or execute the main processing loop

### Advanced Configuration

#### Adjusting Detection Parameters
- `CONF_THRESHOLD`: Lower values detect more objects but may include false positives
- `MASK_THRES`: Controls mask sensitivity for segmentation
- `max_long_side`: Higher values provide better quality but slower processing

#### Performance Optimization
- Use GPU for faster inference
- Adjust `max_long_side` based on your hardware capabilities
- Enable mixed precision for CUDA devices

## 📊 Model Information

### Mask R-CNN Architecture
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pre-trained on**: COCO dataset
- **Classes**: 80 object categories
- **Input**: RGB images of any size
- **Output**: Bounding boxes, class labels, confidence scores, and segmentation masks

### COCO Classes Supported
The model can detect 80 different object classes including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard
- **And many more...**

## 📹 Sample Videos

### Upload Your Own Videos

To test the model with your own videos:

1. **Supported Formats**: MP4, AVI, MOV, MKV
2. **Recommended Specifications**:
   - Resolution: 720p to 1080p
   - Frame rate: 24-30 FPS
   - Duration: 10-60 seconds for testing

3. **Upload Methods**:

#### Method 1: Direct Upload
```bash
# Copy your video to the input directory
cp your_video.mp4 video_dir_input/
```

#### Method 2: Google Colab Upload
If using Google Colab:
```python
from google.colab import files
uploaded = files.upload()
# Move uploaded file to video_dir_input/
```

#### Method 3: Google Drive (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy from your Google Drive
!cp "/content/drive/MyDrive/your_video.mp4" "video_dir_input/"
```

## 🔧 Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONF_THRESHOLD` | 0.5 | Minimum confidence for object detection |
| `MASK_THRES` | 0.5 | Threshold for binary mask generation |
| `max_long_side` | 1280 | Maximum dimension for input resizing |
| `alpha` | 0.5 | Transparency factor for mask overlay |

### Color Customization
Each object class gets a unique color. You can modify colors in the `COCO_COLORS` dictionary:
```python
COCO_COLORS = {
    idx: (R, G, B)  # Custom RGB values
    for idx in range(len(COCO_INSTANCE_CATEGORY_NAMES))
}
```

### Optimization Tips
1. Reduce `max_long_side` for faster processing
2. Use GPU acceleration when available
3. Process shorter video clips for testing
