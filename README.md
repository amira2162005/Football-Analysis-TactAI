# Football Analysis System

An advanced computer vision system for analyzing football matches using YOLO object detection, multi-object tracking, and team classification. This system provides real-time player tracking, team assignment, ball control analysis, and comprehensive match statistics.

## 🚀 Features

- **Multi-Object Detection & Tracking**: Track players, referees, and ball using YOLOv8 and ByteTrack
- **Automatic Team Assignment**: AI-powered team classification based on jersey colors
- **Ball Control Analysis**: Real-time calculation of team possession statistics
- **Enhanced Visualizations**: Professional-grade annotations with team colors and statistics
- **Temporal Consistency**: Advanced algorithms to reduce tracking jitter and team assignment flickering
- **Performance Optimization**: Batch processing and intelligent caching for faster analysis

## 📋 Requirements

### Dependencies
```bash
ultralytics>=8.0.0
supervision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
pickle-mixin
```

### System Requirements
- Python 3.8+
- GPU recommended (CUDA-compatible) for real-time processing
- Minimum 8GB RAM (16GB recommended)
- 2GB free disk space for models and cache

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mohamed20384/Football-Analysis-TactAI
cd football-analysis
```

2. **Create virtual environment**
```bash
python -m venv football_env
source football_env/bin/activate  # On Windows: football_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLO model**
   - Place your trained YOLO model (`best.pt`) in the `models/` directory
   - Ensure the model is trained to detect: `player`, `goalkeeper`, `referee`, `ball`

## 📁 Project Structure

```
football-analysis/
│
├── main.py                 # Main execution script
├── utils.py                # Utility functions for video and geometry
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── trackers/
│   └── tracker.py         # Enhanced tracking system
│
├── team_assigner.py       # AI team classification system
│
├── models/
│   └── best.pt           # YOLO model weights
│
├── input_videos/
│   └── CV_Task.mkv       # Input football videos
│
├── output_videos/
│   └── (generated outputs)
│
└── cache/
    └── (tracking cache files)
```

## 🎯 Usage

### Basic Usage

```bash
python main.py
```

### Advanced Configuration

Modify parameters in `main.py`:

```python
# Video settings
video_path = "input_videos/your_video.mp4"
output_path = "output_videos/analyzed_video.mp4"

# Tracking parameters
tracker = Tracker("models/best.pt")
tracks = tracker.get_object_tracks(
    video_frames, 
    read_from_stub=True,    # Enable caching
    stub_path="cache/tracks.pkl"
)
```

### Custom Team Colors

```python
# Manual team color override (optional)
team_assigner = TeamAssigner()
team_assigner.team_colors = {
    1: np.array([255, 0, 0]),    # Red team
    2: np.array([0, 0, 255])     # Blue team
}
```

## ⚙️ Configuration Options

### Tracker Settings
- **Confidence threshold**: `conf=0.15` (adjust in `detect_frames()`)
- **IoU threshold**: `iou=0.5` (for non-max suppression)
- **Batch size**: `batch_size=20` (optimize based on GPU memory)

### ByteTracker Parameters
- **Track threshold**: `track_thresh=0.25`
- **Track buffer**: `track_buffer=50` (frames to keep lost tracks)
- **Match threshold**: `match_thresh=0.8`

### Team Assignment
- **Sampling frames**: Number of frames for initial color detection
- **Temporal window**: `window_size=7` (for consistency smoothing)
- **Color confidence**: Minimum threshold for team assignment confidence

## 📊 Output Features

### Visual Annotations
- **Player Tracking**: Colored ellipses with unique IDs
- **Team Colors**: Automatic jersey color detection and assignment
- **Ball Tracking**: Green triangle marker with smooth trajectory
- **Referee Detection**: Distinct cyan highlighting
- **Ball Possession Indicator**: Red triangle above player with ball

### Statistics Display
- **Real-time Ball Control**: Percentage display with progress bars
- **Team Possession**: Cumulative statistics throughout the match
- **Console Output**: Match summary with final statistics

### Generated Files
- **Annotated Video**: Complete match analysis with overlays
- **Tracking Cache**: Serialized tracking data for quick re-processing
- **Statistics Log**: Detailed frame-by-frame analysis (optional)

## 🔧 Troubleshooting

### Common Issues

**1. Low Detection Accuracy**
```python
# Reduce confidence threshold
detections_batch = self.model.predict(frames, conf=0.1)  # Lower from 0.15

# Check input video quality and lighting conditions
```

**2. Team Assignment Errors**
```python
# Increase sample frames for better color detection
sample_frames = min(20, len(video_frames))  # Increase from 10

# Manual color override if needed
team_assigner.team_colors = {1: [R,G,B], 2: [R,G,B]}
```

**3. Memory Issues**
```python
# Reduce batch size
batch_size = 10  # Reduce from 20

# Enable frame skipping for long videos
frame_step = 2  # Process every 2nd frame
```

**4. Slow Performance**
```python
# Enable caching
tracks = tracker.get_object_tracks(frames, read_from_stub=True)

# Use GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Performance Optimization

**GPU Usage**
```python
# Verify CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Memory Management**
```python
# Clear cache periodically
if frame_num % 100 == 0:
    torch.cuda.empty_cache()  # For GPU
    gc.collect()  # For RAM
```

## 🎨 Customization

### Custom Colors
```python
# Define custom team colors
TEAM_COLORS = {
    1: (0, 100, 255),    # Orange
    2: (255, 100, 0),    # Blue
    'referee': (0, 255, 255),  # Yellow
    'ball': (0, 255, 0)  # Green
}
```

### Additional Statistics
```python
# Add custom metrics
def calculate_player_speed(tracks):
    # Implementation for speed calculation
    pass

def detect_ball_possession_changes(tracks):
    # Implementation for possession change detection
    pass
```

### Export Options
```python
# Export tracking data
import json
with open('tracking_data.json', 'w') as f:
    json.dump(tracks, f, indent=2)

# Export statistics
stats_df = pd.DataFrame(statistics)
stats_df.to_csv('match_statistics.csv', index=False)
```

# Jersey Number Detection Using YoloV11
---

This project using YoloV11 for detection jersey player number (football, basketball, etc) that dataset collected 
from public dataset [RoboFLow](https://universe.roboflow.com/flashxyz/jerseydetection). The results from train
the model is 0.97.
---

# Train Results
|mAP50|mAP50-95|Precison|Recall|  Size  |
|-----|--------|--------|------|--------|
| 0.97|  0.81  |  0.93  | 0.93 |40.6 MB |


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **Ultralytics YOLO**: State-of-the-art object detection
- **Supervision**: Advanced computer vision utilities
- **ByteTrack**: High-performance multi-object tracking
- **OpenCV**: Computer vision and image processing
- **scikit-learn**: Machine learning for team classification

## 🚀 Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Advanced player statistics (heat maps, speed analysis)
- [ ] Formation detection and tactical analysis
- [ ] Web dashboard for match analysis
- [ ] Mobile app integration
- [ ] Multi-camera angle support
- [ ] Player identification and jersey number recognition
- [ ] Automated highlight generation

---