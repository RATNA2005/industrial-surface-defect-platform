# Data Directory: NEU Surface Defect Database

This directory stores the dataset used for the Industrial Surface Defect Inspection Platform.

## Directory Structure
```
data/
└── raw/
    ├── images/       # 1,800 grayscale JPEG images (200x200) across 6 defect classes
    └── annotations/  # 1,800 Pascal VOC XML annotation files with bounding box coordinates
```

## Classes
1. Crazing (`crazing`) - 300 samples
2. Inclusion (`inclusion`) - 300 samples
3. Patches (`patches`) - 300 samples
4. Pitted Surface (`pitted_surface`) - 300 samples
5. Rolled-in Scale (`rolled-in_scale`) - 300 samples
6. Scratches (`scratches`) - 300 samples

Total: 1,800 images (perfectly balanced 16.67% per class).

Detailed audit results and file checks are available in `docs/dataset_audit.md`.
