Dataset
This project uses dermoscopic skin images with binary lesion segmentation masks.
The original project files contain image identifiers in the `ISIC\_\*` format, but the exact ISIC release/source was not recorded.
Expected Structure
```text
dataset/
├── images/
│   ├── ISIC\_0000001.jpg
│   └── ...
└── masks/
    ├── ISIC\_0000001\_segmentation.png
    └── ...
```
Each image should have a corresponding segmentation mask using the same image identifier followed by `\_segmentation.png`.
Before redistributing the dataset, verify the original source, license, and attribution requirement
