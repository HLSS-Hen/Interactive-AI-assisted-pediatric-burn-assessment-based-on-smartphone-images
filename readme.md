# Interactive AI assisted pediatric burn assessment based on smartphone images

This repository contains partial code accompanying the same name paper.

## Getting Started

### Prerequisites
- Python 3.12
- Required Python packages (listed in requirements.txt)

### Model Weights
Download the pre-trained model weights from Hugging Face: [pytorch_model.bin](https://huggingface.co/HLSS/SAM-DR/resolve/main/pytorch_model.bin?download=true)

### Using the Diagnosis-Annotation Tool

#### Step 1: Upload Images
Place your smartphone images into the `demo/data/` folder.  
> 💡 You can skip this step and use the provided example images instead.

#### Step 2: Load Image in Gradio App
- Click **Refresh Image List** to refresh image list in gradio app
- Click to select the image you want to annotate
- The app will display the **full-image heatmap** and **histogram**

#### Step 3: Select Region of Interest (ROI)
- Click **two points** on the image to draw a bounding box
- The tool will display the **heatmap** and **histogram** computed within the selected ROI (the rest of the image is masked out)

#### Step 4: Apply Threshold Segmentation
- Click on a position within the histogram to select a threshold
- The segmentation **mask** will be generated and displayed

#### Step 5: Classify and Annotate
- Select the **burn depth classification** (2:superficial / 3:partial / 4:full thickness)
- Click **"Add Annotation"** to save the current region

#### Step 6: Complete Multi-Region Annotation
- Repeat **Steps 3–5** until all burn regions in the image are annotated

#### Step 7: Save Annotations
- Click **"Save Annotations"**
- A JSON file with the **same name as the image** will be created in `demo/data/`, containing all annotation data for that image

## License
This project is licensed under the Apache 2.0 License.

## Citation
```bibtex
@article{wang2024interactive,
title={Interactive AI assisted pediatric burn assessment based on smartphone images},
author={Wang, Hao and Zeng, Shuaidan and Li, Weiqing and Chen, Xiaodi and Mei, Qianqian and Chen, Huating and Fu, Lina and Zhao, Zhenhui and Tang, Shengping and Zheng, Kaize and Liang, Yanyan and Xiong, Zhu},
journal={Scientific Reports},
year={2026}
}
```