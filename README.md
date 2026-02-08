# Understanding Convolutional Layers in PyTorch

This project documents my hands-on exploration of **convolutional layers and pooling operations** in PyTorch. The focus of this work is to understand how convolution operates on real image data, how feature maps are generated, and how spatial dimensions change through convolution and pooling.

---

## Objective

The objective of this work is to:
- Understand how 2D convolution works on image data
- Explore kernel size, padding, and channel expansion
- Study how feature maps are generated
- Understand pooling operations and spatial downsampling
- Learn how global pooling layers reduce spatial dimensions

---

## Input Data

- Sample RGB images loaded using `sklearn.datasets.load_sample_images`
- Images are converted to floating-point tensors
- Pixel values are normalized to the range `[0, 1]`
- Tensor dimensions are rearranged to match PyTorch format:
  - From `(N, H, W, C)` → `(N, C, H, W)`

This ensures compatibility with PyTorch convolution layers.

---

## Image Preprocessing

- `CenterCrop((70, 120))` is applied to standardize spatial dimensions
- Ensures consistent input size before convolution
- Mimics real-world preprocessing pipelines used in CNNs

---

## Convolution Layer Exploration

### Basic Conv2D Layer
- A `Conv2d` layer is applied with:
  - Input channels: 3 (RGB)
  - Output channels: 32
  - Kernel size: 7 × 7
  - No padding (initially)

This demonstrates how spatial dimensions shrink when padding is not used.

---

### Convolution with Padding
- Padding is set to `"same"`
- Output feature maps retain original spatial dimensions
- Demonstrates how padding preserves border information

---

### Parameter Inspection

- Convolutional weights and biases are explicitly inspected
- Observations:
  - Each output channel has its own kernel
  - Weights are shared across spatial locations
  - Bias is applied per output channel

This reinforces the concept of parameter sharing in CNNs.

---

## Pooling Operations

### Max Pooling
- `MaxPool2d` is applied after convolution
- Spatial dimensions are reduced
- Channel depth remains unchanged
- Demonstrates how pooling summarizes local features

---

### Custom Depth Pooling

- A custom `DepthPool` module is implemented
- Pools across the **channel dimension** instead of spatial dimensions
- Demonstrates flexible tensor manipulation and non-standard pooling strategies

---

## Global Average Pooling

- `AdaptiveAvgPool2d(output_size=1)` is applied
- Collapses spatial dimensions to a single value per channel
- Converts feature maps into global feature representations
- Commonly used before classification layers in CNNs

---

## Key Observations

- Convolution extracts local spatial features using shared kernels
- Kernel size controls the receptive field
- Padding affects spatial resolution of feature maps
- Pooling reduces spatial dimensions while retaining important information
- Global average pooling removes spatial structure and keeps semantic information

---

## Conclusion

This project provides a foundational understanding of convolutional layers and pooling mechanisms in PyTorch. By working directly with image tensors and inspecting intermediate outputs, this work builds intuition for how CNNs process visual data before moving on to full convolutional neural network architectures.

---
