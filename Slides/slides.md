---
marp: true
---

# traffic Light Detection
<style>
    img[alt~="center"]{
        display: block;
        margin: 0 auto;
    }
</style>
---

Model
===
We use Mobilenet ssd v3

---

# Mobilenet ssd
MobileNet SSD v3 is primarily used for the task of object detection, which involves identifying and locating objects of interest within an image or video frame. It can detect multiple object classes simultaneously. In the COCO dataset, the 10th class is traffic lights

---
# Why Mobilenet
1. **Real-time Applications:** MobileNet SSD v1 is designed to be lightweight and efficient, making it suitable for real-time or embedded applications where computational resources are limited.
2. **Pre-Trained:** The model is pretrained to show good performance on object detection task, it has already been trained to detect traffic lights.


---

# Example
**Test Image**
![center width:700](lajpat.png)
<p align='center'>courtesy: Google Maps</p>

---
# Bounding Boxes
![center width:700](sample_result.png)

---

# Color Detection
We use a simple algorithm to find the color of the traffic light

1. Convert image to HSV.
2. Use only the pixels which have high saturation.
3. Use the hue values to map to a color.


---

# References
1. [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
