<div align="center">

<img src="assets/vsr_banner.png" alt="VSR Project Banner" width="100%"/>

<br/>

# 🎬 VSR — Deep Learning Approaches for<br/>Artistic & Temporal-Consistent Video Super-Resolution

<p><em>MCA Dissertation Project · Amity University, Noida · 2024–2026</em></p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-F7C948?style=for-the-badge)](LICENSE)

<br/>

![Repo Visitors](https://visitor-badge.laobi.icu/badge?page_id=Uvesh-Ahmad.VSR-Dissertation&left_color=1a1a2e&right_color=6c63ff&left_text=Repo%20Visitors&style=flat-square)
![Profile Views](https://komarev.com/ghpvc/?username=Uvesh-Ahmad&label=Profile+Views&color=0e75b6&style=flat-square)
[![GitHub Stars](https://img.shields.io/github/stars/Uvesh-Ahmad/VSR-Dissertation?style=flat-square&color=yellow&label=Stars)](https://github.com/Uvesh-Ahmad/VSR-Dissertation)

<br/>

**Author:** [Uvesh Ahmad](https://github.com/Uvesh-Ahmad) &nbsp;·&nbsp; `AU/2021/CSE/AI-0847`  
**Supervisor:** Prof. Dr. Rajbala Simon &nbsp;·&nbsp; **Amity University, Noida** &nbsp;·&nbsp; Batch 2021–2025

<br/>

[![Google Scholar](https://img.shields.io/badge/Scholar-FHoUy3QAAAAJ-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=FHoUy3QAAAAJ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-uvesh--ahmad000-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/uvesh-ahmad000)
[![GitHub](https://img.shields.io/badge/GitHub-Uvesh--Ahmad-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Uvesh-Ahmad)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--3573--320X-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0001-3573-320X)

</div>

---

## 🖼️ Pipeline Overview

<div align="center">

<img src="assets/vsr_pipeline.png" alt="Hybrid VSR Pipeline Diagram" width="92%"/>

<sub><b>Figure 1.</b> Proposed Hybrid VSR Framework — GAN-based upscaling fused with optical flow–guided temporal warping via learnable alpha blending.</sub>

</div>

---

## 📌 Abstract

**Video Super-Resolution (VSR)** reconstructs high-resolution (HR) video sequences from low-resolution (LR) inputs. Existing approaches face a fundamental trade-off: GAN-based methods produce sharp textures but introduce inter-frame flickering, while RNN-based methods ensure temporal smoothness at the cost of spatial detail.

This work proposes a **Hybrid Alpha-Blend Framework** that combines per-frame ESRGAN upscaling with Farneback optical flow–guided warping to achieve both sharpness and temporal consistency — without additional training overhead.

> 💡 **Core Formula:**
> ```
> Output[t] = α × SR_GAN[t] + (1−α) × Warp(SR[t−1], Flow[t])
> ```

---

## ⚠️ Problem Statement

| Method | Strength | Limitation |
|:---|:---|:---|
| GAN-based (Real-ESRGAN) | Sharp, detailed textures | ❌ Inter-frame flickering |
| RNN-based (BasicVSR) | Temporally smooth output | ❌ Over-smoothed / blurry |
| **Proposed Hybrid** ✅ | Sharp **+** Temporally stable | ✅ Best of both worlds |

---

## 🧠 Methodology — Hybrid VSR Framework

**Four-Stage Pipeline:**

| Stage | Module | Description |
|:---:|:---|:---|
| 1️⃣ | **ESRGAN Upscaling** | Per-frame 4× super-resolution via Real-ESRGAN |
| 2️⃣ | **Optical Flow Estimation** | Farneback dense flow between consecutive HR frames |
| 3️⃣ | **Frame Warping** | Warp previous HR frame using estimated flow field |
| 4️⃣ | **Alpha Blending** | Weighted fusion: `α·GAN + (1−α)·Warped` |

```
┌─────────────┐     Real-ESRGAN ×4     ┌──────────────────┐
│  LR Frame t │ ─────────────────────► │  SR_GAN Frame[t] │──────┐
└─────────────┘                        └──────────────────┘      │
                                                                  ▼
┌──────────────┐   Optical Flow (Farneback)  ┌──────────────┐  Alpha
│  HR Frame    │ ──────────────────────────► │ Warped Frame │  Blend ──► HR Output[t]
│  [t−1]       │                             └──────────────┘    ▲
└──────────────┘                                                  │
                                              α = 0.70 (default) ┘
```

---

## 📊 Experimental Results

<div align="center">

| Method | PSNR ↑ | SSIM ↑ | User Preference ↑ |
|:---|:---:|:---:|:---:|
| BasicVSR | 28.45 dB | 0.903 | 68% |
| Real-ESRGAN | 28.90 dB | 0.911 | 72% |
| **Proposed Hybrid** 🏆 | **29.12 dB** | **0.925** | **85%** |

</div>

> 📈 **+0.22 dB PSNR** · **+0.014 SSIM** over best baseline · **85% user preference** in perceptual evaluation

---

## 🚀 Quick Start

### Step 1 — Clone & Install

```bash
git clone https://github.com/Uvesh-Ahmad/VSR-Dissertation.git
cd VSR-Dissertation

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Step 2 — Download Pre-trained Weights (~50 MB, one-time)

```bash
python -c "
import torch
torch.hub.download_url_to_file(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'weights/RealESRGAN_x4plus.pth'
)"
```

### Step 3 — Run Inference

```python
from hybrid_vsr import hybrid_vsr

hybrid_vsr("input.mp4", "output.mp4", alpha=0.70)
```

---

## ⚡ Content-Adaptive Alpha Values

```python
hybrid_vsr("sports.mp4",   "sports_hr.mp4",   alpha=0.60)  # Fast motion → temporal stability
hybrid_vsr("cartoon.mp4",  "cartoon_hr.mp4",  alpha=0.80)  # Edges → spatial sharpness
hybrid_vsr("nature.mp4",   "nature_hr.mp4",   alpha=0.70)  # General → balanced
```

| Content Type | Recommended α | Rationale |
|:---|:---:|:---|
| Sports / Action | 0.55 – 0.60 | Prioritize temporal stability over sharpness |
| Cartoon / Animation | 0.78 – 0.82 | Prioritize edge sharpness |
| Nature / Documentary | 0.68 – 0.72 | Balanced blend |
| Archival / Old Film | 0.65 – 0.70 | Grain-aware blending |

---

## 📁 Project Structure

```
VSR-Dissertation/                    (~300 MB total)
│
├── README.md
├── hybrid_vsr.py                    # Core inference pipeline
├── evaluate.py                      # PSNR / SSIM evaluation script
├── requirements.txt
│
├── assets/                          # README images
│   ├── vsr_banner.png               # ← Upload your title banner
│   └── vsr_pipeline.png             # ← Upload your pipeline diagram
│
├── weights/
│   └── RealESRGAN_x4plus.pth        # Pre-trained model (~50 MB)
│
├── datasets/
│   ├── sample_nature.mp4            # Test clip (~20 MB)
│   ├── sample_sports.mp4            # Test clip (~20 MB)
│   └── sample_cartoon.mp4           # Test clip (~15 MB)
│
├── outputs/
│   ├── logs/
│   └── results/
│
└── demo/
    └── index.html
```

---

## 🔧 Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
scikit-image>=0.20.0
scipy>=1.10.0
```

```bash
pip install -r requirements.txt    # ~1–2 GB total (PyTorch included)
```

---

## 💻 System Requirements

| Component | Requirement | Notes |
|:---|:---|:---:|
| RAM | 8 GB minimum | 16 GB recommended |
| Storage | 2 GB free | For model + outputs |
| CPU | Any modern (2015+) | No GPU required |
| GPU | Optional (CUDA) | ⚡ ~10× faster |

> **CPU-only mode fully supported.** Estimated time: ~45–60 min per 30-sec video (900 frames).

---

## 📝 Core Implementation

<details>
<summary><b>▶ Click to expand — <code>hybrid_vsr.py</code></b></summary>

```python
#!/usr/bin/env python3
"""
Hybrid VSR Inference — Lightweight Version
Author  : Uvesh Ahmad | AU/2021/CSE/AI-0847 | Amity University
Advisor : Prof. Dr. Rajbala Simon
"""

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def hybrid_vsr(input_path: str, output_path: str, alpha: float = 0.70, scale: int = 4):
    """
    Hybrid VSR with alpha-blend temporal refinement.

    Args:
        input_path  : Path to input low-resolution video.
        output_path : Path to save output high-resolution video.
        alpha       : Blend factor. Higher = sharper; lower = smoother.
                      Recommended: 0.60 (sports) · 0.70 (general) · 0.80 (cartoon)
        scale       : Upscale factor (default: 4×).
    """
    # Load Real-ESRGAN (pre-trained, no fine-tuning needed)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
    upsampler = RealESRGANer(
        scale=scale,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        half=False  # FP32 for CPU stability
    )

    cap    = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w * scale, h * scale))

    prev_hr, prev_gray, frame_count = None, None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Per-frame ESRGAN upscaling
        sr_gan, _ = upsampler.enhance(frame, outscale=scale)
        sr_gray   = cv2.cvtColor(sr_gan, cv2.COLOR_BGR2GRAY)

        if prev_hr is not None:
            # Stage 2: Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, sr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Stage 3: Warp previous HR frame
            h_hr, w_hr = sr_gan.shape[:2]
            map_x = (flow[:, :, 0] + np.arange(w_hr)).astype(np.float32)
            map_y = (flow[:, :, 1] + np.arange(h_hr).reshape(-1, 1)).astype(np.float32)
            warped = cv2.remap(prev_hr, map_x, map_y, cv2.INTER_LINEAR)

            # Stage 4: Alpha blending
            result = alpha * sr_gan.astype(np.float32) + (1 - alpha) * warped.astype(np.float32)
            output = np.clip(result, 0, 255).astype(np.uint8)
        else:
            output = sr_gan

        out.write(output)
        prev_hr, prev_gray = output.copy(), sr_gray.copy()
        frame_count += 1
        print(f"  [Frame {frame_count:04d}] processed.")

    cap.release()
    out.release()
    print(f"\n✅ Output saved → {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hybrid VSR Inference")
    p.add_argument("--input",  required=True,          help="Input video path")
    p.add_argument("--output", required=True,          help="Output video path")
    p.add_argument("--alpha",  type=float, default=0.70, help="Blend factor (default: 0.70)")
    args = p.parse_args()
    hybrid_vsr(args.input, args.output, args.alpha)
```

</details>

---

## 📚 Citation

If this work is useful for your research, please cite:

```bibtex
@mastersthesis{ahmad2025vsr,
  author     = {Uvesh Ahmad},
  title      = {Deep Learning Approaches for Artistic and Temporal-Consistent
                Video Super-Resolution},
  school     = {Amity University, Noida},
  year       = {2025},
  supervisor = {Prof. Dr. Rajbala Simon},
  note       = {MCA Dissertation, Enrollment: AU/2021/CSE/AI-0847}
}
```

---

## 👤 About the Author

<table>
<tr>
<td width="62%">

### Uvesh Ahmad
**Teaching Assistant & Researcher**  
Amity University, Noida &nbsp;·&nbsp; ⭐ QS Rank 951–1000  
Supervisor: **Prof. Dr. Rajbala Simon**  
`AU/2021/CSE/AI-0847` &nbsp;·&nbsp; Batch 2021–2025

**🎓 MCA (AI/ML)** · Amity University, Noida · 2024–2026  
**🏫 BCA** · Lovely Professional University, Punjab · 2021–2024

</td>
<td width="38%" align="center">

![Repo Visitors](https://visitor-badge.laobi.icu/badge?page_id=Uvesh-Ahmad.VSR-Dissertation&left_color=1a1a2e&right_color=6c63ff&left_text=Repo%20Visitors&style=flat-square)

![Profile Views](https://komarev.com/ghpvc/?username=Uvesh-Ahmad&label=Profile+Views&color=0e75b6&style=flat-square)

</td>
</tr>
</table>

### 🔬 Research at Amity University

Working as **Teaching Assistant & Researcher** (May 2025 – May 2026) under **Prof. Dr. Rajbala Simon**:

- 📄 **IEEE Xplore (2025)** — *"Optimized Tour Planning Advisor Using Regression Models"*
- 🎥 MCA Dissertation — **Video Super-Resolution** *(this project)*
- 🧠 Ongoing research in **Neuro-Symbolic AI**, **Real-Time Object Detection**, **Computer Vision**
- 👨‍🏫 Mentored **7+ students** in Dissertations, Research Papers & Academic Projects

### 🛠️ Technical Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

**Research Areas:** Computer Vision · Deep Learning · Video Super-Resolution · GAN-based Methods · Optical Flow · Temporal Consistency

---

### 📈 GitHub Activity Graph

[![Uvesh's GitHub Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=Uvesh-Ahmad&theme=react-dark&bg_color=0d1117&color=6c63ff&line=6c63ff&point=ffffff&area=true&area_color=6c63ff&hide_border=true)](https://github.com/Uvesh-Ahmad)

---

### 📊 GitHub Stats

<div align="center">

![Uvesh's GitHub Stats](https://github-readme-stats.vercel.app/api?username=Uvesh-Ahmad&show_icons=true&theme=react&bg_color=0d1117&title_color=6c63ff&icon_color=6c63ff&text_color=ffffff&border_color=6c63ff&hide_border=false&count_private=true)&nbsp;&nbsp;
![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Uvesh-Ahmad&layout=compact&theme=react&bg_color=0d1117&title_color=6c63ff&text_color=ffffff&border_color=6c63ff&hide_border=false)

</div>

---

### 🔗 Academic Profiles

| Platform | Link |
|:---|:---|
| 📚 Google Scholar | [FHoUy3QAAAAJ](https://scholar.google.com/citations?user=FHoUy3QAAAAJ) |
| ⚡ IEEE Xplore | [Published Paper](https://ieeexplore.ieee.org) |
| 💼 LinkedIn | [uvesh-ahmad000](https://linkedin.com/in/uvesh-ahmad000) |
| 🐙 GitHub | [Uvesh-Ahmad](https://github.com/Uvesh-Ahmad) |
| 🆔 ORCID | [0009-0001-3573-320X](https://orcid.org/0009-0001-3573-320X) |
| 📧 Email | uveshah20@gmail.com |

---

<div align="center">

⭐ **If this project helped you, please consider starring the repository** ⭐

<br/>

*© 2025 Uvesh Ahmad · Amity University, Noida · MIT License*  
*Supervised by Prof. Dr. Rajbala Simon*

</div>
