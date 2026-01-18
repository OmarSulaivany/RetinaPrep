# RetinaPrep: Human-in-the-Loop Fundus Pre-processing

**RetinaPrep** is a web-based, "Human-in-the-Loop" tool designed to standardize the cropping and quality assurance of retinal fundus images for deep learning applications. It bridges the gap between fully automated scripts and manual curation by providing an interactive interface to verify, correct, and export high-quality datasets.

![RetinaPrep Interface](figures/fig6.png)
*Figure 1: The Main Interface showing the Tri-Pane view (Original, AI Mask, Interactive Crop).*

## 🚀 Key Features

* **Intelligent Discovery:** "Gallery Grid" view with color-coded status badges (Approved/Rejected) based on AI confidence scores.
* **Tri-Pane Editing:** Synchronized view of the **Original Input**, **AI Segmentation Mask** (SegFormer), and **Interactive Crop**.
* **Precision Tools:**
    * **Click-to-Center:** Manually correct optic disc alignment with a single click.
    * **Dynamic Resizing:** Adjust crop radius pixel-perfectly using drag handles.
    * **Visual Enhancement:** Hardware-accelerated brightness slider for inspecting dark fundus images without altering the export.
* **Quality Assurance:** Real-time metrics for Quality Score, Circularity, and Geometric Area.
* **Structured Output:**
    * Auto-generated CSV reports (containing filenames, geometric metrics, and validation status).
    * Standardized ZIP archives containing separated `Crops/` (for training) and `Visualizations/` (for debugging) folders.

## 🛠️ Tech Stack

* **Backend:** Python 3.x, Flask
* **Frontend:** HTML5, JavaScript, Bootstrap
* **AI Model:** SegFormer (HuggingFace Transformers)
* **Image Processing:** OpenCV, Pillow (PIL)

---

## 💻 Installation & Setup

Follow these steps to run RetinaPrep locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/OmarSulaivany/RetinaPrep.git](https://github.com/OmarSulaivany/RetinaPrep.git)
cd RetinaPrep

