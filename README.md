# Neuroimaging Robot Powered by Brain Foundation Models

A B.Tech final-year project developing a brain MRI classification system using a fine-tuned BrainIAC-based 3D Vision Transformer (ViT-B).

The project combines volumetric MRI preprocessing, model fine-tuning, evaluation, ONNX model export, inference, and a graphical interface. The completed application runs on an NVIDIA Jetson AGX Orin connected to a monitor, keyboard, and mouse.

## Team

- Srinjay Sarkar
- Sujatra Biswas
- Mahargha Kundu

**Faculty Guide:** Dr. Shridevi S

School of Computer Science and Engineering  
Vellore Institute of Technology, Chennai, India

## Classification Categories

- Normal
- Mild Cognitive Impairment (MCI)
- Alzheimer's disease
- Stroke

## System Workflow

1. Prepare labelled brain MRI volumes in NIfTI format.
2. Apply the required MRI preprocessing.
3. Fine-tune the BrainIAC-based 3D ViT-B model.
4. Evaluate the classifier and generate performance visualisations.
5. Export the model to ONNX.
6. Run the inference application on the NVIDIA Jetson AGX Orin.
7. Display classification scores and axial, coronal, and sagittal MRI views.

## Repository Structure

- `app/` — Python GUI application
- `assets/` — Result screenshots and visualisations
- `data/` — Dataset archive, distributed separately
- `docs/` — Project report and implementation roadmap
- `models/` — Model artifacts, distributed separately
- `notebooks/` — Model development and evaluation notebook

## Dataset

The prepared dataset contains 60 MRI volumes, with 15 volumes per class across four categories.

## Evaluation

The project includes training and validation curves, a confusion matrix, class-wise precision, recall and F1 scores, ROC curves, and precision-recall curves.

Results should be interpreted in the context of the dataset size and evaluation methodology. They do not establish clinical reliability.

## Deployment

The completed deployment runs on an NVIDIA Jetson AGX Orin. A connected monitor displays the GUI, which is operated using a keyboard and mouse. No touchscreen is used.

Refer to the application source and implementation roadmap for environment requirements and launch instructions.

## Project Artifacts

**Google Drive:** https://drive.google.com/drive/folders/1ISF30wpHTtsws32YH2VQdwa2VKH58RGM?usp=sharing

The full dataset and large model artifacts are hosted separately because of their size.

## Limitations and Responsible Use

This is an academic research prototype, not a clinically validated diagnostic system. Its predictions must not be used as a substitute for assessment by qualified healthcare professionals.

## Acknowledgements

We thank our faculty guide, Dr. Shridevi S, for her guidance and support throughout the project.
