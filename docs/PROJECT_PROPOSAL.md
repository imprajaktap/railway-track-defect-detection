Project Proposal
Project Title
Automated Railway Track Surface Defect Detection and Localization Using Deep Learning-Based Object Detection
1. Introduction
Railway infrastructure requires regular inspection to identify visible defects and abnormalities that may require maintenance. Visual inspection can be supported by computer-vision systems that analyze images and video, locate relevant regions, and provide consistent screening outputs for human review.

This project proposes an academic prototype for detecting and localizing visible railway-track surface defects using a deep-learning object-detection model. The system will use a publicly available, appropriately licensed annotated dataset and will produce class labels, bounding boxes, and confidence scores for detected objects. A video-processing component will explore sequential frame inference as an inspection simulation.
2. Problem Statement
Manual inspection of railway images can be time-consuming, and visual abnormalities may be difficult to consistently identify across varying conditions. An automated image-based system could assist with screening by highlighting candidate defect regions. However, performance depends on dataset quality, defect visibility, image resolution, camera conditions, and model generalization.
3. Aim
To develop and evaluate a computer-vision prototype that detects and localizes dataset-defined visible railway track defects in images and video frames.
4. Objectives
- Identify a suitable public railway-track defect dataset and document its provenance and annotation format.
- Explore the dataset, class distribution, image quality, and annotation characteristics.
- Prepare the images and labels for object detection.
- Train or fine-tune a YOLO-based detector as a baseline.
- Evaluate the model using precision, recall, F1-score, mAP@50, and mAP@50–95.
- Implement image inference and a video frame-processing simulation.
- Create an interactive interface for visualizing predictions and exporting summaries.
5. Scope
The prototype is limited to visible defect categories represented in the selected dataset. The project will not claim to detect all railway defects, measure track geometry, identify subsurface faults, or certify railway safety. Video inference will be an offline or simulated workflow unless an authorized and safe real-world data source is available.
6. Proposed Tools
- Python
- PyTorch
- YOLO implementation selected for the experiment
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Jupyter Notebook
7. Proposed Workflow
1. Dataset identification and download under applicable license/usage terms.
2. Dataset inspection and annotation verification.
3. Train/validation/test split, preferably grouped to avoid leakage from related frames or images.
4. Preprocessing and training-only augmentation.
5. Baseline model training.
6. Evaluation and error analysis.
7. Image inference.
8. Video frame sampling and sequential inference.
9. Dashboard development.
10. Documentation and reproducibility review.
8. Evaluation
The model will be assessed using precision, recall, F1-score, mAP@50, and mAP@50–95. Where practical, inference speed and model size will also be reported. All reported performance values will be based on experiments actually conducted using a documented test set.
9. Expected Outcome
The expected outcome is an academic prototype that can annotate images with detection boxes and labels, process sampled video frames, and display model outputs and evaluation summaries through a simple interface.
10. Risks and Limitations
Potential challenges include class imbalance, small defects, noisy or inconsistent labels, lighting and viewpoint changes, data leakage, and limited generalization. The prototype will be treated as a research screening aid rather than a safety-critical system.
11. Safety and Data Ethics
Data will be obtained from public datasets, authorized sources, or safely and lawfully collected material. The project will not require entering railway tracks or restricted infrastructure. Detection outputs will not be used independently for operational or safety decisions.
12. Project Status
Under development. Dataset selection, model training, evaluation, and dashboard results will be documented as they are completed.
