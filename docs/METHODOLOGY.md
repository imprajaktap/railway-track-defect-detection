Project Methodology
1. Overview
This document describes the proposed experimental workflow for the railway track defect detection project. The workflow is designed to be reproducible and to avoid reporting results before they are measured.
2. Dataset Selection
Select a publicly available railway-track defect dataset that:
- Provides image files and usable annotations.
- Defines its defect classes clearly.
- Permits the intended academic use.
- Includes sufficient information to document provenance and limitations.
Record the dataset source, version or access date, license/usage terms, classes, annotation format, and any exclusions. Do not assume defect classes before inspecting the dataset.
3. Dataset Exploration
Perform an initial analysis:
- Count images and annotated instances.
- Inspect class distribution.
- Review image dimensions and image quality.
- Examine bounding-box sizes and annotation consistency.
- Inspect representative images and labels.
- Identify duplicates or near-duplicates where feasible.
Suggested output: dataset summary table, class-count graph, and annotated sample images.
4. Dataset Partitioning
Create training, validation, and test partitions. Use a documented split ratio and random seed. If images come from videos, recording sessions, or related image sequences, group related samples when possible to reduce leakage between partitions.

The validation set is used for tuning and checkpoint selection. The test set is reserved for final evaluation.
5. Preprocessing
Document and implement:
- Image loading and color-channel handling.
- Annotation conversion to the detector's required format.
- Image resizing and padding.
- Any normalization required by the selected model.
Ensure labels and image transformations remain aligned. Keep evaluation and inference preprocessing consistent with the model's requirements.
6. Data Augmentation
Use augmentation only on training data. Potential augmentations include moderate brightness and contrast changes, scaling, cropping, translation, and other transformations that preserve label meaning.

Review augmented samples visually. Avoid transformations that create unrealistic images or invalid labels. Do not augment validation or test data as a means of increasing their sample count.
7. Baseline Model
Train or fine-tune a YOLO-based detector. Document:
- Model variant and initialization.
- Input resolution.
- Batch size.
- Epochs.
- Optimizer and learning-rate settings.
- Augmentation configuration.
- Hardware and software versions.
- Random seed, where applicable.
Start with a baseline before exploring more complex model variants.
8. Training and Validation
Monitor training and validation metrics. Use validation data to identify overfitting and select a checkpoint. Save training logs, configuration, and model weights where repository size and licensing permit; large weights may be stored separately.
9. Evaluation
Evaluate the selected model on the held-out test set using:
- Precision.
- Recall.
- F1-score.
- mAP@50.
- mAP@50–95.
Where practical, additionally report class-wise metrics, inference latency, throughput, and model size. Include the evaluation configuration and the test-set definition. Do not invent or estimate metrics.
10. Error Analysis
Review examples of:
- False positives.
- False negatives.
- Localization errors.
- Class confusion.
- Small or visually ambiguous defects.
- Difficult lighting, background, or viewpoint conditions.
Use error analysis to identify limitations and potential future experiments. Avoid broad generalization from a small number of examples.
11. Image Inference
Implement inference for a single uploaded image. The output should include:
- Annotated image.
- Detection class.
- Confidence score.
- Bounding-box coordinates, if available.
Confidence values are model scores and are not calibrated safety probabilities.
12. Video Inspection Simulation
For a video:
1. Read the video and its metadata.
2. Sample frames at a configurable interval.
3. Run detector inference on sampled frames.
4. Save timestamps and detections.
5. Render annotated frames or a preview.
6. Summarize detections and export a log.
If speed is assumed to estimate distance covered, label that estimate as approximate and document the assumption. Timestamp alone is not geographic localization. Do not count every repeated frame detection as a unique defect without a validated grouping method.
13. Dashboard
The Streamlit interface may include:
- Image upload and detection.
- Video upload and frame sampling.
- Confidence threshold control for display.
- Annotated output preview.
- Detection summary table.
- Downloadable results.
Include a disclaimer that this is an academic visual-screening prototype, not a certified railway inspection system.
14. Reproducibility
Maintain:
- requirements file.
- dataset documentation.
- model and experiment configuration.
- train/validation/test split details.
- logs and metric results.
- code version and commit history.
15. Limitations
The system is limited by its dataset, annotations, camera conditions, visibility, model capacity, and evaluation design. It may miss defects or produce false detections. It cannot inspect subsurface faults or certify railway safety.
16. Proposed Directory Mapping
- notebooks/: exploration and experiments.
- src/preprocessing.py: preprocessing and annotation utilities.
- src/train.py: training entry point.
- src/predict.py: image prediction.
- src/video_pipeline.py: frame extraction and video inference.
- src/evaluation.py: evaluation utilities.
- app/app.py: dashboard.
- docs/: proposal, methodology, dataset documentation, and project notes.