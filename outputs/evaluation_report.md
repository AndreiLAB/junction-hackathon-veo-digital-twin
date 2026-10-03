# Visual Localization Evaluation Report

## Methodology
- **Reference Database**: 108 original E57 extracted images.
- **Query Generation**: 3 augmented queries generated dynamically per reference image.
- **Augmentations**: ColorJitter, RandomPerspective, RandomResizedCrop, GaussianBlur.
- **Embedding Model**: dinov2_vits14 (Cosine Similarity Retrieval).
- **Note on Architecture**: The primary localization system is image retrieval, not scan-ID classification. While `train_model.py` explores a lightweight scan classifier, classification cannot generalize to unseen scans not present in the training set. Retrieval naturally handles this by simply adding new reference views.

## Results
- **Total Queries**: 324
- **Top-1 Accuracy**: 93.83% (Query matched its exact un-augmented source image)
- **Top-5 Accuracy**: 100.00%
- **Median Spatial Error**: 0.000 meters
- **Mean Spatial Error**: 0.093 meters
- **Median Angular Error**: 0.00 degrees
- **Avg Inference Time**: 58.5 ms per image
