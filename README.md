# Grammar_Scoring
Develop a machine learning model that listens to spoken audio samples, extracts meaningful audio features, and predicts a grammar proficiency score for each sample.
🎓 Project Title
Build a Grammar Scoring Engine for Voice Samples
📌 Objective
To design and implement a machine learning pipeline that evaluates the grammar proficiency of individuals based on spoken audio samples, assigning them a grammar score or grade that reflects their language quality.

🧠 Problem Statement
Given a dataset of voice samples (spoken English), predict a grammar proficiency score that correlates with how well the speaker uses grammar. The challenge involves analyzing audio features, not textual transcription, making it a speech-based grammar scoring task.

🔄 Pipeline Overview
1. Data Loading
Input Files:

train.csv (filenames + grammar scores)

audios_train/ (voice samples for training)

test.csv (filenames to predict)

audios_test/ (unlabeled voice samples)

2. Preprocessing & Feature Extraction
Using Librosa, extract meaningful features from each audio file:

MFCCs (Mel-Frequency Cepstral Coefficients) – captures vocal characteristics

Zero Crossing Rate (ZCR) – measures noisiness

Root Mean Square Energy (RMS) – indicates loudness

Spectral Centroid – reflects brightness of sound

Each audio file is converted into a feature vector combining these stats.

3. Label Transformation
The grammar score (ranging from 0 to 5) is converted into grades (A–D) to reduce noise and imbalance.

Rare classes (like E) are merged with similar ones (e.g. D) for better training.

4. Modeling
Used XGBoost Classifier to learn from extracted features.

Labels are encoded using LabelEncoder.

Hyperparameters like n_estimators, max_depth, learning_rate, etc., are tuned for better performance.

5. Evaluation
Validation Accuracy is calculated on a holdout validation set.

Displayed Classification Report (Precision, Recall, F1-score) per class.

Confusion Matrix shows class-wise prediction performance.

6. Test Prediction & Submission
Features are extracted from test audio.

Predicted grades are converted back into grammar scores (e.g., A → 4.5, B → 3.8, etc.)

Results saved in submission.csv for Kaggle submission.

✅ Key Metrics
Validation Accuracy: How accurately the model classifies grammar levels

Grammar Score (proxy): Mapped from classification to expected numeric scores

💡 Impact
This engine enables automated spoken grammar assessment, potentially useful for:

English learning platforms

Language proficiency tests

Voice-based educational tools
