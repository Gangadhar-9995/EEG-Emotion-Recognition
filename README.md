# EEG-Based Emotion Recognition Using the DREAMER Dataset

## 📌 Project Overview

This project focuses on recognizing human emotional states from Electroencephalogram (EEG) signals using machine learning.

EEG signals record electrical activity in the brain and can provide useful information about a person's emotional state. The goal of this project is to process EEG signals, extract meaningful features, and build machine-learning models that predict three emotional dimensions:

- **Valence:** How positive or negative an emotion is.
- **Arousal:** The level of emotional activation, from calm to excited.
- **Dominance:** The degree of control or influence a person feels.

The project uses the **DREAMER dataset**, which contains EEG recordings collected while participants watched emotional video stimuli.

The current implementation includes EEG preprocessing, segmentation, feature extraction, subject-independent data splitting, and classical machine-learning experiments.

## 🎯 Objectives

- Understand and analyze the DREAMER EEG dataset.
- Extract EEG recordings and emotion ratings.
- Reduce noise using digital filtering techniques.
- Divide continuous EEG recordings into fixed-length segments.
- Extract time-domain and frequency-domain features.
- Train machine-learning models to predict Valence, Arousal, and Dominance.
- Evaluate model performance on previously unseen subjects.
- Identify limitations and explore improvements for EEG-based emotion recognition.

## 📊 Dataset

### DREAMER Dataset

DREAMER is an EEG and ECG dataset developed for research on emotion recognition.

The dataset used in this project contains:

| Property | Description |
|---|---|
| Subjects | 23 participants |
| Video sequences per subject | 18 |
| Total trials | 414 |
| EEG channels | 14 |
| Sampling frequency | 128 Hz |
| Emotion dimensions | Valence, Arousal, Dominance |
| Emotion score range | 1–5 |

Each participant watched emotional video sequences while EEG signals were recorded. The dataset provides baseline EEG recordings, stimulus-related EEG recordings, and corresponding self-reported emotion scores.

### Emotion Labels

Each trial has three emotion ratings:

| Score | Interpretation |
|---|---|
| 1 | Lowest rating |
| 2 | Low rating |
| 3 | Middle rating |
| 4 | High rating |
| 5 | Highest rating |

The scores are treated as ordinal ratings, not as universal emotional categories. A higher Valence score indicates a more positive rating, while higher Arousal and Dominance scores indicate higher ratings on those respective dimensions.

**Dataset access:** Obtain the DREAMER dataset from its official publication or an authorized research source. The dataset file is not included in this repository because it is a large research dataset and may be subject to access and usage conditions.

## ⚙️ Project Workflow

The implementation follows this pipeline:

```text
DREAMER Dataset
       |
       v
Dataset Analysis
       |
       v
EEG Signal Extraction
       |
       v
Baseline Correction
       |
       v
50 Hz Notch Filtering
       |
       v
1–45 Hz Band-Pass Filtering
       |
       v
5-Second EEG Segmentation
       |
       v
Emotion Label Preparation
       |
       v
Subject-Independent Data Splitting
       |
       v
EEG Normalization
       |
       v
Time-Domain and Frequency-Domain
Feature Extraction
       |
       v
Feature Scaling
       |
       v
Classical Machine Learning
       |
       v
Model Evaluation
```

## 🧠 Methodology

### 1. EEG Data Extraction

The EEG signals and associated emotion scores are extracted from the MATLAB dataset file, `DREAMER.mat`.

The EEG recordings contain 14 channels sampled at 128 Hz.

The channels are:

`AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`

### 2. Baseline Correction

Baseline EEG is used to estimate the average signal level for each channel.

The baseline mean is subtracted from the corresponding stimulus recording:

```python
baseline_mean = np.mean(baseline_trial, axis=0)
baseline_corrected = stimulus_trial - baseline_mean
```

This helps account for channel-specific baseline offsets.

### 3. EEG Filtering

Two filtering operations are applied:

- **50 Hz notch filter:** Attenuates potential power-line interference.
- **1–45 Hz band-pass filter:** Retains the selected EEG frequency range while attenuating lower-frequency drift and higher-frequency components.

The filters are applied along the time axis of each EEG channel.

Filtering does not guarantee the removal of all artifacts. Artifact detection and more advanced artifact handling remain potential improvements.

### 4. EEG Segmentation

The preprocessed EEG signals are divided into fixed-length windows.

| Parameter | Value |
|---|---|
| Window duration | 5 seconds |
| Sampling rate | 128 Hz |
| Samples per segment | 640 |
| Channels per segment | 14 |
| Overlap | 0% |

Each segment has the shape:

```text
(640, 14)
```

The complete segmented dataset contains **17,020 EEG segments**.

Every segment inherits the Valence, Arousal, and Dominance scores assigned to its original trial.

### 5. Emotion Label Preparation

The original emotion ratings range from 1 to 5.

For five-class classification, the ratings are encoded as:

```text
Original score -> Model class

1 -> 0
2 -> 1
3 -> 2
4 -> 3
5 -> 4
```

Separate prediction tasks are created for Valence, Arousal, and Dominance.

An additional exploratory binary Valence experiment maps scores 1–2 to Low and scores 4–5 to High, excluding score 3. This is a separate experiment and does not replace the five-class task.

### 6. Subject-Independent Data Splitting

To evaluate generalization to unseen participants, subjects are separated into training, validation, and test sets before model fitting.

| Split | Subjects | EEG segments |
|---|---:|---:|
| Training | 15 | 11,100 |
| Validation | 4 | 2,960 |
| Test | 4 | 2,960 |
| **Total** | **23** | **17,020** |

The split uses a fixed random seed of `42`.

- **Training set:** Used to fit models and preprocessing parameters.
- **Validation set:** Used to compare models and select configurations.
- **Test set:** Used for the final evaluation of selected models.

No subject appears in more than one split. This helps prevent information leakage between training and testing.

### 7. EEG Normalization

Each EEG channel is standardized using its mean and standard deviation calculated from the training set.

The same parameters are applied to the training, validation, and test sets.

### 8. Feature Extraction

The current classical machine-learning pipeline extracts 196 features from each EEG segment.

#### Time-Domain Features

Four features are calculated for each channel:

- Mean
- Standard deviation
- Root Mean Square (RMS)
- Line length

These describe signal amplitude and temporal variation.

#### Frequency-Domain Features

Welch's method is used to estimate power spectral density. Power features are extracted from five frequency bands:

| Band | Frequency range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

For each band, the pipeline calculates log absolute power and relative power.

The final feature count is:

```text
4 time-domain features
+ 5 log band-power features
+ 5 relative band-power features
= 14 features per channel

14 channels × 14 features = 196 features
```

The feature matrices have the following shapes:

```text
Training:   (11100, 196)
Validation: (2960, 196)
Test:       (2960, 196)
```

### 9. Feature Scaling

A `StandardScaler` is fitted on the training feature matrix only. The fitted scaler is then applied to validation and test features.

This avoids using information from held-out data to estimate scaling parameters.

### 10. Machine-Learning Models

Three classical machine-learning algorithms are evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Tree-based classification |
| Support Vector Machine (RBF) | Nonlinear classification |

The models are trained separately for Valence, Arousal, and Dominance.

Class weighting is used to help address class imbalance. Models are selected using validation macro-F1 rather than test performance.

## 📈 Experimental Results

### Five-Class Classification

The following results were obtained from the initial subject-independent experiment.

| Emotion dimension | Selected model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|---:|
| Valence | Logistic Regression | 29.32% | 0.2840 | 19.80% | 0.1874 |
| Arousal | SVM (RBF) | 22.50% | 0.2078 | 23.34% | 0.2041 |
| Dominance | SVM (RBF) | 27.67% | 0.2006 | 19.36% | 0.1282 |

These results represent an initial baseline, not a final optimized system.

The models perform better on some validation metrics than on the held-out test set, indicating that generalization across subjects remains challenging.

### Exploratory Binary Valence Classification

An additional Logistic Regression experiment uses two classes:

- Low: Original scores 1–2
- High: Original scores 4–5
- Score 3: Excluded

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 55.46% | 51.24% |
| Macro-F1 | 0.5472 | 0.5118 |

The binary experiment changes the prediction task and uses fewer samples. Its results should therefore be reported separately from the five-class experiment.

### Evaluation Metrics

The project uses:

- **Accuracy:** Proportion of correctly classified samples.
- **Macro-F1:** Average F1-score across classes, giving each class equal weight.
- **Weighted-F1:** F1-score averaged according to class support.
- **Balanced accuracy:** Average recall across classes.
- **Confusion matrix:** Displays correct predictions and class-level errors.

Because emotion ratings are imbalanced, accuracy is considered alongside macro-F1 and balanced accuracy.

## 🛠️ Technologies Used

- **Python** — Main programming language
- **NumPy** — Numerical computations
- **Pandas** — Dataset and label management
- **SciPy** — EEG signal processing and power spectral density estimation
- **Matplotlib** — Signal visualization and plots
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Preprocessing, machine-learning models, and evaluation
- **Jupyter Notebook** — Interactive development and experimentation

## 🚀 Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python version
- Jupyter Notebook
- Access to the DREAMER dataset file

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the URL with your actual GitHub repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Configure the dataset path

Download or obtain authorized access to `DREAMER.mat` and update the file path in the notebook.

```python
from scipy.io import loadmat

file_path = r"path/to/DREAMER.mat"
dreamer = loadmat(file_path)
DREAMER = dreamer["DREAMER"]
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project's notebook and execute the cells in the order documented in the notebook.

**Note:** The dataset is not included in the repository. Model results may vary if preprocessing settings, subject splits, dependencies, or random seeds are changed.

## 📁 Suggested Repository Structure

```text
eeg-emotion-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── DREAMER_Dataset_Analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrices/
│   └── model_comparison.csv
│
└── docs/
    └── methodology.md
```

This is a suggested structure. Only include files and folders that actually exist in your repository.

## ⚠️ Limitations

- The current dataset contains only 23 subjects.
- Emotion labels are self-reported and may not perfectly represent the emotion experienced during every five-second window.
- EEG signals vary between individuals, making subject-independent prediction challenging.
- Class imbalance affects some emotion categories.
- Artifact detection has been explored, but comprehensive artifact removal is not yet part of the established pipeline.
- The current classical models have limited test performance.
- Segment-level evaluation does not make every segment an independent emotion observation, since segments inherit trial-level labels.
- The initial evaluation uses four held-out test subjects, so results may depend on the selected subjects.

The reported metrics should be interpreted as preliminary experimental results rather than evidence of a highly accurate emotion-recognition system.

## 🔮 Future Improvements

Potential next steps include:

- More robust EEG artifact handling.
- Additional frequency-domain and differential entropy features.
- Feature selection and dimensionality reduction.
- Hyperparameter tuning using validation data.
- A 1D CNN trained directly on normalized EEG segments.
- Comparison of handcrafted features with deep-learning representations.
- Subject-wise cross-validation for more robust evaluation.
- Trial-level aggregation of segment predictions.
- Analysis of low/high emotion classification for all three dimensions.
- Visualization of EEG channels, feature importance, and model performance.
- Development of a simple interface for demonstrating predictions.

These improvements are planned extensions and are not claimed as completed features.

## 📚 References

1. Katsigiannis, S., & Ramzan, N. (2018). *DREAMER: A Database for Emotion Recognition Through EEG and ECG Signals from Wireless Low-cost Off-the-Shelf Devices.* IEEE Journal of Biomedical and Health Informatics, 22(1), 98–107. https://doi.org/10.1109/JBHI.2017.2688239

2. Scikit-learn documentation: https://scikit-learn.org/stable/

3. SciPy signal processing documentation: https://docs.scipy.org/doc/scipy/reference/signal.html

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/YOUR_USERNAME

## 📄 License

Add a license that matches your intended use and the rights you have to the code and any included materials. Check the DREAMER dataset's terms separately; a license for your code does not automatically grant rights to redistribute the dataset.

---

*This project is intended for educational and research purposes. EEG-based emotion predictions are experimental and should not be treated as definitive assessments of a person's emotional state.*
