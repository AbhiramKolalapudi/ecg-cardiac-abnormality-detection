# ECG Cardiac Abnormality Detection

## Day 1 - Project Foundation, ECG Fundamentals & Dataset Planning

**Date:** 6 July 2026

### Objectives
- Set up the development environment.
- Initialize Git and GitHub.
- Design the project architecture.
- Learn the fundamentals of ECG and cardiac electrical activity.
- Explore ECG datasets.
- Select the primary dataset for the project.

### Work Completed
- Created the GitHub repository **ecg-cardiac-abnormality-detection**.
- Cloned the repository locally.
- Opened the project in Visual Studio Code.
- Created a Python virtual environment.
- Designed a production-style project structure for:
  - Machine Learning
  - FastAPI Backend
  - React Frontend
  - Documentation
- Added:
  - `README.md`
  - `.gitignore`
  - `requirements.txt`
  - `LICENSE`
- Planned the overall AI pipeline and project architecture.
- Learned how the heart generates electrical impulses.
- Studied the cardiac conduction system:
  - SA Node
  - AV Node
  - Bundle of His
  - Bundle Branches
  - Purkinje Fibers
- Understood how ECG machines record electrical activity.
- Learned the ECG waveform:
  - P Wave
  - PR Interval
  - QRS Complex
  - T Wave
- Learned the difference between electrodes and leads.
- Understood that ECG is a time-series signal.
- Learned about ECG sampling frequency and why the MIT-BIH dataset uses 360 Hz.
- Explored PhysioNet ECG datasets.
- Compared:
  - MIT-BIH Arrhythmia Database
  - MIT-BIH Supraventricular Arrhythmia Database (SVDB)
  - PTB-XL
  - Chapman-Shaoxing ECG Database
- Selected the MIT-BIH Arrhythmia Database as the primary training dataset.
- Learned the structure and purpose of:
  - `.dat`
  - `.hea`
  - `.atr`
- Understood how heartbeat annotations are used for supervised machine learning.
- Planned the complete AI workflow from ECG upload to prediction and report generation.

### Concepts Learned
- Cardiac electrophysiology
- Cardiac conduction system
- ECG signal acquisition
- ECG waveform interpretation
- Electrodes vs Leads
- Time-series data
- Sampling frequency
- ECG preprocessing pipeline
- Heartbeat annotations
- Supervised learning
- PhysioNet
- MIT-BIH Arrhythmia Database
- ECG dataset selection
- AI inference pipeline
- Production project architecture

### Files Added
- `README.md`
- `.gitignore`
- `requirements.txt`
- `LICENSE`
- `docs/learning-journal.md`

### Git Commit
```
Initialize project and plan ECG AI pipeline
```

### Next Session
- Download the MIT-BIH Arrhythmia Database.
- Learn the WFDB Python library.
- Read `.dat`, `.hea` and `.atr` files.
- Plot a real ECG signal.
- Visualize heartbeat annotations.
- Extract the first heartbeat.
- Begin building the preprocessing pipeline.


## Day 2 - Reading ECG Data, Visualization & Heartbeat Extraction

**Date:** 7 July 2026

### Objectives

* Download and organize the MIT-BIH Arrhythmia Database.
* Learn the WFDB file format and Python library.
* Read and visualize a real ECG recording.
* Understand heartbeat annotations.
* Learn the fundamentals of heartbeat extraction.

### Work Completed

* Downloaded and organized the MIT-BIH Arrhythmia Database.
* Learned the purpose of `.dat`, `.hea`, and `.atr` files.
* Installed and configured the WFDB Python library.
* Loaded Record 100 using `wfdb.rdrecord()`.
* Explored the `Record` object, including:

  * Sampling frequency
  * Signal length
  * Lead names
  * Signal units
  * ECG waveform (`p_signal`)
* Converted ECG sample numbers into time (seconds).
* Plotted the first ECG waveform using Matplotlib.
* Loaded annotations using `wfdb.rdann()`.
* Explored `annotation.sample` and `annotation.symbol`.
* Learned the difference between heartbeat and non-heartbeat annotations.
* Understood common annotation symbols (`N`, `V`, `A`, `L`, `R`, `+`, `~`).
* Learned NumPy Boolean masking and vectorized indexing.
* Overlayed heartbeat annotations on the ECG waveform.
* Understood why heartbeat annotations are aligned with the R peak.
* Learned why machine learning models classify individual heartbeats instead of entire ECG recordings.
* Extracted the first heartbeat window from a real ECG signal.
* Designed a scalable heartbeat extraction pipeline for all ECG records.

### Concepts Learned

* WFDB library
* ECG metadata
* Record and Annotation objects
* NumPy arrays
* Boolean masking
* ECG visualization
* Sample-to-time conversion
* Heartbeat annotations
* R peak
* Heartbeat extraction
* Boundary checking
* ECG preprocessing
* Machine learning dataset creation

### Files Added / Modified

* `src/data/load_record.py`

### Git Commit

```
Read ECG records and build heartbeat extraction pipeline
```

### Next Session

* Refactor heartbeat extraction into reusable functions.
* Process all 48 MIT-BIH records.
* Build the complete heartbeat dataset (`X` and `y`).
* Analyze class distribution.
* Prepare the dataset for preprocessing and model training.


## Day 3 - Dataset Architecture and Single Record Processing

**Date:** 9 July 2026

### Objectives

* Design the machine learning dataset structure.
* Refactor heartbeat extraction into reusable components.
* Build a production-style dataset pipeline architecture.
* Process an entire ECG record using the extraction pipeline.
* Validate heartbeat extraction using real ECG data.

### Work Completed

* Defined the machine learning sample representation for the project.
* Decided to perform heartbeat-level classification instead of record-level classification.
* Designed the dataset structure:

  * `X.shape = (num_heartbeats, 250)`
  * `y.shape = (num_heartbeats,)`
* Designed a reusable heartbeat extraction function:

  * `extract_heartbeat()`
* Implemented boundary validation for heartbeat extraction.
* Decided to skip heartbeats whose extraction windows exceed signal boundaries.
* Selected a heartbeat window of:

  * 100 samples before the R peak
  * 150 samples after the R peak
  * 250 samples total
* Learned NumPy slicing behavior and the difference between views and copies.
* Implemented `.copy()` to ensure extracted heartbeats are independent from the original ECG signal.
* Designed a reusable record processing function:

  * `process_record()`
* Implemented:

  * heartbeat label filtering
  * heartbeat extraction
  * invalid beat handling
  * dataset construction for a single record
* Selected the initial heartbeat classes:

  * `N`
  * `V`
  * `A`
  * `L`
  * `R`
* Chose to use Python sets for efficient label membership checks.
* Built a notebook for validating the extraction pipeline.
* Processed MIT-BIH Record 100 using the complete pipeline.
* Generated:

  * `X_record`
  * `y_record`
* Verified dataset dimensions:

  * `X_record.shape = (2271, 250)`
  * `y_record.shape = (2271,)`
* Analyzed class distribution for Record 100:

  * `N = 2237`
  * `A = 33`
  * `V = 1`
* Visualized extracted heartbeats from each class.
* Verified that the R peak consistently appears at sample index `100`.
* Validated that the heartbeat extraction pipeline behaves correctly on real ECG data.
* Finalized the dataset module architecture.

### Concepts Learned

* Heartbeat-level classification
* Dataset design for machine learning
* Feature matrices and target vectors
* NumPy views vs copies
* Dataset construction pipelines
* Record-level processing
* Label filtering
* Boundary validation
* Class imbalance
* Medical AI dataset design
* Production ML architecture
* Incremental validation and scaling strategies

### Files Added / Modified

* `ml/src/datasets/heartbeat_extractor.py`
* `ml/src/datasets/record_processor.py`
* `ml/notebooks/record_100_validation.ipynb`

### Git Commit

```
Build heartbeat extraction and record processing pipeline
```

### Next Session

* Create `constants.py` for centralized configuration.
* Build `dataset_builder.py`.
* Process all 48 MIT-BIH records.
* Construct the complete heartbeat dataset (`X` and `y`).
* Analyze global class distribution.
* Investigate dataset imbalance.
* Prepare the dataset for preprocessing and model training.

## Day 4 - Configuration Management and Multi-Record Dataset Construction

**Date:** 10 July 2026

### Objectives

* Centralize dataset configuration.
* Build a scalable dataset construction pipeline.
* Process multiple ECG records automatically.
* Learn software engineering principles for ML systems.
* Analyze initial multi-record class distribution.

### Work Completed

* Designed and implemented centralized project configuration using:

  * `constants.py`
* Moved hardcoded values into configuration:

  * heartbeat extraction parameters
  * target heartbeat classes
  * dataset paths
  * test record selection
* Learned the concept of configuration separation:

  * logic vs configuration
  * reproducibility
  * experiment management
* Added:

  * `SAMPLES_BEFORE_R`
  * `SAMPLES_AFTER_R`
  * `HEARTBEAT_LENGTH`
  * `TARGET_CLASSES`
  * `TEST_RECORDS`
  * `MITDB_PATH`
* Learned Python import resolution and module search paths.
* Learned the difference between:

  * current working directory
  * `sys.path`
  * package imports
  * relative file paths
* Standardized ML imports using:

  * `from src...`
* Learned why notebooks and scripts may behave differently with imports.
* Replaced fragile relative dataset paths with `pathlib`.
* Introduced:

  * `PROJECT_ROOT`
  * robust path construction
* Designed and implemented:

  * `dataset_builder.py`
* Built a scalable multi-record dataset construction pipeline.
* Processed multiple MIT-BIH records automatically.
* Learned why NumPy arrays should not be concatenated repeatedly inside loops.
* Used list accumulation followed by a single `np.concatenate()` operation.
* Learned the computational complexity difference between:

  * repeated concatenation (`O(n²)`)
  * single concatenation (`O(n)`)
* Processed the first development subset:

  * Records `100`
  * `101`
  * `102`
  * `103`
  * `104`
* Generated the first multi-record dataset:

  * `X.shape = (6484, 250)`
  * `y.shape = (6484,)`
* Performed the first multi-record class distribution analysis:

  * `N = 6439`
  * `A = 38`
  * `V = 7`
* Observed that:

  * class distributions vary significantly between patients
  * some arrhythmia classes may be absent from subsets of the dataset
* Learned the importance of:

  * patient-level train/test splitting
  * avoiding heartbeat-level data leakage
  * class imbalance handling strategies

### Concepts Learned

* Configuration management
* Reproducible ML experiments
* Python import system
* `sys.path`
* Current working directory
* Absolute vs relative imports
* `pathlib`
* Robust file path handling
* Dataset orchestration
* Multi-record dataset construction
* Computational complexity
* NumPy memory allocation behavior
* Efficient concatenation strategies
* Patient-level dataset splitting
* Medical dataset leakage
* Class imbalance in healthcare AI

### Files Added / Modified

* `ml/src/config/constants.py`
* `ml/src/datasets/dataset_builder.py`
* `ml/src/datasets/heartbeat_extractor.py`
* `ml/src/datasets/record_processor.py`

### Git Commit

```text
Build scalable ECG dataset construction pipeline
```

### Next Session

* Add all MIT-BIH records to configuration.
* Process the complete MIT-BIH dataset.
* Analyze global class distribution.
* Study patient-level distribution differences.
* Design train/validation/test patient splits.
* Prepare preprocessing and normalization pipeline.


## Day 5 - Full Dataset Construction, Patient Metadata and Leakage Analysis

**Date:** 11 July 2026

### Objectives

* Scale the dataset pipeline from the development subset to the complete MIT-BIH dataset.
* Introduce patient metadata tracking.
* Study class imbalance and patient-level distributions.
* Understand medical data leakage and patient-level splitting.
* Investigate the impact of ignored annotation classes.

### Work Completed

* Extended dataset construction from the 5-record development subset to the complete MIT-BIH Arrhythmia Database.

* Added the complete MIT-BIH record list to configuration.

* Processed all 48 MIT-BIH recordings successfully.

* Generated the first full-scale dataset:

  * `X.shape = (100022, 250)`
  * `y.shape = (100022,)`
  * `patient_ids.shape = (100022,)`

* Introduced patient metadata tracking:

  * `patient_ids[i]` stores the originating patient/record for each heartbeat.
  * Preserved alignment between:

    * `X[i]`
    * `y[i]`
    * `patient_ids[i]`

* Verified successful processing of all MIT-BIH records:

  * `100`
  * `101`
  * `102`
  * ...
  * `234`

* Learned the concept of heartbeat provenance and metadata preservation in machine learning pipelines.

* Discussed architectural ownership of metadata and decided that:

  * `dataset_builder.py` owns patient metadata generation.
  * `record_processor.py` remains responsible only for signal processing.

* Decided to:

  * store patient IDs as integers.
  * preserve labels as symbolic annotations (`N`, `A`, `V`, `L`, `R`) until model training.

* Decided to persist patient metadata alongside datasets for reproducibility and future experiments.

* Added:

  * `MITBIH_RECORDS`
  * `TEST_RECORDS`

* Discussed explicit configuration versus automatic dataset discovery.

* Adopted a fail-fast philosophy for dataset construction:

  * missing or corrupted records should stop the build immediately.

### Dataset Statistics

#### Global Dataset Size

* Total heartbeats: `100022`
* Total records: `48`

#### Class Distribution

| Class | Count |
| ----- | ----: |
| N     | 75020 |
| L     |  8072 |
| R     |  7255 |
| V     |  7129 |
| A     |  2546 |

### Patient-Level Analysis

* Built patient-level class distribution tables using Pandas.
* Observed strong patient-specific concentration of arrhythmias.

Examples:

* Patient `232` contributes `1382` atrial beats (`A`).

* Patient `109` contributes `2491` left bundle branch block beats (`L`).

* Patient `118` contributes `2165` right bundle branch block beats (`R`).

* Patient `208` contributes `992` ventricular beats (`V`).

* Learned that many patients contain only a subset of classes.

* Learned that patient-level splitting is significantly more difficult than random heartbeat splitting.

### Concepts Learned

* Patient metadata tracking
* Heartbeat provenance
* Medical dataset leakage
* Identity leakage
* Patient-level train/test splitting
* Group-aware validation
* Group stratification
* Explicit configuration
* Fail-fast pipeline design
* Class imbalance in medical AI
* Exploratory data analysis with Pandas
* Long vs wide tabular representations

### Annotation Analysis

* Investigated record `102` to understand why only `103` beats were retained.

* Discovered the original annotation distribution:

  * `/` = 2028 (paced beats)
  * `N` = 99
  * `f` = 56 (fusion paced beats)
  * `+` = 5 (rhythm changes)
  * `V` = 4

* Learned that:

  * paced beats belong to a different clinical problem.
  * rhythm markers are not heartbeats.
  * target class selection directly determines dataset composition.

### Files Added / Modified

* `ml/src/config/constants.py`
* `ml/src/datasets/dataset_builder.py`

### Git Commit

```text
Scale ECG dataset pipeline to full MIT-BIH dataset and add patient metadata tracking
```

### Next Session

* Design patient-level train/validation/test splits.
* Learn group-aware splitting strategies.
* Study normalization and scaling of ECG signals.
* Understand normalization leakage.
* Decide where preprocessing belongs in the project architecture.
* Begin building the preprocessing pipeline.

## Day 6 - Patient Splitting and Preprocessing Architecture

**Date:** 12 July 2026

### Objectives

* Design patient-aware train/validation/test splitting.
* Study group-aware splitting methods for medical datasets.
* Prevent patient leakage.
* Design preprocessing architecture for model training.
* Begin implementing the preprocessing pipeline.

### Work Completed

* Studied why heartbeat-level train/test splitting causes medical data leakage.

* Learned the difference between:

  * heartbeat-level splitting
  * patient-level splitting

* Understood that models can memorize patient-specific ECG morphology rather than learning actual diseases.

* Learned the concept of identity leakage in medical machine learning.

* Studied group-aware splitting methods.

* Installed and configured:

  * `scikit-learn`

* Explored:

  * `StratifiedGroupKFold`

* Learned how group-aware splitting attempts to preserve:

  * patient independence
  * class distributions

* Designed a patient split persistence strategy.

* Decided to save:

  * `train_patients.npy`
  * `val_patients.npy`
  * `test_patients.npy`

instead of heartbeat indices.

* Added split configuration to `constants.py`:

  * `TRAIN_RATIO`
  * `VALIDATION_RATIO`
  * `TEST_RATIO`
  * `RANDOM_STATE`
  * `PROCESSED_DATA_PATH`
  * `SPLITS_PATH`

* Designed and implemented:

  * `splitter.py`

* Implemented:

  * `save_patient_splits()`
  * `load_patient_splits()`
  * `generate_patient_splits()`
  * `get_patient_splits()`

* Learned how to reconstruct heartbeat datasets using:

  * `np.isin()`

* Built patient masks for:

  * training
  * validation
  * testing

* Performed split validation and class distribution analysis.

* Observed limitations of patient-level stratification in small medical datasets.

* Discovered that preserving patient independence is often more important than achieving perfect class balance.

* Finalized the patient splitting strategy for the project.

### Concepts Learned

* Medical data leakage
* Identity leakage
* Patient-level splitting
* Group-aware validation
* StratifiedGroupKFold
* Metadata persistence
* Split reproducibility
* Boolean masking
* `np.isin()`
* Dataset reconstruction
* Preprocessing architecture
* Production ML pipeline design

### Files Added / Modified

* `ml/src/preprocessing/splitter.py`
* `ml/src/config/constants.py`

### Git Commit

```text
Implement patient-aware dataset splitting and split persistence
```

### Next Session

* Implement ECG normalization.

* Study normalization leakage.

* Build:

  * `normalize_heartbeat()`
  * `normalize_dataset()`

* Design the preprocessing pipeline.

* Prepare datasets for model training.


## Day 7 - Manual Patient Splitting and Preprocessing Pipeline

**Date:** 13 July 2026

### Objectives

* Improve patient-level train/validation/test splits.
* Eliminate limitations of automatic stratified group splitting.
* Implement ECG normalization.
* Build label encoding.
* Design and implement the preprocessing pipeline.
* Prepare the dataset for model training.

### Work Completed

* Evaluated the limitations of `StratifiedGroupKFold` for small medical datasets.

* Observed that automatic splitting produced missing classes in validation and test sets.

* Studied the tradeoff between:

  * perfect class balance
  * patient independence
  * reproducibility

* Designed a manually curated patient split protocol.

### Final Patient Split

#### Training Patients

* 31 patients

#### Validation Patients

* 8 patients

#### Test Patients

* 9 patients

### Final Split Statistics

#### Training

| Class | Count |
| ----- | ----: |
| N     | 57922 |
| V     |  5251 |
| L     |  4492 |
| R     |  4387 |
| A     |  1992 |

#### Validation

| Class | Count |
| ----- | ----: |
| N     | 10792 |
| V     |  1603 |
| L     |  1457 |
| R     |  1338 |
| A     |   250 |

#### Test

| Class | Count |
| ----- | ----: |
| N     |  6306 |
| L     |  2123 |
| R     |  1530 |
| A     |   304 |
| V     |   275 |

* Verified:

  * no patient leakage
  * all classes present in every split
  * approximately 74/15/11 split ratio
  * reproducible experimental protocol

* Refactored `splitter.py` to use deterministic patient splits instead of automatic generation.

* Preserved split persistence using:

  * `train_patients.npy`
  * `val_patients.npy`
  * `test_patients.npy`

### ECG Normalization

* Studied the purpose of ECG normalization.

* Learned the difference between:

  * Min-Max normalization
  * Global standardization
  * Per-heartbeat Z-score normalization

* Selected per-heartbeat Z-score normalization.

* Learned that normalization removes:

  * baseline offsets
  * gain differences
  * electrode placement effects
  * machine calibration differences

* Learned that normalization preserves:

  * QRS morphology
  * P wave morphology
  * T wave morphology
  * temporal relationships
  * relative amplitudes

* Implemented:

  * `normalize_heartbeat()`
  * `normalize_dataset()`

* Added numerical stability using:

  * `EPSILON = 1e-8`

* Used vectorized NumPy operations for dataset normalization.

### Normalization Validation

* Verified:

  * mean ≈ 0
  * standard deviation ≈ 1
  * morphology preservation
  * correct dataset-wide behavior

* Validated normalization on all 100,022 heartbeats.

### Label Encoding

* Studied explicit label mappings versus automatic encoders.
* Selected explicit deterministic mappings.

Implemented:

* `LABEL_TO_INDEX`
* `INDEX_TO_LABEL`

Mappings:

* N → 0
* A → 1
* V → 2
* L → 3
* R → 4

Implemented:

* `encode_labels()`
* `decode_labels()`

Validated:

* `decode(encode(y)) == y`

### Preprocessing Pipeline

* Designed and implemented:

  * `pipeline.py`

Implemented:

* patient-level splitting
* normalization
* label encoding

Pipeline flow:

MIT-BIH Records
→ Heartbeat Extraction
→ Dataset Construction
→ Patient Splitting
→ Normalization
→ Label Encoding
→ Model-Ready Dataset

### Concepts Learned

* Deterministic experimental protocols
* Manual split design for medical datasets
* ECG normalization
* Z-score normalization
* Numerical stability
* Broadcasting in NumPy
* Vectorized preprocessing
* Label encoding
* Reversible mappings
* Preprocessing pipelines
* Separation of concerns
* Dataset immutability
* Reproducible ML experiments

### Files Added / Modified

* `ml/src/preprocessing/splitter.py`
* `ml/src/preprocessing/normalizer.py`
* `ml/src/preprocessing/encoder.py`
* `ml/src/preprocessing/pipeline.py`
* `ml/src/config/constants.py`

### Git Commit

```
Build ECG preprocessing pipeline and finalize patient split protocol
```

### Project Status

✅ Dataset Engineering Phase Complete

✅ Preprocessing Phase Complete

### Next Session

* Study why CNNs work for ECG signals.
* Compare MLPs, RNNs and CNNs.
* Design the baseline 1D CNN architecture.
* Build the training pipeline.
* Train the first baseline model.
* Establish baseline performance metrics.

## Day 8 - CNN Fundamentals and ECG Model Architecture Design

**Date:** 14 July 2026

### Objectives

* Understand why CNNs are effective for ECG classification.
* Compare MLPs, RNNs and CNNs for heartbeat classification.
* Select an appropriate deep learning architecture for the project.
* Design the baseline ECG classifier.
* Learn the fundamentals of PyTorch training pipelines.
* Prepare the project for model implementation.

### Work Completed

* Studied ECG heartbeat classification as a morphology recognition problem.

* Learned why traditional fully connected networks perform poorly on ECG data:

  * ignore temporal locality
  * fail to exploit waveform structure
  * are sensitive to small signal shifts

* Compared:

  * MLPs
  * RNNs/LSTMs
  * CNNs

for ECG heartbeat classification.

* Learned that heartbeat classification depends primarily on local morphology rather than long-range temporal dependencies.

* Selected a CNN architecture as the primary modeling approach.

* Studied convolution filters and local feature extraction.

* Learned the concepts of:

  * kernels
  * channels
  * feature maps
  * weight sharing
  * translation invariance

* Learned how convolution filters slide across ECG signals to detect waveform patterns.

* Compared:

  * small kernels
  * large kernels

and selected:

* `kernel_size = 5`

for the baseline architecture.

* Studied the purpose of max pooling.

* Learned that pooling:

  * reduces dimensionality
  * reduces computation
  * improves robustness to small temporal shifts

* Selected a 1D CNN architecture instead of a 2D CNN architecture.

* Designed the baseline ECG classifier:

```text
Input
→ Conv1D
→ ReLU
→ MaxPool
→ Conv1D
→ ReLU
→ MaxPool
→ Flatten
→ Dense Layers
→ Output Layer
```

* Learned the role of flattening and fully connected layers in CNN classifiers.

* Studied multi-class classification loss functions.

* Selected:

  * `CrossEntropyLoss`

for heartbeat classification.

* Analyzed class imbalance in the training set.

* Calculated that an always-normal classifier would achieve approximately:

  * `78.2%`

training accuracy despite being clinically useless.

* Learned why accuracy is insufficient for evaluating medical AI systems.

* Selected primary evaluation metrics:

  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix

* Studied weighted loss functions for imbalanced datasets.

* Selected balanced class weighting for the baseline model.

* Designed the PyTorch training architecture.

* Defined ownership boundaries for:

  * `models/`
  * `training/`

* Learned the distinction between:

  * `Dataset`
  * `DataLoader`

* Designed the tensor flow for the complete training pipeline.

### Concepts Learned

* CNNs
* Convolution filters
* Kernel size
* Feature maps
* Channels
* Weight sharing
* Translation invariance
* Pooling
* Hierarchical feature learning
* Flattening
* Dense layers
* Cross entropy loss
* Class imbalance
* Precision
* Recall
* F1 score
* Confusion matrix
* PyTorch Dataset
* DataLoader
* Mini-batch training
* Tensor dimensions

### Files Added / Modified

* None

### Git Commit

```
No code changes - model architecture and training pipeline design
```

### Next Session

* Implement `baseline_cnn.py`.
* Build `ECGDataset`.
* Create DataLoaders.
* Implement weighted loss functions.
* Build the training loop.
* Train the first CNN model.
* Establish baseline metrics.
* Begin model evaluation.

## Day 9 - PyTorch Pipeline, CNN Implementation and First Model Training

**Date:** 15 July 2026

### Objectives

* Implement the baseline CNN architecture in PyTorch.
* Build the PyTorch dataset pipeline.
* Learn mini-batch training fundamentals.
* Implement weighted loss functions for class imbalance.
* Build the training infrastructure.
* Train the first ECG CNN model.

### Work Completed

* Designed and implemented:

  * `ECGDataset`

* Learned the responsibilities of:

  * `__init__()`
  * `__len__()`
  * `__getitem__()`

* Learned the distinction between:

  * `Dataset`
  * `DataLoader`

* Understood that:

  * `Dataset` returns a single sample.
  * `DataLoader` creates mini-batches.

* Built PyTorch datasets for:

  * training
  * validation
  * testing

* Built DataLoaders for:

  * training
  * validation
  * testing

* Selected:

  * `batch_size = 64`

* Verified DataLoader output dimensions:

  * `signals.shape = (64, 1, 250)`
  * `labels.shape = (64,)`

### Baseline CNN Implementation

* Implemented:

  * `baseline_cnn.py`

* Built the baseline architecture:

```text
Input (1,250)
→ Conv1D(1→32, kernel_size=5)
→ ReLU
→ MaxPool1D
→ Conv1D(32→64, kernel_size=5)
→ ReLU
→ MaxPool1D
→ Flatten
→ Linear(3776→128)
→ Linear(128→5)
```

* Verified tensor dimensions throughout the network:

```text
(64,1,250)
→ (64,32,246)
→ (64,32,123)
→ (64,64,119)
→ (64,64,59)
→ (64,3776)
→ (64,128)
→ (64,5)
```

* Verified successful forward propagation using real training batches.

### Class Imbalance Handling

* Studied weighted loss functions for imbalanced medical datasets.

* Implemented:

  * `compute_class_weight()`

* Generated balanced class weights:

| Class | Weight |
| ----- | -----: |
| N     | 0.2557 |
| A     | 7.4341 |
| V     | 2.8202 |
| L     | 3.2967 |
| R     | 3.3756 |

* Implemented weighted:

  * `CrossEntropyLoss`

* Learned that minority classes contribute more strongly to training loss.

### Optimizer and Backpropagation

* Selected:

  * `Adam`

as the baseline optimizer.

* Learned the complete training cycle:

```text
Forward Pass
→ Loss Calculation
→ Gradient Computation
→ Weight Update
```

* Studied:

  * gradients
  * backpropagation
  * optimizer updates

* Learned the role of:

  * `loss.backward()`
  * `optimizer.step()`
  * `optimizer.zero_grad()`

* Understood that:

  * forward pass uses model weights
  * backward pass computes gradients
  * optimizer updates weights using gradients

### Training Infrastructure

* Designed and implemented:

  * `trainer.py`

* Implemented:

  * `train_one_epoch()`

* Designed and implemented:

  * `train.py`

* Implemented:

  * device selection
  * dataset construction
  * preprocessing pipeline integration
  * model initialization
  * class weight generation
  * loss function creation
  * optimizer creation
  * epoch loop

* Learned the purpose of:

  * `model.train()`

### First Training Run

* Successfully trained the first ECG CNN model end-to-end.

* Verified successful learning behavior.

Training loss progression:

| Epoch |   Loss |
| ----: | -----: |
|     1 | 0.1646 |
|     2 | 0.0731 |
|     3 | 0.0556 |
|     4 | 0.0407 |
|     5 | 0.0334 |
|   ... |    ... |
|    20 | 0.0069 |

* Confirmed that:

  * forward propagation works
  * backpropagation works
  * gradients are computed correctly
  * optimizer updates are functioning correctly
  * the CNN successfully learns ECG morphology

### Concepts Learned

* PyTorch Dataset
* DataLoader
* Mini-batch training
* Conv1D
* Flattening
* Fully connected layers
* CrossEntropyLoss
* Class weighting
* Adam optimizer
* Forward propagation
* Backpropagation
* Gradients
* Weight updates
* Epochs
* Batches
* Training loops
* GPU vs CPU execution
* Model parameter optimization

### Files Added / Modified

* `ml/src/training/ecg_dataset.py`
* `ml/src/models/baseline_cnn.py`
* `ml/src/training/trainer.py`
* `ml/src/training/train.py`
* `ml/src/config/constants.py`

### Git Commit

```text
Implement baseline ECG CNN and complete first end-to-end training pipeline
```

### Project Status

✅ Dataset Engineering Complete

✅ Preprocessing Pipeline Complete

✅ Baseline CNN Implemented

✅ PyTorch Training Pipeline Complete

✅ First Successful Model Training Complete

### Next Session

* Implement validation infrastructure.

* Learn:

  * `model.eval()`
  * `torch.no_grad()`

* Build:

  * `validate_one_epoch()`

* Compute:

  * validation loss
  * accuracy
  * precision
  * recall
  * F1 score

* Generate confusion matrices.

* Detect overfitting and evaluate generalization performance.

* Implement model checkpoint saving based on validation performance.
