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


