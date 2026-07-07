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
