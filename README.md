# Grey Seal Vocalisation Classification

## Project Overview

This project applies machine learning techniques to the classification of grey seal vocalisations using audio recordings and annotation data.
Spectrogram representations of annotated seal calls are extracted from raw audio and used to train a convolutional neural network (CNN) to distinguish between different call types.

The project focuses on a binary classification task between **Rupe A** and **Rupe B** vocalisations and is structured to demonstrate a clear, reproducible machine learning workflow from raw data preprocessing to model evaluation.

This work was completed as part of a machine learning module and follows the recommended GitHub Classroom workflow.

---

## Repository Structure

```bash

machine-learning-projects/
│
├── notebooks/
│ ├── 01_preprocessing.ipynb # Data extraction, spectrogram generation, dataset creation
│ ├── 02_training.ipynb # CNN training and evaluation
│ └── 03_research.ipynb # Optional experimentation (if used)
│
├── src/
│ └── spectrogram_utils.py # Reusable audio and spectrogram utilities
│
├── BackgroundInfo/
│ └── Grey Seal - Software, Paper and Sounds.pptx
│ ├── Spectrogram.ipynb
│ ├── Spectrogram.py
│ └── spectrogram.png
│
├── data_processed/ # Locally generated datasets (ignored by git)
├── README.md
└── .gitignore
 ```

 ---

## Data

Raw audio files and annotation data are **not included** in this repository.
They must be downloaded separately and stored locally.

The location of the raw dataset is defined explicitly in `01_preprocessing.ipynb` using an absolute path on the local machine.

Processed datasets (e.g. `.npz` files) are generated locally and ignored by version control.

---

## Environment & Dependencies

The project was developed using Python and the following key libraries:

- NumPy
- Pandas
- SciPy
- Matplotlib
- scikit-learn
- TensorFlow / Keras

It is recommended to run the notebooks in a Python environment with TensorFlow installed.
GPU acceleration is optional but supported.

---

## How to Run the Project

1. Clone the repository.
2. Download and extract the Grey Seal dataset locally.
3. Open `notebooks/01_preprocessing.ipynb` and update the `DATA_DIR` path to point to the local dataset.
4. Run all cells in `01_preprocessing.ipynb` to generate the processed dataset.
5. Open `notebooks/02_training.ipynb` to train and evaluate the baseline CNN model.

---

## Results

The baseline CNN achieves approximately **77% test accuracy** on the Rupe A vs Rupe B classification task.
Performance is evaluated using accuracy, precision, recall, and a confusion matrix.
Detailed results and discussion are provided directly within the training notebook.

---

## Notes

References and background literature are cited within the Jupyter notebooks rather than duplicated here.

This project was completed by Hugo Camacho Romero.
