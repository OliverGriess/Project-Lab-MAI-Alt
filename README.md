# Weakly Supervised Deepfake Localization Pipeline

This repository provides the implementation of LipFD, along with the code required to run, train, and evaluate the model on LAV-DF using different localization strategies. Additionally, it includes preprocessing scripts to convert LAV-DF into the expected format for LipFD.

## Project Structure

```
src/
├── LipFD/                  
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   ├── train/               # Training scripts
│   ├── eval/                # Evaluation scripts
│   ├── options/             # Configuration options
│   └── checkpoints/         # Model checkpoints
├── DDRCF/   
│   ├── finetune_predict.py  # Finetuning and evaluation             
│   └── checkpoints/         # Model checkpoints
├── localize/                
│   ├── TimeSerieClustering.py  # Time series clustering and localization method comparison implementation
│   ├── DDM.py                  # Drift Detection implementation
│   ├── hysteresis_tresholding.py  # Hysteresis Tresholding implementation
│   ├── visualize_pred_dist.py   # Visualization of the prediction score distribution
│   └── run_full_eval.py    # Entry point to run the full evaluation of all localization methods
└── eval/                    # Evaluation utilities
```

## Setup and Installation

1. Clone the repository:

```bash
git clone git@github.com:NiclasDev63/PL_MAI_Group_04.git
cd PL_MAI_Group_04
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

The preprocessing utilities are located in `src/LipFD/data/`. This module handles:

1. Choose the script for the dataset split of LAV-DF you want to preprocess (i.e. train or val/test)
    - ``src/LipFD/data/train_preprocessing.py`` for the train set
    - ``src/LipFD/data/val_preprocessing.py`` for the test and validation set
2. Fill out the missing parameters located at the top of the respective file
3. Execute the skript
    ```bash
    python src/LipFD/data/train_preprocessing.py
    ```
    OR
    ```bash
    python src/LipFD/data/val_preprocessing.py
    ```



### 2. Training

To train the model:

1. Configure your training parameters in `src/LipFD/options/`
2. Run the training script:

```bash
python src/LipFD/train/train.py
```

We recommend you run

```bash
python src/LipFD/train/train.py -h
```

OR 

have a look at the ``src/LipFD/options`` directory to get a overview of all available options

### 3. Evaluation

To evaluate the model:

```bash
python src/LipFD/eval/evaluate.py --model_path [path_to_checkpoint]
```

This will:

- Load the trained model
- Run inference on the test set
- Generate evaluation metrics

### 4. Localization

The localization module provides several approaches for deepfake localization:

1. **Time Series Clustering** (`TimeSerieClustering.py`):

   - Implements clustering-based detection
   - Handles temporal patterns in the data

2. **Drift Detection Method** (`DDM.py`):

   - Detects concept drift in the data

3. **Hysteresis Thresholding** (`hysteresis_tresholding.py`):
   - Implements adaptive thresholding

To run localization:

```bash
python src/localize/run_full_eval.py
```

## Visualization

The project includes visualization tools to help analyze results:

```bash
python src/localize/visualize_pred_dist.py --predictions [pred_file] --output [output_dir]
```

## License

[Add license information here]

## :mailbox: Citation

If you use this code in your research, please cite:

```
@software{PL_MAI_Group_4,
  author = {Grein, Oliver and Griess, Oliver and Masmoudi, Oussama and Gregor, Niclas},
  doi = {},
  month = {3},
  title = {{Weakly Supervised Deepfake Localization Pipeline}},
  url = {https://github.com/OliverGriess/Project-Lab-MAI-Alt},
  year = {2025}
}
```
