# Voice Gender Recognition

This repository contains my personal reimplementation of the paper named ["Voice Gender Recognition Using Deep Learning"](https://www.atlantis-press.com/proceedings/msota-16/25868884) proposed by Mucahit Buyukyilmaz and Ali Osman Cibikdiken. All credits goes to the authors.

How to cite the original authors:

```bibtex
@inproceedings{buyukyilmaz2016voice,
  title={Voice gender recognition using deep learning},
  author={Buyukyilmaz, Mucahit and Cibikdiken, Ali Osman},
  booktitle={2016 International Conference on Modeling, Simulation and Optimization Technologies and Applications (MSOTA2016)},
  pages={409--411},
  year={2016},
  organization={Atlantis Press}
}
```

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install this package, firstly clone the repository to the directory of your choice using the following command:
```bash
git clone https://github.com/rafaelgreca/voice_gender_recognition.git
```

Finally, you need to create a conda environment and install the requirements. This can be done using conda or pip. For `conda` use the following command:
```bash
conda create --name vgr --file requirements.txt python=3.10
conda activate vgr
```

## Getting Started

### Downloading Dataset

Before continuing, to the code work properly you need to download the datasets correctly. If you install using other sources, the code might not work due to different folder organization and differents files/folders names. The dataset **MUST** be installed using [this link](https://www.kaggle.com/datasets/primaryobjects/voicegender).

### Directory Structure

```bash
├── __init__.py
├── LICENSE
├── notebook/
│   └── visualize_training.ipynb
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
└── train.py
```

Explaining briefly the main folders and files:

* `train.py`: the main file responsible to train the model.
* `src`: where the core functions are implemented, such as: models/datasets creation and input/output functions.
* `requirements.txt`: the requirement file for the installation step.
* `notebook`: where the notebook used to visualize the training/validation/test step is saved;

### Inputs Parameters

The following arguments **MUST** be passed:

* `-i` or `--input_dir`: the CSV file path.
* `-o` or `--output_dir`: the folder where the best models checkpoints will be saved on.
* `-l` or `--logging_dir`: the folder where the training loggings will be saved on.

### Running the Code

To train the model, run the following command:
```python3
python3 train.py -i CSV_FILE_PATH -o OUTPUT_DIR -l LOGGING_DIR
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

## Contact

Author: Rafael Greca Vieira - [GitHub](github.com/rafaelgreca/) - [LinkedIn](https://www.linkedin.com/in/rafaelgreca/) - rgvieira97@gmail.com