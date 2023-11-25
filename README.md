# File Usage:
* `Model/`: Contains code related to models
* `Preprocess/`: Contains code for preprocessing
* `data/`: Contains the dataset, specified later
* `README.md`: This file
* `requirements.txt`: Required packages
* `utils.py`: Contains utility functions used by other classes

```
./
├── data
│   ├── dataset
│   │   ├── private_1_processed.csv
│   │   ├── public.csv
│   │   ├── public_processed.csv
│   │   └── training.csv
│   └── misc
│       └── 31_範例繳交檔案.csv
├── Model
│   ├── config0.json
│   ├── config1.json
│   ├── config2.json
│   ├── README.md
│   ├── test.py
│   ├── trainer.py
│   └── train.py
├── Preprocess
│   ├── analysis.py
│   ├── preprocess.py
│   └── README.md
├── README.md
├── requirements.txt
└── utils.py
```

# Execution Flow:
## 1. Environment
* OS: Ubuntu 20.04
* CPU: Intel i9-10900K
* RAM: 128 GB
* GPU: NVIDIA TITAN RTX *2
* Python: 3.11
* Create conda environment:
    ```
    $ conda create -n fintech python=3.11
    $ conda activate fintech
    $ python -m pip install -r requirements.txt
    ```

## 2. Data
The data folder must be like this:
```
data
├── dataset
│   ├── private_1_processed.csv
│   ├── public.csv
│   ├── public_processed.csv
│   └── training.csv
└── misc
    └── 31_範例繳交檔案.csv
```
## 3. Execution
```
# preprocess data
$ python Preprocess/preprocess.py

# train models
$ python Model/train.py

# inference: output will be in result/output.csv
$ python Model/test.py
```
* our trained models can be downloaded from [gdrive](https://drive.google.com/file/d/1JsPNSk1-PzP0JxN72lNnrlfUS52BRzjz/view?usp=sharing) 
