# Wav2SQL

This repository contains code for Wav2SQL.



## Usage

### Step 1: Download third-party datasets & dependencies

Download the datasets: [Spider](https://yale-lily.github.io/spider) and [WikiSQL](https://github.com/salesforce/WikiSQL). In case of Spider, make sure to download the `08/03/2020` version or newer.
Unpack the datasets somewhere outside this project to create the following directory structure:
```
/path/to/data
├── spider
│   ├── database
│   │   └── ...
│   ├── dev.json
│   ├── dev_gold.sql
│   ├── tables.json
│   ├── train_gold.sql
│   ├── train_others.json
│   └── train_spider.json
└── wikisql
    ├── dev.db
    ├── dev.jsonl
    ├── dev.tables.jsonl
    ├── test.db
    ├── test.jsonl
    ├── test.tables.jsonl
    ├── train.db
    ├── train.jsonl
    └── train.tables.jsonl
```

To work with the WikiSQL dataset, clone its evaluation scripts into this project:
``` bash
mkdir -p third_party
proxychains git clone https://github.com/salesforce/WikiSQL third_party/wikisql
```

### Step 2: Build and run the Docker image

We have provided a `Dockerfile` that sets up the entire environment for you.
It assumes that you mount the datasets downloaded in Step 1 as a volume `/mnt/data` into a running image.
Thus, the environment setup for RAT-SQL is:
``` bash
docker build -t wav2sql .
docker run --rm -m4g -v /path/to/data:/mnt/data -it wav2sql
```
Note that the image requires at least 4 GB of RAM to run preprocessing.
By default, [Docker Desktop for Mac](https://hub.docker.com/editions/community/docker-ce-desktop-mac/) and [Docker Desktop for Windows](https://hub.docker.com/editions/community/docker-ce-desktop-windows) run containers with 2 GB of RAM.
The `-m4g` switch overrides it; alternatively, you can increase the default limit in the Docker Desktop settings.

> If you prefer to set up and run the codebase without Docker, follow the steps in `Dockerfile` one by one.
> Note that this repository requires Python 3.7 or higher and a JVM to run [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).

### Step 3: Run the experiments

Every experiment has its own config file in `experiments`.
The pipeline of working with any model version or dataset is: 

``` bash
python run.py preprocess experiment_config_file  # Step 3a: preprocess the data
python run.py train experiment_config_file       # Step 3b: train a model
python run.py eval experiment_config_file        # Step 3b: evaluate the results
```

Use the following experiment config files to reproduce our results:

* Spider, GloVE version: `experiments/spider-glove-run.jsonnet`
* Spider, BERT version (requires a GPU with at least 16GB memory): `experiments/spider-bert-run.jsonnet`
* WikiSQL, GloVE version: `experiments/wikisql-glove-run.jsonnet`



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
