# NER Project

## Environment Set-Up

1. Have python 3 and pip installed.
2. Create a virtual environment and pip install packages from the requirements.txt file.


### Pretrained Models

- Create the `bert_data` directory, place the directories for each pretrained model that will be used. The hierarchy of the `bert_data` directory is as follows:

    - bert_data/
    - nb-bert-base-ner/
      - config.json
      - model.safetensors
      - pytorch_model.bin
      - ...
    - nbailab-base-ner-scandi/
      - config.json
      - model.safetensors
      - pytorch_model.bin
      - ...

### Finetuned Models

- Create the `bert_output` directory, create folders for each model type that will be used for finetuning. 
- In each of these model directories, new directories would need to be made for every new finetuned model that has been trained, e.g., `v1`, `v2`. In each of these directories, create a `cache` directory. 
- The hierarchy of the `bert_output` directory will be as follows:

    - bert_output/
        - scandi/
        - v1/
          - cache
        - v2/
          - cache

- An example of what the output directory for the finetuned model would look like is: `bert_output/scandi/v1`.

## Finetuning Process

The `NBAiLab_Finetuning_and_Evaluating_a_BERT_model_for_NER_and_POS.ipynb` jupyter notebook, from [link](https://github.com/NbAiLab/notram?tab=readme-ov-file#colab-notebooks), is used as the baseline script for the finetuning process. It is important to note, however, that there have been changes made to the version of the notebook in this repository to accommodate it to our use case.

- Before running make sure to go to setting cell and change the output_dir and cache_dir variables to a new folder under bert_output/scandi 
