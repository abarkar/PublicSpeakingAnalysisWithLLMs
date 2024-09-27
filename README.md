# PublicSpeakingAnalysisWithLLMs
This is automatic public speaking analysis framework based on open-source Large Language Models (LLMs). This repository uses ollama library to sent requests through curl. 

## Table of Contents

- [Installation](#installation)
- [Pipeline](#pipeline)
- [Results](#results)
- [Model Parameters Used in Experiments](#modelparametersusedinexperiments)
- [Acknowledgements](#acknowledgements)

## Installation

Once you clones this repository, install requirements from the requirements file:

```bash
# Install dependencies
pip3 install -r requirements.txt
```

Or directly create conda environment:
```bash
# Install dependencies
conda env create -f environment.yml
```

We used conda version 23.10.0 on the machine with following properties:

Linux Precision-3581 6.5.0-1024-oem
Architecture: x86_64
CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 20 CPUs, 14 cores
Ubuntu 22.04.4 LTS

Than natigate to the ./analysis_configuration.sh file to check the system setup. There you can manipulate which subjective dimensions you would like to evaluate with LLMs or which criteria. 

## Pipeline

- Set Variables in ./analysis_configuration.sh file.
- Add your dataset to the ./data folder. Please, check the format of the other datasets to make sure that it is the same.
- Create prompts based on the transcripts that you provide:
    - For dimension evaluation:

```bash
# Initialise .sh file
chmod +x dimension_prompt_creation.sh

# Execute .sh file
./dimension_prompt_creation.sh
```
You should obtain prompts in the folder: ./prompts/$dataset/$dimension/$prompt_version/$clip/

    - For criteria evaluation:

```bash
# Initialise .sh file
chmod +x criteria_prompt_creation.sh

# Execute .sh file
./criteria_prompt_creation.sh
```
You should obtain prompts in the folder: ./prompts/$dataset/$criteria_category/$criteria/$prompt_version/$clip/

- Now you can exacute the LLM evaluation:

```bash
# Initialise .sh file for dimension evaluation
chmod +x dimension_analysis_with_LLM.sh

# Execute .sh file for dimension evaluation
./dimension_analysis_with_LLM.sh

# Initialise .sh file for criteria evaluation
chmod +x criteria_analysis_with_LLM.sh

# Execute .sh file for criteria evaluation
./criteria_analysis_with_LLM.sh

```

- Open-source models return answers in the .json format one file per prompt. In our case each transcript had corresponding prompt. Therefore, answers has to be aggregated, use this code:

```bash
# This code aggregated the scores
python3 aggregare_raw_outputs.py 
```

- Finally, not all of the prompts gave us the answer in the form of the one value. To extract the scores from the text provided by LLM we used scapy library and following script:

```bash
# This code aggregated the scores
python3 score_extraction_with_spacy.py 
```
# Results

This directory contains results that we obtained for the LLaMA2, LLaMA3, Mistral models for the persuasiveness analysis and criteria evaluation with LLMs on [3MT_French dataset ](https://lineact.cesi.fr/en/publications/introducing-the-3mt_french-dataset-to-investigate-the-timing-of-public-speaking-judgements/). The results saved under the different names than it is described in the provided scripts. This happen due to the different version of this repository. Meanwhile, different names does not change the content and presented results. 

# Model Parameters Used in Experiments

| **Model**      | **Parameters**                               |
|----------------|----------------------------------------------|
| **LLaMa2**     | temperature: 0.9 (in [0, 1]) <br> top\_p: 0.6 |
| **Mistral**    | temperature: 0.7 (in [0, 1.5]) <br> top\_p: 1 |
| **LLaMa3**     | temperature: 0.6 (in [0, 1]) <br> top\_p: 0.9 |
| **GPT-4o-mini**| temperature: 1 (in [0, 2]) <br> top\_p: 1     |



# Acknowledgements
This research was partly funded under the ANR REVITALISE grant ANR-21-CE33-0016-02.
