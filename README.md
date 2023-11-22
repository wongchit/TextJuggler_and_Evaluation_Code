

<p align="center">TextJuggler and Evaluations based  TextAttack 🐙</p>




## About

This is an introduction to adversarial attack TextJuggler method and quality evaluation of adversarial examples. Code based on TextAttack.

## Setup

### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. TextJuggler is available through pip:

```bash
pip install -r ./requirements.txt 
```

## Attack Usage

### Running Attacks: `textattack attack --help`

The easiest way to try out an attack is via the command-line interface, `textattack attack`for TextJuggler or Baselines. 

Here are some concrete examples:

*TextFooler on BERT trained on the MR sentiment classification dataset*: 

```bash
textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 1000
```

OLM on DistilBERT trained on MNLI dataset*: 

```bash
textattack attack --model distilbert-base-uncased-MNLI --recipe olm --num-examples 1000
```

*TextJuggler* uses the imdb dataset on CNN and save the CSV file:  

```bash
textattack attack --model cnn-imdb --recipe textjuggler --num-examples 1000 --log-to-csv 
```

#### HuggingFace support: `transformers` models and `datasets` datasets

We also provide built-in support for [`transformers` pretrained models](https://huggingface.co/models) 
and datasets from the [`datasets` package](https://github.com/huggingface/datasets)! Here's an example of loading
and attacking a pre-trained model and dataset:

```bash
textattack attack --model-from-huggingface affahrizain/roberta-base-finetuned-jigsaw-toxic --dataset-from-huggingface affahrizain/jigsaw-toxic-comment --recipe pwws --num-examples 1000
```

**Tip:** INeed to add corresponding label classification parameters in the code.






## Evaluate Usage


### Preparation 

Use the "EvaluateText.py" file in the "Adversarial_example_evaluation" folder, in addition to the gpt2 language model that needs to be deposited in the "gpt2" folder.

### **Evaluate**

Users need to use the csv files generated by the attack for each type of adversarial attack method, in addition, the file naming format needs to follow the adversarial attack examples naming format inside the "results" folder.



The metrics required for the experimental comparison are shown in the file "EvaluateText.py".

## Contributing to TextAttack

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner. TextAttack is currently in an "alpha" stage in which we are working to improve its capabilities and design.

See [CONTRIBUTING.md](https://github.com/QData/TextAttack/blob/master/CONTRIBUTING.md) for detailed information on contributing.

## Citing TextAttack

If you use TextAttack for your research, please cite [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909).

```bibtex
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

