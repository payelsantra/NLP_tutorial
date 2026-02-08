# NLP_tutorial
# Word2Vec → Sentence Similarity → BERT Models for Sentiment Classification

This repository provides a **hands-on tutorial** that helps students understand the transition from:

**traditional word-level embeddings → modern contextual sentence representations using Transformer models.**

The tutorial is divided into two parts:

- **Part 1 – Traditional word embeddings (Word2Vec / GloVe)**  Hands_on_NLP1.ipynb
- **Part 2 – Transformer-based models (BERT family)** BERT_training.ipynb
 
The main objective of this tutorial is to clearly demonstrate how **representation learning evolved** from:

- static word vectors  
to  
- contextual, sentence-level representations used in modern NLP systems.

## Prerequisites

Basic familiarity with:

- Python
- NumPy / pandas
- PyTorch (basic level)
- Jupyter Notebook

---

## Installation

It is recommended to use a virtual environment.

```bash
pip install numpy pandas scikit-learn gensim
pip install torch transformers datasets
pip install notebook
```




## Finetuning BERT and SciBERT
The file BERT_training.ipynb contains the code for finetuning BERT and SciBERT. Trained models may be loaded and directly used for inference. The trained weights for BERT and SciBERT can be downloaded from the following links -
1. [BERT](https://drive.google.com/file/d/1s7f4gHhmV47YcHWMGo9z6ZYKthh4Icd9/view?usp=sharing)
2. [SciBERT](https://drive.google.com/file/d/16Wz5SI8dkxtdnje9Aq7duNMzFV9Ted66/view?usp=sharing)

## Results
The table presents results comparing the performance of BERT and SciBERT models without finetuning and after finetuning for 10 epochs. 
| Model                    | Accuracy | F1 Score |
|--------------------------|----------|----------|
| Pretrained BERT          | 0.78     | 0.79     |
| Pretrained SciBERT       | 0.72     | 0.75     |
| Fine-tuned BERT (10 ep)  | 0.86     | 0.86     |
| Fine-tuned SciBERT (10 ep)| 0.52    | 0.45     |



# Acknowledgements
1. [BERT](https://https://huggingface.co/docs/transformers/model_doc/bert)
2. [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)
