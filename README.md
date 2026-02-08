# NLP_tutorial
# Word2Vec → Sentence Similarity → BERT Models for Sentiment Classification

This repository provides a **hands-on tutorial** that helps students understand the transition from:

**traditional word-level embeddings → modern contextual sentence representations using Transformer models.**

The tutorial is divided into two parts:

- **Part 1 – Traditional word embeddings (Word2Vec / GloVe)**  ```Hands_on_NLP1.ipynb```
- **Part 2 – Transformer-based models (BERT family)** ```BERT_training.ipynb```
 
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

### Types of BERT models used in this project (BERT vs SciBERT)

| Feature | BERT | SciBERT |
|------|------|--------|
| Training Objectives | MLM (Masked Language Model), NSP (Next Sentence Prediction) | MLM (Masked Language Model), NSP (Next Sentence Prediction) |
| Pretraining Corpus | General-domain text (Wikipedia + BooksCorpus) | Scientific text (biomedical and computer science papers) |
| Domain Focus | General-purpose language understanding | Scientific and technical language understanding |
| Vocabulary | General-domain WordPiece vocabulary | Domain-specific vocabulary (scientific terminology and symbols) |
| Tokenization | WordPiece tokenizer | WordPiece tokenizer (trained on scientific corpus) |
| Next Sentence Prediction (NSP) | Yes | Yes |
| Sentence Embeddings | Uses the `[CLS]` token representation | Uses the `[CLS]` token representation |
| Architecture | Same as original BERT architecture | Same as BERT architecture |
| Model Size | Base / Large (e.g., 12 or 24 layers) | Base-sized architecture (typically 12 layers) |
| Number of Layers | Configurable, typically 12 (Base) or 24 (Large) | Typically 12 (Base) |
| Training Duration | Large-scale pretraining on general data | Large-scale pretraining on scientific papers |
| Language Coverage | Broad, everyday and web-style English | Scientific writing style (papers, abstracts, technical text) |
| Performance | Strong general benchmark performance | Significantly better performance on scientific NLP tasks (e.g., NER, classification, relation extraction in scientific domains) |
| Typical Use Case | General NLP tasks such as sentiment analysis, QA, and NLI | Scientific and domain-specific NLP tasks (biomedical and computer science text) |



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

## Observations

From the results shown above, the following observations can be made:

- Fine-tuning substantially improves the performance of the standard BERT model.  
  The accuracy increases from **0.78 to 0.86** and the F1 score from **0.79 to 0.86**, indicating that task-specific supervision is highly effective for general-domain BERT.

- In contrast, SciBERT does **not** benefit from fine-tuning on this dataset.  
  Its performance drops sharply after fine-tuning (accuracy from **0.72 to 0.52**, F1 score from **0.75 to 0.45**).

- The superior performance of BERT over SciBERT suggests that the dataset used in this project is likely **general-domain and sentiment-oriented**, rather than scientific or technical in nature.

- SciBERT is pretrained on scientific literature, and its vocabulary and language representations are specialized for scientific writing.  
  When applied to a general sentiment classification task, this domain mismatch can lead to weaker representations and unstable fine-tuning.

- The large degradation in SciBERT’s fine-tuned performance may also indicate **overfitting or optimization instability**, especially if the training dataset is small or contains informal language (e.g., reviews, social media text).

- Overall, these results highlight an important practical insight:  
  **domain-specific pretrained models are not always better**, and choosing a pretrained model whose pretraining domain matches the target task is crucial for achieving good performance.


# Models used
1. [BERT](https://https://google-bert/bert-base-uncased)
2. [SciBERT](https://allenai/scibert_scivocab_uncased)
