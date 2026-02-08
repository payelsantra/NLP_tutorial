# NLP_tutorial




## Finetuning BERT and SciBERT
The file BERT_training.ipynb contains the code for finetuning BERT and SciBERT. Trained models may be loaded and directly used for inference. The trained weights for BERT and SciBERT can be downloaded from the following links -
1. [BERT](https://drive.google.com/file/d/1s7f4gHhmV47YcHWMGo9z6ZYKthh4Icd9/view?usp=sharing)
2. [SciBERT](https://drive.google.com/file/d/16Wz5SI8dkxtdnje9Aq7duNMzFV9Ted66/view?usp=sharing)

## Results
The table presents results comparing the performance of BERT and SciBERT models without finetuning and after finetuning for 10 epochs. 
|        | w/o finetuning      | with finetuning      |
| Models | w/o | w/o | with  | with |
| Models | Accuracy | F1_macro | Accuracy  | F1_macro |
|--------|----------|-----------|---------|----------|
| BERT  |   0.78   |   0.79    |  0.86   |   0.86   |
| SciBERT  |   0.72   |   0.75    |  0.52   |   0.45   |


# Acknowledgements
1. [BERT](https://https://huggingface.co/docs/transformers/model_doc/bert)
2. [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)
