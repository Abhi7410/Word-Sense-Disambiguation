<h1>README</h1
    <h5>Directory Structure</h5>

```.
├── codes
│   ├── Baseline
│   │   ├── codeKNN_Naive.ipynb
│   │   └── codeLEST.ipynb
│   └── BERT-WSD
│       ├── code.ipynb
│       ├── codeLSTM.ipynb
│       ├── createFeatures.py
│       ├── datasetPreProcess.py
│       ├── modelBERT.py
│       └── training.py
├── Data
│   ├── knn_bert3.npy
│   ├── naive_bert_embeddings.npy
│   ├── semcor3.csv
│   └── semcor_copy.csv
├── Final_ReportNLP.pdf
├── NLP_ppt.pdf
├── README.md
└── Results
    ├── senseval2.gold.key.txt
    └── senseval2_predictions.txt
```
<h5>Checkpoints of Model </h5>

[1000 Checkpoint](https://drive.google.com/drive/folders/1-2FgXOB7RRynmdHkgenUxkTY5rImbECp?usp=sharing)

 [2000-Checkpoint](https://drive.google.com/drive/folders/101BHK7vlTERTvoO-4RRPJ-IFsqY7piuh?usp=sharing)

<h5>Github </h5>

[Github Repo](https://github.com/Abhi7410/Word-Sense-Disambiguation)

<h5>Instructions to Run</h5>

For KNN, Naive Bayes and Lesk algoritm, notebook files are added so if dataset is available , then run block by block for both the files. 

For biLSTM model, follow  `codeLSTM.ipynb`

**Model Link** : [BiLSTM_Model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/abhishek_shar_students_iiit_ac_in/EZKmiJlq6UdJrujB0gR2MBMBv8xuW-lRMWXlf5Rv8XYonw?e=9g5hH6)

For BERT-model, prepared dataset is added already. If not present, then run `python3 datasetPreProcess.py` with editing xml file and gold.key.txt file in the code. After creating dataset, you can create features out of this dataset by running `python3 createFeatures.py`. For training, run `python3 training.py`which will use model and tokenizer defined in the file `model.py` and will start training.

<h5>Checkpoints of Model </h5>

[1000 Checkpoint](https://drive.google.com/drive/folders/1-2FgXOB7RRynmdHkgenUxkTY5rImbECp?usp=sharing)

[2000-Checkpoint](https://drive.google.com/drive/folders/101BHK7vlTERTvoO-4RRPJ-IFsqY7piuh?usp=sharing)

<h5>Dataset</h5>

Download this data folder from below link.

[Dataset](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/abhishek_shar_students_iiit_ac_in/EuY2tLElJ9RLmhpHkUs8zdMBE51Hetw8JfkMUR8UtL2vCg?e=5jyX9R)
