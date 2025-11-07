# Distracted Driver Detection 

## Objectives:

In this project, we develop deep learning classifiers to detect distracted drivers from **multimodal time series** data. The three main algorithms developed are: a Transformer classifier (self-attention encoder), a CNN/ResNet classifier, and an LSTM classifier.

## Dataset:

FordChallenge from Kaggle. 

## Jupyter Notebooks:

[nb]Objectives_and_Data_Exploration.ipynb <br>
[nb]Pre-processing_Modeling.ipynb

## Code:
-- Core models and data structures<br>
&nbsp;⟙ TransformerTimeseriesClassifier.py <br>
&nbsp;⏐ &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; ├ Embedding.py <br>
&nbsp;⏐ &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; ⟘ SelfAttentionEncoder.py  <br>
&nbsp;⏐ &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; ⟘ MultiHeadSelfAttention.py <br>
&nbsp;├ CNNTimeseriesClassifier.py <br>
&nbsp;⟘ RNNTimeseriesClassifier.py <br>


-- Utilities and parameters <br>
&nbsp; ⟙ Trainer.py <br>
&nbsp; ⟘ parameters.py<br>





