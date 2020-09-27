# HENIN
Official implementation of HENIN model(EMNLP2020)

## Abstract
In computational detection of cyberbullying, existing work largely focused on building generic classifiers that rely exclusively on text analysis of social media sessions. Despite their empirical success, we argue that a critical missing piece is the model explainability, i.e., why a particular piece of media session is detected as cyberbullying. In this paper, therefore, we propose a novel deep model, HEterogeneous Neural Interaction Networks (HENIN), for explainable cyberbullying detection. HENIN contains the following components: a comment encoder, a post-comment co-attention sub-network, and session-session and post-post interaction extractors. Extensive experiments conducted on real datasets exhibit not only the promising performance of HENIN, but also highlight evidential comments so that one can understand why a media session is identified as cyberbullying.

<img src="/images/HENIN-model.PNG" width="70%">

## Datasets:
- Instagram dataset: contains image description and user comments.
- Vine dataset: a mobile application website that allows users to record and edit a few seconds looping videos. Each vine session also contains video description and user comments.

Dataset available at https://sites.google.com/site/cucybersafety/home/cyberbullying-detection-project/dataset

## Requirements
* python>=3.5
* keras
* gensim
* network
* nltk

## Citation
Hsin-Yu Chen and Cheng-Te Li. HENIN: Learning Heterogeneous Neural Interaction Networks for Explainable Cyberbullying Detection on Social Media. The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP'20)
