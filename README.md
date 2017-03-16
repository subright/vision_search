# WimNet: Vision Search for Web Logs

## Abstract
With the growing popularity of mobile devices, user web logs are more heterogeneous than ever, across an increased number of devices and websites. As a result, identifying users with similar usage patterns within these large sets of web logs is increasingly challenging and critical for personalization and user experience in many areas, from recommender systems to digital marketing.

In this work, we explore the use of visual search for top-k user retrieval based on similar user behavior. We introduce a convolution neural network (WimNet) that learns latent representation from a set of web logs represented as images. Specifically, it contains two convolution layers take row- and column-wise convolutions to capture user behavior across multiple devices and websites and learns latent representation and reconstructs a transition matrix between user activities of given web logs. To evaluate our method, we conduct conventional top-k retrieval task on the simulated dataset, and the preliminary analysis results suggest that our method produces more accurate and robust results regardless of the complexity of query log. 

## Description
Here are more descriptions about the files.

### Requirements
- Python opencv: brew install opencv
- numpy, scipy, PIL, matplotlib, pandas

### Files

- png4_[5, 10, 15, 20, 25] - User behavior images with [5, 10, 15, 20, 25]% of noise over its length.

- log4_[5, 10, 15, 20, 25] - Files contatining user activity logs which are generated from 'cross_device_img_generator_noise.py' during image generation.

- log4_dict - A file contains user activity list, [device id: site id: normalized frequency]

- cross_device_img_generator_noise.py - Based on the parameters (# of devices, # of logs, time length, and so on), it generates simularted logs according to predefined probability (temporarily removed)

- logencoder_trans.py - Given a set of user bahevior logs, it trains CNN for learning feature representation (temporarily removed)

## Experiment

### Description
Experimental setting: top-5 retrieval results from 5 different approaches (RAW, HIST, SIFT, CHI, OURS from left to right)
Dataset: 4 devices (row) and 10 websites (color)

### Model
- RAW: L1-distance between raw images
- HIST: Euclidean distance between two vectors of color histogram in RGB channel
- SIFT: Average distance between two sets of SIFT descriptor
- CHI: polar distance between two subsequent sets of user activity
- OURS: Euclidean distance between two latent representations

### Result
Q5 (+5% noise): 2 devices X 1 websites - Most approaches retrieve reasonable results except SIFT

<img src="https://github.com/subright/vision_search/blob/master/result/5_q.png"/ width="150">
<p align="center">
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_5_5_raw.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_5_5_hist.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_5_5_sift.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_5_5_seq.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_5_5_ae.png" width="150"/>
</p>

Q20 (+5% noise): 2 devices X 2 websites - Most approaches retrieve reasonable results except RAW and SIFT

<img src="https://github.com/subright/vision_search/blob/master/result/20_q.png"/ width="150">
<p align="center">
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_20_5_raw.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_20_5_hist.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_20_5_sift.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_20_5_seq.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_20_5_ae.png" width="150"/>
</p>

Q85 (+5% noise): 3 devices X 2 websites - Most approaches retrieve reasonable results except RAW and SIFT

<img src="https://github.com/subright/vision_search/blob/master/result/85_q.png"/ width="150">
<p align="center">
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_5_raw.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_5_hist.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_5_sift.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_5_seq.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_5_ae.png" width="150"/>
</p>

Q85 (+15% noise): 3 devices X 2 websites - Our approaches find the most reliable results compared to others. The results of HIST show similar patterns in terms of visited websites, it cannot capture device transition properly.

<img src="https://github.com/subright/vision_search/blob/master/result/85_q_20.png"/ width="150">
<p align="center">
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_15_raw.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_15_hist.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_15_sift.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_15_seq.png" width="150"/>
  <img src="https://github.com/subright/vision_search/blob/master/result/png4_85_15_ae.png" width="150"/>
</p>

## Citation

- Sungchul Kim, Sana Malik, Nedim Lipka, and Eunyee Koh, WimNet: Vision Search for Web Logs, WWW'17 (poster)
