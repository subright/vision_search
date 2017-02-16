# vision_search

## Abstract
With the growing popularity of mobile devices, user web logs are more heterogeneous than ever, across an increased number of devices and websites. As a result, identifying users with similar usage patterns within these large sets of web logs is increasingly challenging and critical for personalization and user experience in many areas, from recommender systems to digital marketing.

In this work, we explore the use of visual search for top-k user retrieval based on similar user behavior. We introduce a convolution neural network (WimNet) that learns latent representation from a set of web logs represented as images. Specifically, it contains two convolution layers take row- and column-wise convolutions to capture user behavior across multiple devices and websites and learns latent representation and reconstructs a transition matrix between user activities of given web logs. To evaluate our method, we conduct conventional top-k retrieval task on the simulated dataset, and the preliminary analysis results suggest that our method produces more accurate and robust results regardless of the complexity of query log. 

## Description
Here are more descriptions about the files.

### Requirements
- Python opencv: brew install opencv
- numpy, scipy, PIL, matplotlib, pandas

### Files

-  png4_[5, 10, 15, 20, 25] - User behavior images with [5, 10, 15, 20, 25]% of noise over its length.

- log4_[5, 10, 15, 20, 25] - Files contatining user activity logs which are generated from 'cross_device_img_generator_noise.py' during image generation.

- log4_dict - A file contains user activity list, [device id: site id: normalized frequency]

- cross_device_img_generator_noise.py - Based on the parameters (# of devices, # of logs, time length, and so on), it generates simularted logs according to

- logencoder_trans.py - Given a set of user bahevior logs, it trains CNN for learning feature representation

### Citation

- Sungchul Kim, Sana Malik, Nedim Lipka, and Eunyee Koh, WimNet: Vision Search for Web Logs, WWW'17 (poster)
