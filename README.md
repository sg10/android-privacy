# Understanding Privacy Awareness in Android App Descriptions Using Deep Learning

This is the implementation of our framework to identify critical discrepancies between developer-described app behavior and permission usage in Android applications, as presented at CODASPY '20.
See the [paper](https://graz.pure.elsevier.com/en/publications/understanding-privacy-awareness-in-android-app-descriptions-using) by Feichtner and Gruber for more details.

In this repository you find our solution for:
 
- Linking permission usage to app descriptions,
- Extracting semantic knowledge from app descriptions,
- Explaining predictions using the LIME algorithm.

## Abstract

> Permissions are a key factor in Android to protect users' privacy. As it is often not obvious why applications require certain permissions, developer-provided descriptions in Google Play and third-party markets should explain to users how sensitive data is processed. Reliably recognizing whether app descriptions cover permission usage is challenging due to the lack of enforced quality standards and a variety of ways developers can express privacy-related facts.  
>
> We introduce a machine learning-based approach to identify critical discrepancies between developer-described app behavior and permission usage. By combining state-of-the-art techniques in natural language processing (NLP) and deep learning, we design a convolutional neural network (CNN) for text classification that captures the relevance of words and phrases in app descriptions in relation to the usage of _dangerous permissions_. Our system predicts the likelihood that an app requires certain permissions and can warn about descriptions in which the requested access to sensitive user data and system features is textually not represented.
>
> We evaluate our solution on 77,000 real-world app descriptions and find that we can identify individual groups of _dangerous permissions_ with a precision between 71% and 93%. To highlight the impact of individual words and phrases, we employ a model explanation algorithm and demonstrate that our technique can successfully bridge the semantic gap between described app functionality and its access to security- and privacy-sensitive resources.

## Setup

```bash
conda install -c anaconda keras nltk pillow scikit-learn
conda install -c conda-forge langdetect lime
pip install androguard
```

Sample output the framework produces can be found at https://sg10.github.io/android-privacy/