# Understanding Privacy Awareness in Android App Descriptions Using Deep Learning

> This is the implementation of our framework to identify critical discrepancies between developer-described app behavior and permission usage in Android applications, as presented at CODASPY '20.
> See the [paper](https://pure.tugraz.at/ws/portalfiles/portal/26260487) by Feichtner and Gruber for more details.

In this repository you find our solution to:
 
- Linking permission usage to app descriptions,
- Extracting semantic knowledge from app descriptions,
- Explaining predictions using the LIME algorithm.

## Setup

```bash
conda install -c anaconda keras nltk pillow scikit-learn
conda install -c conda-forge langdetect lime
pip install androguard
```

Sample output the framework produces can be found at https://iaik.github.io/android_privacy/