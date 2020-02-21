# ERNIE for the Rest of Us

The [ERNIE](https://github.com/PaddlePaddle/ERNIE) model is a pre-trained model for natural language processing
from Baidu. ERNIE 2.0 claimed [the top GLUE score](http://research.baidu.com/Blog/index-view?id=128) as of
December 11, 2019. However, ERNIE is implemented in Baidu's [Paddle deep learning framework](https://github.com/PaddlePaddle/Paddle)
whereas the rest of us typically use more popular framework, e.g. Tensorflow. This project provides an accessible 
package to recreate the ERNIE model in tensorflow initialized with the original ERNIE-trained weights.

# Installation and Usages

First install `ernie4us` python package via `pip`:

```
pip install ernie4us
```

Checkout the [ERNIE4us_demo.ipynb](ERNIE4us_demo.ipynb) jupyter notebook on how
to load and use the converted model using tensorflow API.

## Extracting the ERNIE model yourself

First download a pre-trained ERNIE 2.0 model in paddle format. The avialable variations and download location
can be found at [ERNIE's github project](https://github.com/PaddlePaddle/ERNIE#pre-trained-models--datasets).

Run the [`extract_ernie_params.sh`](extract_ernie_params.sh) script to extract model parameters and copy artifacts:
```
$ cd ~/ERNIE-for-the-rest-of-us
$ tar -C model_artifacts/ERNIE_Large_en_stable-2.0.0/paddle -zxf ~/Downloads/ERNIE_Large_en_stable-2.0.0.tar.gz
$ ./extract_ernie_params.sh
Usage: ./extract_ernie_params.sh ERNIE_MODEL_NAME
  ERNIE_MODEL_NAME the model identifier. Allowed identifiers are
    ERNIE_Base_en_stable-2.0.0
    ERNIE_Large_en_stable-2.0.0
$ ./extract_ernie_params.sh ERNIE_Large_en_stable-2.0.0
# .... some outputs ...
totally 392 persistables
copying artifacts...
total 2621456
-rw-r--r--@ 1 user_x  staff   330B Feb 20 16:36 ERNIE_Large_en_stable-2.0.0_config.json
-rw-r--r--@ 1 user_x  staff   226K Feb 20 16:36 ERNIE_Large_en_stable-2.0.0_vocab.txt
-rw-r--r--  1 user_x  staff   1.2G Feb 20 16:36 ERNIE_Large_en_stable-2.0.0_persistables.pkl
drwxr-xr-x  5 user_x  staff   160B Feb 20 16:36 paddle
ERNIE_Large_en_stable-2.0.0 parameters are exported to ./model_artifacts/ERNIE_Large_en_stable-2.0.0
```
Please note that the extraction script will install a specific version of tensorflow and thus overriding any
current tensorflow version. It is adviced that one sets up a specific virtual environment for this work.

After that, one can run the [ERNIE4us_verification.ipynb](ERNIE4us_verification.ipynb) Jupyter note book to verify
the corresponding ERNIE model recreated in tensorflow produces the same hidden features as the the orginal Paddle
implementation.

To use the extracted parameters in the model, put the artifact files in a subfolder named by the model name under the
path that you would be using in the `ernie4s.load_ernie_model` method, e.g.:
```python
import ernie4us
# Extracted ERNIE artifacts in /user/local/lib/ernie4us/ERNIE_Large_en_stable-2.0.0/
input_builder, ernie_tf_inputs, ernie_tf_outputs = ernie4us.load_ernie_model(
  model_name='ERNIE_Large_en_stable-2.0.0', 
  model_path='/user/local/lib/ernie4us')
```

# References and Credits

The modeling codes are adopted from the original [BERT](https://github.com/google-research/bert) 
and modified to accept ERNIE parameters

# License

Apache 2.0
