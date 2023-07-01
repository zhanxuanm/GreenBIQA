# GreenBIQA

## Collaborative project between Meta RP Media <> USC MCL on Lightweight BIQA

**Introduction**

GreenBIQA is a novel BIQA model, which aims at high performance, low computational complexity and a small model size. GreenBIQA adopts an unsupervised feature generation method and a supervised feature selection method to extract quality-aware features. Then, it trains an XGBoost regressor to predict quality scores of test images.

![image](./figures/GreenIQA-pipeline.png)

**Description**

Official python implementation for [GreenBIQA]([https://arxiv.org/abs/2206.14400](https://ieeexplore.ieee.org/abstract/document/9948927?casa_token=5P6D4gqVUfEAAAAA:J5J3QE3w61RurD9HBzPlwJlc_g9dmshZKcIj-gecQWH_E27R0459whsxtMOwcuQ5G-jAKnQg)).

- Mei, Z., Wang, Y. C., He, X., & Kuo, C. C. J. (2022, September). Greenbiqa: A lightweight blind image quality assessment method. In 2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP) (pp. 1-6). IEEE.

If you found the source codes useful, please consider citing our paper:

```
@inproceedings{mei2022greenbiqa,
  title={GreenBIQA: A Lightweight Blind Image Quality Assessment Method},
  author={Mei, Zhanxuan and Wang, Yun-Cheng and He, Xingze and Kuo, C-C Jay},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```


## Usage

**Dependencies**

Codes are written in python3. Install the required packages 
by running:

    pip install -r requirements.txt

**Data**

- Download the [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/), [LIVE-C](https://live.ece.utexas.edu/research/ChallengeDB/index.html), [KADID-10K](http://database.mmsp-kn.de/kadid-10k-database.html) and [KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html) dataset.

- Put all the images under ``data``.

- ``mos.csv`` is required to map images to their corresponding 
  user annotations. An example ``mos.csv`` can be found 
  [here](data/test/mos.csv). Only column *'image_name'* and 
  *'MOS'* are required.

**Run the codes**

The feature extractor and regressor are trained using an
authentic public dataset 
[KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html).
The trained models can be found under ``model``.
To re-train the feature extractor and regressor using your 
own data, run:

    python main.py --do_train --data_dir data

To predict the MOS using the trained model, run:

    python main.py --do_test --save --data_dir data

**Other usage**

```
  --do_train        If specified, re-train the feature extractor and regressor.
  --do_test         If specified, predict the MOS for images under the data directory.
  --yuv             If specified, using yuv420 as the input format.
  --height          Specify the height of input images, required if yuv.
  --width           Specify the width of input images, required if yuv.
  --save            If specified, save the extracted image features.
  --data_dir        Specify the data directory.
  --output_dir      Specify the path for logs and saved features.
  --model_dir       Specify the path to load/store the models. 
  --num_aug         Number of cropped patches per image.
```

**Benchmarks**

- Model size
    - Feature extractor: 608 KB 
    - Regressor: 2.6 MB
- Feature extraction time
    - Take 5.499 seconds to extract 928 384x384 patches.
    - In average, take 0.0237 ms to predict the quality score for one image.
- Performance
    
![image](./figures/benchmark.png)
