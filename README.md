# DQU-CIR

### [SIGIR 2024] - Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.15875)
![GitHub Repo stars](https://img.shields.io/github/stars/haokunwen/DQU-CIR)

This is the official implementation of our paper "Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval".

### Requirements

- python == 3.9.18
- pytorch == 2.0.1
- open-clip-torch == 2.23.0
- opencv-python == 4.9.0
- CUDA == 12.4
- 1 A100-40G GPU

### Datasets

As our method focuses on the raw-data level multimodal fusion, the dataset preparation steps might require some extra attention. We've released all the necessary files for you to download easily!

- Shoes:
  - We noticed that the images of the Shoes dataset are currently unavailable from the [original source](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset), so we provided it on [Google Drive](https://drive.google.com/file/d/18DEWXvuyp2vXHv4tAw6fcD2ehEtrvyIL/view?usp=sharing). Unzip the file and make the images by their names inside the `./data/Shoes/womens_*` folders.
  - We also corrected the modification text in Shoes by the [pyspellchecker](https://pypi.org/project/pyspellchecker/) tool. The file is available at `./data/Shoes/correction_dict_shoes.json`.
  - **For obtaining the Unified Textual Query:** Although Shoes dataset already contains the image captions at `./data/Shoes/captions_shoes.json`, we also generate the image captions by [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b). The generated image caption file is located at `./data/Shoes/image_captions_shoes.json`.
  - **For obtaining the Unified Visual Query:** Similar to FashionIQ, We extract the key words descriptions with the prompt like:
    ```
    Analyze the provided text, which details the differences between two images. Extract and list the distinct features of the target image mentioned in the caption, separating each feature with a comma. Ensure to eliminate any redundancies and correct typos. For instance, given the input 'has long heels and is black instead of white', the response should be 'long heels, black'. If the input is 'change from red to blue', respond with 'blue'. If there are no changes, respond with 'same'. The input text for this task is:
    ```
    The extracted key words files are available at `./data/Shoes/keywords_in_mods_shoes.json`.

### Usage

- FashionIQ: `python train.py --dataset {'dress'/'shirt'/'toptee'} --lr 1e-4 --clip_lr 1e-6 --fashioniq_split={'val-split'/'original-split'}`
- Shoes: `python train.py --dataset 'shoes' --lr 5e-5 --clip_lr 5e-6`
- CIRR:  
  Training: `python train.py --dataset 'cirr' --lr 1e-4 --clip_lr 1e-6 `  
  Testing: `python cirr_test_submission.py --i xx `
- Fashion200K: `python train.py --dataset 'fashion200k' --lr 1e-4 --clip_lr 1e-6`

### Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{dqu_cir,
    author = {Wen, Haokun and Song, Xuemeng and Chen, Xiaolin and Wei, Yinwei and Nie, Liqiang and Chua, Tat-Seng},
    title = {Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval},
    booktitle = {Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {229-239},
    publisher = {{ACM}},
    year = {2024}
}
```
