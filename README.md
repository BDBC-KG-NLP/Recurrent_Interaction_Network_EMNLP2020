## Recurrent Interaction Network for Jointly Extracting Entities and Classifying Relations

Dataset and code for the paper: **Recurrent Interaction Network for Jointly Extracting Entities and Classifying Relations**. Kai Sun, [Richong Zhang](http://act.buaa.edu.cn/zhangrc/), Samuel Mensah, Yongyi Mao, Xudong Liu. EMNLP 2019. [[pdf]](EMNLP2020_Kai_Sun_Rel.pdf)

## Requirement

- Python 3.6.7
- PyTorch 1.2.0
- NumPy 1.17.2
- 100-dimensional word vectors for nyt and webnlg [here](https://github.com/xiangrongzeng/copy_re)
- GloVe pre-trained word vectors for nyt10 and nyt11:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Extract the [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to the `dataset/glove/` folder.

## Usage

Training the model:

```bash
python train.py --dataset [dataset]
```

Prepare vocabulary files for nyt10 and nyt11 datasets (nyt10 and nyt11 will be uploaded soon):

```bash
python prepare_vocab.py --dataset [dataset]
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{Sun2020RIN,
  author    = {Kai Sun and
               Richong Zhang and
               Samuel Mensah and
               Yongyi Mao and
               Xudong Liu},
  title     = {Recurrent Interaction Network for Jointly Extracting Entities and Classifying Relations},
  year      = {2020}
}
```

## License

MIT
