# Sentiment Scope

This page contains the code used in the work "Learning Latent Sentiment Scopes for Entity-Level Sentiment Analysis" published at AAAI 2017.

## Contents
1. [Usage](#usage)
2. [SourceCode](#sourcecode)
3. [Citation](#citation)


## Usage

Prerequisite: JRE (1.8 or later) and Ant

1. Run "make" to compile the source code.
```sh
make
```

2. Run "make init" to initialize files and folders.
```sh
make init
```

3. Run "make <experiment>" for experiments mentioned in the paper.
```sh
make <experiment>
```

Note that the <experiment> refers to anyone of the following experiments


```
sentimentscope_latent_english
sentimentscope_latent_nohiddeninfo_english
sentimentscope_latent_withpostag_english
sentimentscope_latent_wordembedding_english
sentimentscope_nonlatent_english
sentimentscope_semimarkov_nonlatent_english
sentimentscope_semimarkov_latent_english
baseline_collapse_english
baseline_pipeline_english

sentimentscope_latent_spanish
sentimentscope_latent_nohiddeninfo_spanish
sentimentscope_latent_withpostag_spanish
sentimentscope_latent_wordembedding_spanish
sentimentscope_nonlatent_spanish
sentimentscope_semimarkov_nonlatent_spanish
sentimentscope_semimarkov_latent_spanish
baseline_collapse_spanish
baseline_pipeline_spanish
```

## SourceCode

The source code is written in Java, which can be found under the "src" folder.

Note that this is a Maven project. Therefore you can import this project as a Maven project in eclipse.


## Citation
If you use this software for research, please cite our paper as follows:

```
@inproceedings{li2017learning,
  title={Learning Latent Sentiment Scopes for Entity-Level Sentiment Analysis.},
  author={Li, Hao and Lu, Wei},
  booktitle={AAAI},
  pages={3482--3489},
  year={2017}
}
```


Email to hao_li@mymail.sutd.edu.sg if any inquery.
