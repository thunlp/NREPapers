# Must-read papers on NRE
NRE: Neural Relation Extraction.

Contributed by [Tianyu Gao](https://github.com/gaotianyu1350) and [Xu Han](https://github.com/THUCSTHanxu13).

We released [OpenNRE](https://github.com/thunlp/OpenNRE), an open-source framework for neural relation extraction. This repository provides several relation extraction methods and a easy-to-use training and testing framework.

## Journal and Conference papers:

### Distantly Supervised Datasets and Training Methods

1. **Learning to Extract Relations from the Web using Minimal Supervision.**
_Razvan C. Bunescu, Department of Computer Sciences._
ACL 2007.
[paper](http://www.aclweb.org/anthology/P07-1073)
    > We present a new approach to relation extraction that requires only a handful of training examples. Given a few pairs of named entities known to exhibit or not exhibit a particular relation, bags of sentences containing the pairs are extracted from the web.

1. **Distant Supervision for Relation Extraction without Labeled Data.**
_Mike Mintz, Steven Bills, Rion Snow, Dan Jurafsky._
ACL-IJCNLP 2009.
[paper](http://delivery.acm.org/10.1145/1700000/1690287/p1003-mintz.pdf?ip=133.130.111.179&id=1690287&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1536630304_c802fdeac941523f3207459f0142021b)
    > Our experiments use Freebase, a large semantic database of several thousand relations, to provide distant supervision.

1. **Modeling Relations and Their Mentions without Labeled Text.**
_Sebastian Riedel, Limin Yao, Andrew McCallum._
ECML 2010.
[paper](https://link.springer.com/content/pdf/10.1007%2F978-3-642-15939-8_10.pdf)
    > We present a novel approach to distant supervision that can alleviate this problem based on the following two ideas: First, we use a factor graph to explicitly model the decision whether two entities are related, and the decision whether this relation is mentioned in a given sentence; second, we apply constraint-driven semi-supervision to train this model without any knowledge about which sentences express the relations in our training KB.

1. **Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations.**
_Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, Daniel S. Weld._
ACL-HLT 2011.
[paper](http://delivery.acm.org/10.1145/2010000/2002541/p541-hoffmann.pdf?ip=133.130.111.179&id=2002541&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1536594661_4b61c377a78c4b4339d41cb438d8bdb8)
    > This paper presents a novel approach for multi-instance learning with overlapping re- lations that combines a sentence-level extrac- tion model with a simple, corpus-level compo- nent for aggregating the individual facts.

### Neural Encoders

1. **Semantic Compositionality through Recursive Matrix-Vector Spaces.**
_Richard Socher, Brody Huval, Christopher D. Manning, Andrew Y. Ng._
EMNLP-CoNLL 2012.
[paper](http://delivery.acm.org/10.1145/2400000/2391084/p1201-socher.pdf?ip=59.66.131.241&id=2391084&acc=OPEN&key=BF85BBA5741FDC6E%2E587F3204F5B62A59%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1533972543_ff020f3c692b2117fa2230dfe7872f07)
    > We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length.

1. **Relation Classification via Convolutional Deep Neural Network.**
_Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, Jun Zhao._
COLING 2014.
[paper](http://www.aclweb.org/anthology/C14-1220)
    > We exploit a convolutional deep neural network (DNN) to extract lexical and sentence level features. Our method takes all of the word tokens as input without complicated pre-processing.

1. **Classifying Relations by Ranking with Convolutional Neural Networks.**
_C´ıcero Nogueira dos Santos, Bing Xiang, Bowen Zhou._
ACL 2015.
[paper](https://www.aclweb.org/anthology/P15-1061)
    > In this work we tackle the relation classification task using a convolutional neural network that performs classification by ranking (CR-CNN).

1. **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks.**
_Daojian Zeng, Kang Liu, Yubo Chen, Jun Zhao._
EMNLP 2015.
[paper](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)
    > We propose a novel model dubbed the Piecewise Convolutional Neural Networks (PCNNs) with multi-instance learning to address these two problems.

1. **Neural Relation Extraction with Selective Attention over Instances.**
_Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun._
ACL 2016.
[paper](http://www.aclweb.org/anthology/P16-1200)
    > Distant supervision inevitably accompanies with the wrong labelling problem, and these noisy data will substantially hurt the performance of relation extraction. To alleviate this issue, we propose a sentence-level attention-based model for relation extraction.

1. **Adversarial Training for Relation Extraction.**
_Yi Wu, David Bamman, Stuart Russell._
EMNLP 2017.
[paper](http://www.aclweb.org/anthology/D17-1187)
    > Adversarial training is a mean of regularizing classification algorithms by generating adversarial noise to the training
data. We apply adversarial training in relation extraction within the multi-instance multi-label learning framework.

1. **A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction.**
_Tianyu Liu, Kexiang Wang, Baobao Chang, Zhifang Sui._
EMNLP 2017.
[paper](http://www.aclweb.org/anthology/D17-1189)
    > We introduce an entity-pair level denoise method which exploits semantic information from correctly labeled entity pairs to correct wrong labels dynamically during training.

1. **DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction.**
_Pengda Qin, Weiran Xu, William Yang Wang._
[paper](https://arxiv.org/pdf/1805.09929.pdf)
    > We introduce an adversarial learning framework, which we named DSGAN, to learn a sentence-level true-positive generator. Inspired by Generative Adversarial Networks, we regard the positive samples generated by the generator as the negative samples to train the discriminator.

1. **Reinforcement Learning for Relation Classification from Noisy Data.**
_Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu._
AAAI 2018.
[paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)
    > We propose a novel model for relation classification at the sentence level from noisy data. The model has two modules: an instance selector and a relation classifier. The instance selector chooses high-quality sentences with reinforcement learning and feeds the selected sentences into the relation classifier, and the relation classifier makes sentence-level prediction and provides rewards to the instance selector.
    
1. **Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning.**
_Pengda Qin, Weiran Xu, William Yang Wang._
[paper](https://arxiv.org/pdf/1805.09927.pdf)
    > We explore a deep reinforcement learning strategy to generate the false-positive indicator, where we automatically recognize false positives for each relation type without any supervised information.
