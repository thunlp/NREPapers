# Must-read papers on NRE
NRE: Neural Relation Extraction.

Contributed by [Tianyu Gao](https://github.com/gaotianyu1350) and [Xu Han](https://github.com/THUCSTHanxu13).

We released [OpenNRE](https://github.com/thunlp/OpenNRE), an open-source framework for neural relation extraction. This repository provides several relation extraction methods and an easy-to-use training and testing framework.

## Reviews

1. Nguyen Bach, Sameer Badaskar. **A review of relation extraction**. [[paper]](https://www.cs.cmu.edu/~nbach/papers/A-survey-on-Relation-Extraction.pdf)
1. Shantanu Kumar. 2017. **A survey of deep learning methods for relation extraction**. [[paper]](https://arxiv.org/pdf/1705.03645.pdf)
1. Sachin Pawar, Girish K. Palshikara, Pushpak Bhattacharyyab. 2017. **Relation extraction: a survey**. [[paper]](https://arxiv.org/pdf/1712.05191.pdf)

## Datasets

You can download most of the following datasets in `json` format from [OpenNRE](https://github.com/thunlp/OpenNRE).

### Sentence-Level Relation Extraction

1. **ACE 2005 Dataset**. [[link]](https://catalog.ldc.upenn.edu/LDC2006T06) [[paper]](https://www.semanticscholar.org/paper/The-ACE-2005-(-ACE-05-)-Evaluation-Plan-Evaluation-Ntroduction/3a9b136ca1ab91592df36f148ef16095f74d009e)
1. **SemEval-2010 Task 8 Dataset**. [[link]](http://semeval2.fbk.eu/semeval2.php?location=tasks#T11) [[paper]](https://www.aclweb.org/anthology/W09-2415)
1. **TACREDD**. [[link]](https://nlp.stanford.edu/projects/tacred/) [[paper]](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf)

### Distantly Supervised Relation Extraction Datasets

1. **NYT Dataset**. [[link]](http://iesl.cs.umass.edu/riedel/ecml/) [[paper]](https://dl.acm.org/citation.cfm?id=1889799)

### Few-shot Relation Extraction Datasets

1. **FewRel**. [[link]](https://github.com/thunlp/fewrel) [[1.0 paper]](https://www.aclweb.org/anthology/D18-1514/) [[2.0 paper]](https://doi.org/10.18653/v1/D19-1649)

### Document-Level Relation Extraction Datasets

1. **DocRED**. [[link]](https://github.com/thunlp/DocRED) [[paper]](https://www.aclweb.org/anthology/P19-1074/) 

## Papers

### Pattern-Based Methods

1. Stephen Soderland, David Fisher, Jonathan Aseltine, and Wendy Lehnert. 1995. **Crystal inducing a conceptual dictionary**. In Proceedings of IJCAI. [[paper]](https://www.ijcai.org/Proceedings/95-2/Papers/038.pdf)
1. Jun-Tae Kim and Dan I. Moldovan. 1995. **Acquisition of linguistic patterns for knowledge-based information extraction**. TKDE. [[paper]](https://ieeexplore.ieee.org/abstract/document/469825/)
1. Scott B Huffman. 1995. **Learning information extraction patterns from examples**. In Proceedings of IJCAI. [[paper]](https://doi.org/10.1007/3-540-60925-3_51v)
1. Mary Elaine Califf and Raymond J. Mooney. 1997. **Relational learning of pattern-match rules for information extraction**. In Proceedings of CoNLL. [[paper]](https://www.aclweb.org/anthology/W97-1002)
1. Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam R Hruschka, and Tom M Mitchell. 2010. **Toward an architecture for never-ending language learning**. In Proceedings of AAAI. [[paper]](https://doi.org/10.1007/3-540-60925-3_51)
1. Ndapandula Nakashole, Gerhard Weikum, and Fabian Suchanek. 2012. **PATTY: A taxonomy of relational patterns with semantic types**. In Proceedings of EMNLP-CoNLL. [[paper]](https://www.aclweb.org/anthology/D12-1104)
1. Shun Zheng, Xu Han, Yankai Lin, Peilin Yu, Lu Chen, Ling Huang, Zhiyuan Liu, and Wei Xu. 2019. **DIAG-NRE: A neural pattern diagnosis framework for distantly supervised neural relation extraction**. In Proceedings of ACL. [[paper]](https://doi.org/10.18653/v1/P19-1137)

### Statistical Methods

#### Feature-Based

1. Nanda Kambhatla. 2004. **Combining lexical, syntactic, and semantic features with maximum entropy models for extracting relations**. [[paper]](https://www.aclweb.org/anthology/P04-3022.pdf)
1. Guodong Zhou, Jian Su, Jie Zhang, and Min Zhang. 2005. **Exploring various knowledge in relation extraction**. In Proceedings of ACL, pages 427–434. [[paper]](https://www.aclweb.org/anthology/P05-1053)
1. Jing Jiang and ChengXiang Zhai. 2007. **A systematic exploration of the feature space for relation extraction**. In Proceedings of NAACL, pages 113–120. [[paper]](https://www.aclweb.org/anthology/N07-1015.pdf)
1. Dat PT Nguyen, Yutaka Matsuo, and Mitsuru Ishizuka. 2007. **Relation extraction from wikipedia using subtree mining**. In Proceedings of AAAI, pages 1414–1420. [[paper]](https://www.aaai.org/Papers/AAAI/2007/AAAI07-224.pdf)

#### Kernel-Based

1. Aron Culotta and Jeffrey Sorensen. 2004. **Dependency tree kernels for relation extraction**. In Proceedings of ACL, page 423. [[paper]](https://www.aclweb.org/anthology/P04-1054.pdf)
1. Razvan C Bunescu and Raymond J Mooney. 2005. **A shortest path dependency kernel for relation extraction**. In Proceedings of EMNLP, pages 724–731. [[paper]](https://www.aclweb.org/anthology/H05-1091.pdf)
1. Shubin Zhao and Ralph Grishman. 2005. **Extracting relations with integrated information using kernel methods**. In Proceedings of ACL, pages 419–426. [[paper]](https://www.aclweb.org/anthology/P05-1052.pdf)
1. Raymond J Mooney and Razvan C Bunescu. 2006. **Subsequence kernels for relation extraction**. In Proceedings of NIPS, pages 171–178. [[paper]](https://papers.nips.cc/paper/2787-subsequence-kernels-for-relation-extraction.pdf)
1. Min Zhang, Jie Zhang, Jian Su, and Guodong Zhou. 2006. **A composite kernel to extract relations between entities with both flat and structured features**. In Proceedings of ACL, pages 825–832. [[paper]](https://www.aclweb.org/anthology/P06-1104)
1. Mengqiu Wang. 2008. **A re-examination of dependency path kernels for relation extraction**. In Proceedings of IJCNLP, pages 841–846. [[paper]](https://www.aclweb.org/anthology/I08-2119)

#### Graphical Models

1. Dan Roth and Wen-tau Yih. 2002. **Probabilistic reasoning for entity & relation recognition**. In Proceedings of COLING. [[paper]](https://www.aclweb.org/anthology/C02-1151)
1. Sunita Sarawagi and William W Cohen. 2005. **Semimarkov conditional random fields for information extraction**. In Proceedings of NIPS, pages 1185–1192. [[paper]](https://papers.nips.cc/paper/2648-semi-markov-conditional-random-fields-for-information-extraction)
1. Xiaofeng Yu and Wai Lam. 2010. **Jointly identifying entities and extracting relations in encyclopedia text via a graphical model approach**. In Proceedings of ACL, pages 1399–1407. [[paper]](https://www.aclweb.org/anthology/C10-2160)

#### Embedding Models

1. Jason Weston, Antoine Bordes, Oksana Yakhnenko, and Nicolas Usunier. 2013. **Connecting language and knowledge bases with embedding models for relation extraction**. In Proceedings of EMNLP, pages 1366–1371. [[paper]](https://www.aclweb.org/anthology/D13-1136)
1. Sebastian Riedel, Limin Yao, Andrew McCallum, and Benjamin M Marlin. 2013. **Relation extraction with matrix factorization and universal schemas**. In Proceedings of NAACL, pages 74–84. [[paper]](https://www.aclweb.org/anthology/N13-1008.pdf)
1. Matthew R Gormley, Mo Yu, and Mark Dredze. 2015. **Improved relation extraction with feature-rich compositional embedding models**. In Proceedings of EMNLP, pages 1774–1784. [[paper]](https://www.aclweb.org/anthology/D15-1205.pdf)
1. Antoine Bordes, Nicolas Usunier, Alberto Garcia- Duran, Jason Weston, and Oksana Yakhnenko. 2013. **Translating embeddings for modeling multirelational data**. In Proceedings of NIPS, pages 2787– 2795. [[paper]](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
1. Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. **Knowledge graph embedding by translating on hyperplanes**. In Proceedings of AAAI. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531)
1. Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. **Learning entity and relation embeddings for knowledge graph completion**. In Proceedings of AAAI. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523)

### Neural Methods

#### Recursive Neural Networks

1. Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. 2012. **Semantic compositionality through recursive matrix-vector spaces**. In Proceedings of EMNLP, pages 1201–1211. [[paper]](https://www.aclweb.org/anthology/D12-1110)
1. Makoto Miwa and Mohit Bansal. 2016. **End-to-end relation extraction using lstms on sequences and tree structures**. In Proceedings of ACL, pages 1105–1116. [[paper]](https://www.aclweb.org/anthology/P16-1105)

#### Convolutional Neural Networks

1. Chunyang Liu, Wenbo Sun, Wenhan Chao, and Wanxiang Che. 2013. **Convolution neural network for relation extraction**. In Proceedings of ICDM, pages 231–242. [[paper]](https://link.springer.com/content/pdf/10.1007%2F978-3-642-53917-6.pdf)
1. Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. **Relation classification via convolutional deep neural network**. In Proceedings of COLING, pages 2335–2344. [[paper]](https://www.aclweb.org/anthology/C14-1220) 
1. Cicero Nogueira dos Santos, Bing Xiang, and Bowen Zhou. 2015. **Classifying relations by ranking with convolutional neural networks**. In Proceedings of ACL-IJCNLP, pages 626–634. [[paper]](https://www.aclweb.org/anthology/P15-1061.pdf)
1. Thien Huu Nguyen and Ralph Grishman. 2015. **Relation extraction: Perspective from convolutional neural networks**. In Proceedings of the NAACL Workshop on Vector Space Modeling for NLP, pages 39–48. [[paper]](https://www.aclweb.org/anthology/W15-1506)

#### Recurrent Neural Networks

1. Dongxu Zhang and Dong Wang. 2015. **Relation classification via recurrent neural network**. arXiv preprint arXiv:1508.01006. [[paper]](https://arxiv.org/abs/1508.01006)
1. Thien Huu Nguyen and Ralph Grishman. 2015. **Combining neural networks and log-linear models to improve relation extraction**. arXiv preprint arXiv:1511.05926. [[paper]](https://arxiv.org/abs/1511.05926)
1. Ngoc Thang Vu, Heike Adel, Pankaj Gupta, et al. 2016. **Combining recurrent and convolutional neural networks for relation classification**. In Proceedings of NAACL, pages 534–539. [[paper]](https://www.aclweb.org/anthology/N16-1065)
1. Shu Zhang, Dequan Zheng, Xinchen Hu, and Ming Yang. 2015. **Bidirectional long short-term memory networks for relation classification**. In Proceedings of PACLIC, pages 73–78. [[paper]](https://www.aclweb.org/anthology/Y15-1009)

#### Graph Neural Networks

1. Yuhao Zhang, Peng Qi, and Christopher D. Manning. 2018. **Graph convolution over pruned dependency trees improves relation extraction**. In Proceedings of EMNLP, pages 2205–2215. [[paper]](http://aclweb.org/anthology/D18-1244)
1. Hao Zhu, Yankai Lin, Zhiyuan Liu, Jie Fu, Tat-Seng Chua, and Maosong Sun. 2019. **Graph neural networks with generated parameters for relation extraction**. In Proceedings of ACL, pages 1331–1339. [[paper]](https://doi.org/10.18653/v1/P19-1128)

#### Attention

1. Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. 2016. **Attention-based bidirectional long short-term memory networks for relation classification**. In Proceedings of ACL, pages 207–212. [[paper]](https://www.aclweb.org/anthology/P16-2034)
1. Linlin Wang, Zhu Cao, Gerard De Melo, and Zhiyuan Liu. 2016. **Relation classification via multi-level attention cnns**. In Proceedings of ACL, pages 1298–1307. [[paper]](https://www.aclweb.org/anthology/P16-1123)
1. Minguang Xiao and Cong Liu. 2016. **Semantic relation classification via hierarchical recurrent neural network with attention**. In Proceedings of COLING, pages 1254–1263. [[paper]](https://www.aclweb.org/anthology/C16-1119.pdf)

#### Word & Position Embedding

1. Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. **Word representations: a simple and general method for semi-supervised learning**. In Proceedings of ACL, pages 384–394. [[paper]](https://aclweb.org/anthology/P10-1040)
1. Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. **Distributed representations of words and phrases and their compositionality**. In Proceedings of NIPS, pages 3111–3119. [[paper]](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
1. Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. **Relation classification via convolutional deep neural network**. In Proceedings of COLING, pages 2335–2344. [[paper]](https://www.aclweb.org/anthology/C14-1220)

#### Shortest Dependency Path

1. Yang Liu, Furu Wei, Sujian Li, Heng Ji, Ming Zhou, and WANG Houfeng. 2015. **A dependency-based neural network for relation classification**. In Proceedings of ACL-IJCNLP, pages 285–290. [[paper]](https://www.aclweb.org/anthology/P15-2047.pdf)
1. Yan Xu, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng, and Zhi Jin. 2015. **Classifying relations via long short term memory networks along shortest dependency paths**. In Proceedings of EMNLP, pages 1785–1794. [[paper]](https://www.aclweb.org/anthology/D15-1206)

#### Universal Schema

1. Patrick Verga, David Belanger, Emma Strubell, Benjamin Roth, and Andrew McCallum. 2016. **Multilingual relation extraction using compositional universal schema**. In Proceedings of NAACL, pages 886– 896. [[paper]](https://www.aclweb.org/anthology/N16-1103)
1. Patrick Verga and Andrew McCallum. 2016. **Row-less universal schema**. In Proceedings of ACL, pages 63– 68. [[paper]](https://www.aclweb.org/anthology/W16-1312) 
1. Sebastian Riedel, Limin Yao, Andrew McCallum, and Benjamin M Marlin. 2013. **Relation extraction with matrix factorization and universal schemas**. In Proceedings of NAACL, pages 74–84. [[paper]](https://www.aclweb.org/anthology/N13-1008.pdf)

#### Transformer and BERT

1. Jinhua Du, Jingguang Han, Andy Way, and Dadong Wan. 2018. **Multi-level structured self-attentions for distantly supervised relation extraction**. In Proceedings of EMNLP, pages 2216–2225. [[paper]](https://www.aclweb.org/anthology/D18-1245.pdf)
1. Patrick Verga, Emma Strubell, and Andrew McCallum. 2018. **Simultaneously self-attending to all mentions for full-abstract biological relation extraction**. In Proceedings of NAACL-HLT, pages 872–884. [[paper]](https://www.aclweb.org/anthology/N18-1080)
1. Shanchan Wu and Yifan He. 2019. **Enriching pre-trained language model with entity information for relation classification**. arXiv preprint arXiv:1905.08284. [[paper]](https://arxiv.org/abs/1905.08284)
1. Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, and Tom Kwiatkowski. 2019. **Matching the blanks: Distributional similarity for relation learning**. In Proceedings of ACL, pages 2895–2905. [[paper]](https://doi.org/10.18653/v1/P19-1279)


1. **SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals.**
   _Iris Hendrickx , Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid O ́ Se ́aghdha, Sebastian Pado ́, Marco Pennacchiotti, Lorenza Romano, Stan Szpakowicz._
   Workshop on Semantic Evaluations, ACL 2009
   [paper](http://delivery.acm.org/10.1145/1630000/1621986/p94-hendrickx.pdf?ip=133.130.111.179&id=1621986&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1536636784_f0b60686e8866c0c08f63436f3ed81eb)

   > This leads us to introduce a new task, which will be part of SemEval-2010: multi-way classification of mutually exclusive semantic relations between pairs of common nominals.

### Distantly Supervised Datasets and Training Methods

1. **Learning to Extract Relations from the Web using Minimal Supervision.**
  _Razvan C. Bunescu, Department of Computer Sciences._
  ACL 2007.
  [paper](http://www.aclweb.org/anthology/P07-1073)

    > We present a new approach to relation extraction that requires only a handful of training examples. Given a few pairs of named entities known to exhibit or not exhibit a particular relation, bags of sentences containing the pairs are extracted from the web.

2. **Distant Supervision for Relation Extraction without Labeled Data.**
  _Mike Mintz, Steven Bills, Rion Snow, Dan Jurafsky._
  ACL-IJCNLP 2009.
  [paper](https://www.aclweb.org/anthology/P09-1113)

    > Our experiments use Freebase, a large semantic database of several thousand relations, to provide distant supervision.

3. **Modeling Relations and Their Mentions without Labeled Text.**
  _Sebastian Riedel, Limin Yao, Andrew McCallum._
  ECML 2010.
  [paper](https://link.springer.com/content/pdf/10.1007%2F978-3-642-15939-8_10.pdf)

    > We present a novel approach to distant supervision that can alleviate this problem based on the following two ideas: First, we use a factor graph to explicitly model the decision whether two entities are related, and the decision whether this relation is mentioned in a given sentence; second, we apply constraint-driven semi-supervision to train this model without any knowledge about which sentences express the relations in our training KB.

4. **Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations.**
  _Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, Daniel S. Weld._
  ACL-HLT 2011.
  [paper](http://www.aclweb.org/anthology/P11-1055)

    > This paper presents a novel approach for multi-instance learning with overlapping re- lations that combines a sentence-level extrac- tion model with a simple, corpus-level compo- nent for aggregating the individual facts.

### Embeddings

1. **Distributed Representations of Words and Phrases and their Compositionality.**		
   _Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean._
   NIPS 2013.
   [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

   > In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.

2. **GloVe: Global Vectors for Word Representation.**
   _Jeffrey Pennington, Richard Socher, Christopher D. Manning._
   EMNLP 2014.
   [paper](http://www.aclweb.org/anthology/D14-1162)

   > The result is a new global log-bilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods.	

### Neural Encoders

1. **Semantic Compositionality through Recursive Matrix-Vector Spaces.**
  _Richard Socher, Brody Huval, Christopher D. Manning, Andrew Y. Ng._
  EMNLP-CoNLL 2012.
  [paper](https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf)

    > We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length.

2. **Convolution Neural Network for Relation Extraction**
   _Chunyang Liu, Wenbo Sun, Wenhan Chao, Wanxiang Che._
   ADMA 2013
   [paper](https://link.springer.com/chapter/10.1007/978-3-642-53917-6_21)

   > In this paper, we propose a novel convolution network, incorporating lexical features, applied to Relation Extraction. 

3. **Relation Classification via Convolutional Deep Neural Network.**
  _Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, Jun Zhao._
  COLING 2014.
  [paper](http://www.aclweb.org/anthology/C14-1220)

    > We exploit a convolutional deep neural network (DNN) to extract lexical and sentence level features. Our method takes all of the word tokens as input without complicated pre-processing.

4. **Classifying Relations by Ranking with Convolutional Neural Networks.**
  _C´ıcero Nogueira dos Santos, Bing Xiang, Bowen Zhou._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-1061)

    > In this work we tackle the relation classification task using a convolutional neural network that performs classification by ranking (CR-CNN).

5. **Relation Extraction: Perspective from Convolutional Neural Networks.**
   _Thien Huu Nguyen, Ralph Grishman._
   NAACL-HLT 2015
   [paper](http://www.aclweb.org/anthology/W15-1506)	

   > Our model takes advantages of multiple window sizes for filters and pre-trained word embeddings as an initializer on a non-static architecture to improve the performance. We emphasize the relation extraction problem with an unbalanced corpus.		

6. **End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures.**
   _Makoto Miwa, Mohit Bansal._
   ACL 2016.
   [paper](https://www.aclweb.org/anthology/P16-1105)

   > Our recurrent neural network based model captures both word sequence and dependency tree substructure information by stacking bidirectional tree-structured LSTM-RNNs on bidirectional sequential LSTM-RNNs... We further encourage detection of entities during training and use of entity information in relation extraction via entity pre-training and scheduled sampling.

7. **A Walk-based Model on Entity Graphs for Relation Extraction.**
   _Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou._
   ACL 2018.
   [paper](http://aclweb.org/anthology/P18-2014#page=6&zoom=100,0,313)

   > We present a novel graph-based neural network model for relation extraction. Our model treats multiple pairs in a sentence simultaneously and considers interactions among them. All the entities in a sentence are placed as nodes in a fully-connected graph structure.

### Denoising Methods

1. **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks.**
  _Daojian Zeng, Kang Liu, Yubo Chen, Jun Zhao._
  EMNLP 2015.
  [paper](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)

    > We propose a novel model dubbed the Piecewise Convolutional Neural Networks (PCNNs) with multi-instance learning to address these two problems.

2. **Neural Relation Extraction with Selective Attention over Instances.**
  _Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun._
  ACL 2016.
  [paper](http://www.aclweb.org/anthology/P16-1200)

    > Distant supervision inevitably accompanies with the wrong labelling problem, and these noisy data will substantially hurt the performance of relation extraction. To alleviate this issue, we propose a sentence-level attention-based model for relation extraction.

3. **Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks.**
   _Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang._
   COLING 2016.
   [paper](http://www.aclweb.org/anthology/C16-1139)

   > In this paper, we propose a multi-instance multi-label convolutional neural network for distantly supervised RE. It first relaxes the expressed-at-least-once assumption, and employs cross-sentence max-pooling so as to enable information sharing across different sentences.		

4. **Adversarial Training for Relation Extraction.**
  _Yi Wu, David Bamman, Stuart Russell._
  EMNLP 2017.
  [paper](http://www.aclweb.org/anthology/D17-1187)

    > Adversarial training is a mean of regularizing classification algorithms by generating adversarial noise to the training data. We apply adversarial training in relation extraction within the multi-instance multi-label learning framework.

5. **A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction.**
  _Tianyu Liu, Kexiang Wang, Baobao Chang, Zhifang Sui._
  EMNLP 2017.
  [paper](http://www.aclweb.org/anthology/D17-1189)

    > We introduce an entity-pair level denoise method which exploits semantic information from correctly labeled entity pairs to correct wrong labels dynamically during training.

6. **DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction.**
  _Pengda Qin, Weiran Xu, William Yang Wang._
  [paper](https://arxiv.org/pdf/1805.09929.pdf)

    > We introduce an adversarial learning framework, which we named DSGAN, to learn a sentence-level true-positive generator. Inspired by Generative Adversarial Networks, we regard the positive samples generated by the generator as the negative samples to train the discriminator.

7. **Reinforcement Learning for Relation Classification from Noisy Data.**
  _Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu._
  AAAI 2018.
  [paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)

    > We propose a novel model for relation classification at the sentence level from noisy data. The model has two modules: an instance selector and a relation classifier. The instance selector chooses high-quality sentences with reinforcement learning and feeds the selected sentences into the relation classifier, and the relation classifier makes sentence-level prediction and provides rewards to the instance selector.

8. **Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning.**
  _Pengda Qin, Weiran Xu, William Yang Wang._
    2018.
  [paper](https://arxiv.org/pdf/1805.09927.pdf)

    > We explore a deep reinforcement learning strategy to generate the false-positive indicator, where we automatically recognize false positives for each relation type without any supervised information.

### Extensions

1. **Neural Knowledge Acquisition via Mutual Attention between Knowledge Graph and Text**
  _Xu Han, Zhiyuan Liu, Maosong Sun._
  AAAI 2018.
  [paper](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2018_jointnre.pdf)

    > We propose a general joint representation learning framework for knowledge acquisition (KA) on two tasks, knowledge graph completion (KGC) and relation extraction (RE) from text. We propose an effective mutual attention between KGs and text. The recip- rocal attention mechanism enables us to highlight important features and perform better KGC and RE.

1. **Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention**
   _Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun, Peng Li._
   EMNLP 2018.
   [paper](http://www.aclweb.org/anthology/D18-1247)

     > We aim to incorporate the hierarchical information of relations for distantly supervised relation extraction and propose a novel hierarchical attention scheme. The multiple layers of our hierarchical attention scheme provide coarse-to-fine granularity to better identify valid instances, which is especially effective for extracting those long-tail relations.

1. **Incorporating Relation Paths in Neural Relation Extraction**
   _Wenyuan Zeng, Yankai Lin, Zhiyuan Liu, Maosong Sun._
   EMNLP 2017.
   [paper](http://aclweb.org/anthology/D17-1186)

     > We build inference chains between two target entities via intermediate entities, and propose a path-based neural relation extraction model to encode the relational semantics from both direct sentences and inference chains. 

1. **RESIDE: Improving Distantly-Supervised Neural Relation Extractionusing Side Information**
   _Shikhar Vashishth, Rishabh Joshi, Sai Suman Prayaga, Chiranjib Bhattacharyya, Partha Talukdar._
   EMNLP 2018.
   [paper](https://aclweb.org/anthology/D18-1157)

     > In this paper, we propose RESIDE, a distantly-supervised neural relation extraction method which utilizes additional side  information from KBs for improved relation extraction. It uses entity type and relation alias information for imposing soft constraints while predicting relations.
