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

### Distant Supervision

1. Mike Mintz, Steven Bills, Rion Snow, and Dan Jurafsky. 2009. **Distant supervision for relation extraction without labeled data**. In Proceedings of ACLIJCNLP, pages 1003–1011. [[paper]](https://www.aclweb.org/anthology/P09-1113.pdf)
1. Truc-Vien T Nguyen and Alessandro Moschitti. 2011. **End-to-end relation extraction using distant supervision from external semantic repositories**. In Proceedings of ACL, pages 277–282. [[paper]](https://www.aclweb.org/anthology/P11-2048.pdf)
1. Bonan Min, Ralph Grishman, Li Wan, Chang Wang, and David Gondek. 2013. **Distant supervision for relation extraction with an incomplete knowledge base**. In Proceedings of NAACL, pages 777–782. [[paper]](https://www.aclweb.org/anthology/N13-1095.pdf)

#### Selecting Informative Instances

1. Sebastian Riedel, Limin Yao, and Andrew McCallum. 2010. **Modeling relations and their mentions without labeled text**. In Proceedings of ECML-PKDD, pages 148–163. [[paper]](https://dl.acm.org/citation.cfm?id=1889799)
1. Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, and Daniel S Weld. 2011. **Knowledgebased weak supervision for information extraction of overlapping relations**. In Proceedings of ACL, pages 541–550. [[paper]](https://www.aclweb.org/anthology/P11-1055.pdf)
1. Mihai Surdeanu, Julie Tibshirani, Ramesh Nallapati, and Christopher D Manning. 2012. **Multi-instance multi-label learning for relation extraction**. In Proceedings of EMNLP, pages 455–465. [[paper]](https://www.aclweb.org/anthology/D12-1042.pdf)
1. Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao. 2015. **Distant supervision for relation extraction via piecewise convolutional neural networks**. In Proceedings of EMNLP, pages 1753–1762. [[paper]](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)
1. Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 2016. **Neural relation extraction with selective attention over instances**. In Proceedings of ACL, pages 2124–2133. [[paper]](https://www.aclweb.org/anthology/P16-1200v2.pdf)
1. Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D Manning. 2017. **Positionaware attention and supervised data improve slot filling**. In Proceedings of EMNLP, pages 35–45. [[paper]](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf)
1. Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun, and Peng Li. 2018c. **Hierarchical relation extraction with coarse-to-fine grained attention**. In Proceedings of EMNLP, pages 2236–2245. [[paper]](https://www.aclweb.org/anthology/D18-1247)
1. Yang Li, Guodong Long, Tao Shen, Tianyi Zhou, Lina Yao, Huan Huo, and Jing Jiang. 2019. **Self attention enhanced selective gate with entity-aware embedding for distantly supervised relation extraction**. arXiv preprint arXiv:1911.11899. [[paper]](https://arxiv.org/pdf/1911.11899)
1. Linmei Hu, Luhao Zhang, Chuan Shi, Liqiang Nie, Weili Guan, and Cheng Yang. 2019. **Improving distantly-supervised relation extraction with joint label embedding**. In Proceedings of EMNLP-IJCNLP, pages 3812–3820. [[paper]](https://www.aclweb.org/anthology/D19-1395.pdf)

#### Incorporating Extra Context

1. Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao, et al. 2017. **Distant supervision for relation extraction with sentence-level attention and entity descriptions**. In AAAI, pages 3060–3066. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14491)
1. Xu Han, Zhiyuan Liu, and Maosong Sun. 2018b. **Neural knowledge acquisition via mutual attention between knowledge graph and text**. In Proceedings of AAAI. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16691/16013)
1. Ningyu Zhang, Shumin Deng, Zhanlin Sun, Guanying Wang, Xi Chen, Wei Zhang, and Huajun Chen. 2019. **Long-tail relation extraction via knowledge graph embeddings and graph convolution networks**. In Proceedings of NAACL-HLT, pages 3016–3025. [[paper]](https://www.aclweb.org/anthology/N19-1306)
1. Jianfeng Qu, Wen Hua, Dantong Ouyang, Xiaofang Zhou, and Ximing Li. 2019. **A fine-grained and noise-aware method for neural relation extraction**. In Proceedings of CIKM, pages 659–668. [[paper]](https://dl.acm.org/citation.cfm?id=3357384.3357997)
1. Patrick Verga, David Belanger, Emma Strubell, Benjamin Roth, and Andrew McCallum. 2016. **Multilingual relation extraction using compositional universal schema**. In Proceedings of NAACL, pages 886–896. [[paper]](https://www.aclweb.org/anthology/N16-1103)
1. Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2017. **Neural relation extraction with multi-lingual attention**. In Proceedings of ACL, pages 34–43. [[paper]](https://www.aclweb.org/anthology/P17-1004.pdf)
1. Xiaozhi Wang, Xu Han, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2018. **Adversarial multi-lingual neural relation extraction**. In Proceedings of COLING, pages 1156–1166. [[paper]](https://www.aclweb.org/anthology/C18-1099)

#### Sophisticated Mechanisms

1. Ngoc Thang Vu, Heike Adel, Pankaj Gupta, et al. 2016. **Combining recurrent and convolutional neural networks for relation classification**. In Proceedings of NAACL, pages 534–539. [[paper]](https://www.aclweb.org/anthology/N16-1065)
1. Iz Beltagy, Kyle Lo, and Waleed Ammar. 2019. **Combining distant and direct supervision for neural relation extraction**. In Proceedings of NAACL-HLT, pages 1858–1867. [[paper]](https://www.aclweb.org/anthology/N19-1184.pdf)
1. Tianyu Liu, Kexiang Wang, Baobao Chang, and Zhifang Sui. 2017. **A soft-label method for noisetolerant distantly supervised relation extraction**. In Proceedings of EMNLP, pages 1790–1795. [[paper]](https://www.aclweb.org/anthology/D17-1189)
1. Jun Feng, Minlie Huang, Li Zhao, Yang Yang, and Xiaoyan Zhu. 2018. **Reinforcement learning for relation classification from noisy data**. In Proceedings of AAAI, pages 5779–5786. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17151/16140)
1. Xiangrong Zeng, Shizhu He, Kang Liu, and Jun Zhao. 2018. **Large scaled relation extraction with reinforcement learning**. In Proceedings of AAAI, pages 5658–5665. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16257)
1. Yi Wu, David Bamman, and Stuart Russell. 2017. **Adversarial training for relation extraction**. In Proceeding of EMNLP, pages 1778–1783. [[paper]](https://www.aclweb.org/anthology/D17-1187)
1. Xu Han, Zhiyuan Liu, and Maosong Sun. 2018. **Denoising distant supervision for relation extraction via instance-level adversarial training**. arXiv preprint arXiv:1805.10959. [[paper]](https://arxiv.org/pdf/1805.10959.pdf)

### Few-Shot Learning

1. Xu Han, Hao Zhu, Pengfei Yu, ZiyunWang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018d. **Fewrel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation**. In Proceedings of EMNLP, pages 4803–4809. [[paper]](https://www.aclweb.org/anthology/D18-1514)
1. Tianyu Gao, Xu Han, Hao Zhu, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2019. **FewRel 2.0: Towards more challenging few-shot relation classification**. In Proceedings of EMNLP-IJCNLP, pages 6251–6256. [[paper]](https://doi.org/10.18653/v1/D19-1649)
1. Tianyu Gao, Xu Han, Zhiyuan Liu, Maosong Sun. 2019. **Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification**. In Proceedings of AAAI. [[paper]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4604)
1. Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, and Tom Kwiatkowski. 2019. **Matching the blanks: Distributional similarity for relation learning**. In Proceedings of ACL, pages 2895–2905. [[paper]](https://doi.org/10.18653/v1/P19-1279)
1. Zhi-Xiu Ye and Zhen-Hua Ling. 2019. **Multi-level matching and aggregation network for few-shot relation classification**. In Proceedings of ACL, pages 2872–2881. [[paper]](https://doi.org/10.18653/v1/P19-1277)

### Document-Level Relation Extraction

1. Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019. **DocRED: A large-scale document-level relation extraction dataset**. In Proceedings of ACL, pages 764–777. [[paper]](https://doi.org/10.18653/v1/P19-1074)
1. Michael Wick, Aron Culotta, et al. 2006. **Learning field compatibilities to extract database records from unstructured text**. In Proceedings of EMNLP. [[paper]](https://www.aclweb.org/anthology/W06-1671.pdf)
1. Matthew Gerber and Joyce Chai. 2010. **Beyond Nom-Bank: A study of implicit arguments for nominal predicates**. In Proceedings of ACL, pages 1583–1592. [[paper]](https://www.aclweb.org/anthology/P10-1160.pdf)
1. Kumutha Swampillai and Mark Stevenson. 2011. **Extracting relations within and across sentences**. In Proceedings of RANLP. [[paper]](https://www.aclweb.org/anthology/R11-1004)
1. Katsumasa Yoshikawa, Sebastian Riedel, et al. 2011. **Coreference based event-argument relation extraction on biomedical text**. J. Biomed. Semant. [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3239306/)
1. Chris Quirk and Hoifung Poon. 2017. **Distant supervision for relation extraction beyond the sentence boundary**. In Proceedings of EACL, pages 1171–1182. [[paper]](http://www.aclweb.org/anthology/E17-1110)
1. Wenyuan Zeng, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2017. **Incorporating relation paths in neural relation extraction**. In Proceedings of EMNLP, pages 1768–1777. [[paper]](https://www.aclweb.org/anthology/D17-1186)
1. Fenia Christopoulou, Makoto Miwa, and Sophia Ananiadou. 2018. **A walk-based model on entity graphs for relation extraction**. In Proceedings of ACL, pages 81–88. [[paper]](https://www.aclweb.org/anthology/P18-2014.pdf)
1. Nanyun Peng, Hoifung Poon, Chris Quirk, Kristina Toutanova, and Wen-tau Yih. 2017. **Cross-sentence n-ary relation extraction with graph LSTMs**. TACL, 5:101–115. [[paper]](https://transacl.org/ojs/index.php/tacl/article/view/1028)
1. Linfeng Song, Yue Zhang, et al. 2018. **N-ary relation extraction using graph-state lstm**. In Proceedings of EMNLP. [[paper]](https://www.aclweb.org/anthology/D18-1246)
1. Hao Zhu, Yankai Lin, Zhiyuan Liu, Jie Fu, Tat-Seng Chua, and Maosong Sun. 2019. **Graph neural networks with generated parameters for relation extraction**. In Proceedings of ACL, pages 1331–1339. [[paper]](https://doi.org/10.18653/v1/P19-1128)

