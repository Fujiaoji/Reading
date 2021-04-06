##### 2021-1-18 A Survey on Transfer Learning [Paper](https://doi.org/10.1109/TKDE.2009.191)

###### 前言

[Transductive 与 Inductive区别](https://www.zhihu.com/question/68275921/answer/529156908): 我们想要预测的样本，是不是我们在训练的时候已经见（用）过的.

###### 1. Introduction

**机器学习与TL**

machine learning methods' common assumption: the training and test data are drawn from the same feature space and the same distribution. When the distribution changes, most statistical models need to be rebuilt from scratch using newly collected training data.

与transfer learning相近的工作是 multi-task learning-uncover the common (latent) features that can benefit each individual task. 但是, transfer learning 与其他工作的不同之处见表格

![tranfer_ML_difference1](https://github.com/Fujiaoji/Reading/blob/4cdc8f289c34a91247ef78d1bee99bfe4668a2d9/figures/tranfer_ML_difference1.png)

<img src="https://github.com/Fujiaoji/Reading/blob/4cdc8f289c34a91247ef78d1bee99bfe4668a2d9/figures/tranfer_ML_difference2.png" alt="tranfer_ML_difference2" style="zoom: 67%;" />

###### 2. Definition

**Domain** $\mathcal{D}=\{\mathcal{X},P(X)\}$: 包括一个feature space $\mathcal{X}$ 和一个边缘概率分布 $P(X)$, where $X=\{x_1,...,x_n\}\in\mathcal{X}$.

**Task** $\mathcal{T}=\{\mathcal{Y},P(Y|X)\}$ : 包括一个label space $\mathcal{Y}$ 和一个objective predictive function $f(.)$.这是从train data中学到的

**Training Data**: 是一些$\{x_i,y_i\}$

**Source and Target Domain**: $\mathcal{D}_S$ and $\mathcal{D}_T$

**Related**:  source和target domain 有一些关系

**Negative Transfer**:  the source domain and target domain are not related to each other, brute-force transfer may worse the result

<u>==***Transfer Learning(TL)***==</u>: 给定 source domain, source learning task, target domain 和 target learning task, TL  想要用source domain中的知识来improve the learning of the target predictive function $f(.)$, where $D_S \neq D_T$ or $\mathcal{T}_S \neq \mathcal{T}_T$. 

###### 2.3 TL 分类

<img src="D:\typora\Typora\Reading\figures\TL_category_concrete.png" alt="TL_category_concrete" style="zoom: 67%;" />

1. 按照迁移情境分类 situations between source and target domains and tasks

   **Inductive TL**:  $\mathcal{T}_S \neq \mathcal{T}_T$, no matter domains. Target domain labels-available. Labeled data are required to induce the $f_T(.)$ for use in target domain. 根据label在source data 中能不能得到，又分为2种，具体见图 (从$\mathcal{D}_S$与$\mathcal{T}_S$中学到的知道来improve $f_T(.)$ ).

   **Transductive TL**:  $\mathcal{T}_S = \mathcal{T}_T$, $D_S \neq D_T$, target domain-无label, source domain-有 label.然后, 又分为两类:

   ​	1) $\mathcal{X}_S \neq \mathcal{X}_T$, source and target domain 的 feature space 不同

   ​    2) $\mathcal{X}_S = \mathcal{X}_T$,  $P(X_S) \neq P(X_T)$. 

   **Unsupervised TL**: 无 label

<img src="D:\typora\Typora\Reading\figures\category.png" alt="category" style="zoom:80%;" />

2. 按迁移方法分类-what to transfer

   **Instance-based TL**: assumes that certain parts of data in source domain can be reused for leaning in the target domain by <u>re-weighting</u>.

   **Feature-representation-transfer**: Idea-learn a “good” feature representation for the target domain  

   **Parameter-transfer** : assumes that the source tasks and the target tasks share some parameters  

   **Relational knowledge-transfer problem** : assumes that some relationship among the data
   in the source and target domains is similar  

   <img src="D:\typora\Typora\Reading\figures\TL_approaches_2.png" alt="TL_approaches_2" style="zoom:80%;" />

   <img src="D:\typora\Typora\Reading\figures\TL_approaches.png" alt="TL_approaches" style="zoom:80%;" />

###### 3. 热门研究领域-来自PPT

​	3.1 Domain Adaptation 域适配问题

​	3.2 Multi-source TL 多源TL

​	3.3 Deep TL 深度TL

​	3.4 Heterogeneous TL 异构TL

###### 3.1 Domain Adaptation 域适配问题

**Problem Definition (算是Transductive TL)** 

Source domain-有 label, target domain-无label,  $\mathcal{T}_S = \mathcal{T}_T$, $D_S \neq D_T$--( $\mathcal{X}_S = \mathcal{X}_T$,  $P(X_S) \neq P(X_T)$). 



##### 2021-1-26 Domain Adaptive Classification on Heterogeneous Information Networks [Paper](https://doi.org/10.24963/ijcai.2020/196) , [code.](https://github.com/PKUterran/MuSDAC)

###### Abstract

Why-在HIN上进行domain adaptation的困难: 1) semantics需要做 domain alignment 2) trade-off between domain similarity and distinguishability.

###### Introduction

1. 为了解决上面HIN带来的问题
2. ==DANE算是个Homogeneous 的representation learning, 可以参考下; Deep==
   ==collective classification in heterogeneous information networks==

###### Proposed Method 是个unsupervised 见下图

1. Multi-channel shared weight GCN

   将target G 与 source G 都按meta-path 分，都独立的用semi那个 GCN 来产生original channel embedding set. 其中，参数共享，使其映射到同一个嵌入空间中。

2. Multi-space alignment

   用一个1-dimensional convolution. 有些东西不懂，什么鬼

![MuSDAC](D:\typora\Typora\Reading\figures\MuSDAC.png)

3. Model learning ==我觉得，我要做的可能是unsupervised TL==

滴滴，往下不懂，略。。。

###### Experiment

1. Dataset: ACM, Aminer, DBLP
2. Baselines: GCN, GraphInception, HAN, DANE, 以及一些提出方法的变体

##### 2021-1-28 DANE: Domain Adaptive Network Embedding  [Paper](https://www.ijcai.org/Proceedings/2019/0606.pdf)

###### Abstract

Why: previous works focus on learning embeddings for a single network, cannot learn representations transferable on multiple networks-Domian Adaptation

Introduction

1. 用表示学习做domain Adaptation的好处:

   1) It alleviates the cost of training downstream machine learning models by enabling models to be reused on other networks.;

   2) It handles the scarcity of labeled data by transferring downstream models trained well on a labeled network to unlabeled networks. ;

   3) It makes no difference which network is the source network in downstream tasks, enabling bidirectional domain adaptation  

2. 表示学习做Domain Adaptation的challenges

   1) embedding space alignment: similar nodes should have similar representations, even if they are from different networks. **本文: ** project nodes of target and source into a shared embedding space

   2) Distribution shift of embedding vectors influences the performance of model on target networks, since most ML work on similar distribution. **本文:** constrain P source 和 P target close 使节点的表示有相似的distribution (没太看懂). 

3. 本文提出的DANE, ==Unsupervised==, ==Homo==, 用了GCN (source 和 target network share parameters 来解决embedding space shift) 和GAN (来解决distribution shift).

###### Proposed Model



![DANE_Model](D:\typora\Typora\Reading\figures\DANE_Model.png)

1. Shared weight GCN

   有个<u>structural equivalence hypothesis in complex network</u>: two nodes having similar local network neighborhoods can be considered structurally similar even if they are from two different networks. Hence, shared weight graph convolutional network architecture can project the nodes of Gsrc and Gtgt into a shared embedding space, where structurally similar nodes have similar representation vectors.
   
   下面的损失函数，瞅见了吗，可以换成我的想法的那个

2. Adversarial Learning Regularization

   滴滴滴，只是做个初步了解：），所以，停，开始下一个
