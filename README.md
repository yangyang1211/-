# Multi-source domian adaptation paper-collections
Paper collections of multi-source domian adaptation after year 2019, with the criteria of CCF-A or some top-ranked CCF-B.
持续更新中   
*author: 郑海霆*

## Overview Table

id | venue | name                                                                                                                                                                                                                                                                                                                      | with code? | release time | Number of citations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
----|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1| AAAI | [Multi-Source Distilling Domain Adaptation](https://github.com/yangyang1211/-/blob/main/paper/6997-Article%20Text-10226-1-10-20200525.pdf)                              | no       | 2020                                                                                                                                                                                                                                                       | 113
2| NeurIPS | [Multi-source Domain Adaptation for Semantic Segmentation](https://github.com/yangyang1211/paper-collections/blob/main/paper/NeurIPS-2019-multi-source-domain-adaptation-for-semantic-segmentation-Paper.pdf)                              | yes       | 2019                                                                                                                                                                                                                                                      | 118
3| NeurIPS | [Your Classifier can Secretly Suffice Multi-Source Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/NeurIPS-2020-your-classifier-can-secretly-suffice-multi-source-domain-adaptation-Paper.pdf)                              | no       | 2020                                                                                                                                                                                                                                                       | 34
4| ICML | [Domain Aggregation Networks for Multi-Source Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/wen20b.pdf)                              | no       | 2020                                                                                                                                                                                                                                                       | 40
5| ICCV | [Moment Matching for Multi-Source Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf)                              | yes      | 2019                                                                                                                                                                                                                                                       | 730
6| CVPR | [Unsupervised Multi-source Domain Adaptation Without Access to Source Data](https://github.com/yangyang1211/paper-collections/blob/main/paper/cvpr%202021%2037c%20Ahmed_Unsupervised_Multi-Source_Domain_Adaptation_Without_Access_to_Source_Data_CVPR_2021_paper.pdf)                              | yes       | 2021                                                                                                                                                                                                                                                       | 37
7| CVPR | [Dynamic Transfer for Multi-Source Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/cvpr%202021%2025c%20Li_Dynamic_Transfer_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf)                              | yes       | 2021                                                                                                                                                                                                                                                       | 25
8| ECCV | [Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/eccv%202020%2050c%202007.08801.pdf)                              | no       | 2020                                                                                                                                                                                                                                                       | 50
9| AAAI | [Multi-Source Domain Adaptation for Visual Sentiment Classification](https://github.com/yangyang1211/paper-collections/blob/main/paper/aaai%202020%2033c%205651-Article%20Text-8876-1-10-20200513.pdf)                              | no       | 2020                                                                                                                                                                                                                                                       | 33
10| CVPR | [Domain-Specific Batch Normalization for Unsupervised Domain Adaptation](https://github.com/yangyang1211/paper-collections/blob/main/paper/cvpr%202019%20241c%20Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)                              | no       | 2019                                                                                                                                                                                                                                                       | 241
# Test-Time Training OOD Generalization

#### [A Causal View on Robustness of Neural Networks](https://neurips.cc/virtual/2020/protected/poster_02ed812220b0705fabb868ddbf17ea20.html)(NIPS 2020)

**Motivation**: Existed works do not address the vulnerability of DNNs to *unseen* manipulations. However, human recognition is less affected by drastic perturbations due to our ability to perform *causal reasoninig*.

**Contributions**: 

- Input perturbations are generated by their unseen causes that are *artificially manipulatable*.

- A causal inspired deep generative model with a test-time inference method.

**Main ideas**:

![1650795522777](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650795522777.png)

- valid perturbation = intervention on M which is conditionally independent of Y and Z given X

  (X - input data, Y - label, M - a set of var. which can be intervened artificially, Z - other causes)

- test-time fine tuning: fine tune the network depend only on M, and the posterior network since M is involved in the inference of Z.

- extend to generic multi-modal scenarios.

**Limitations:** more complicated causal models?



### Batch Normalization

#### [Improving robustness against common corruptions by covariate shift adaptation](https://neurips.cc/virtual/2020/protected/poster_85690f81aadc1749175c187784afc9ee.html)(NIPS 2020)

**Motivation:** encourage stronger interactions between the currently disjoint fields of domain adaptation and robustness towards common corruptions. In ImageNet-C, there're samples with same corruptions that are available for model adaptation.

**Contributions:** 

- two performance metrics measuring robustness after partial and full unsupervised adaptation to the corrupted images
- adaptation increases the robustness of many models on several datasets
- Wasserstein distance can be used to predict the performance degradation of a non-adapted model

**Main ideas:**

- adapting batch normalization statistics is proposed as a simple baseline.

- covariate shift: given $p_s(\textbf{X}, y)=p_s(\textbf{X})p_s(y|\textbf{X})$ and $p_t(\textbf{X}, y)=p_t(\textbf{X})p_t(y|\textbf{X})$, if $p_s(y|\textbf{X})=p_t(y|\textbf{X})$ and $p_s(\textbf{X}) \neq p_t(\textbf{X})$

- removal of covariate shift: estimate the BN statistics $\mu_t,\sigma_t^2$

   ![1650799945512](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650799945512.png)

  hyperparameter N controls the trade-off between source and estimated target statistics.

- lower and upper bound of the expected Wasserstein distance between source and target statistics:

  ![1650801474290](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650801474290.png)

**Limitations:** BN doesn't improve results on all datasets, which may due to the incorrect feature learning of models $\rightarrow$ do the conclusions still hold for other adaptation methods?

#### [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/pdf/2204.02610.pdf) (ICML 2022) *sample-level but not single sample*

**Motivation**: current Test-time model suffer from two problems: 1) backward computation consuming, 2) trade-off between OOD accuracy and IID accuracy (catastrophic forgetting)

**Contributions**: 

- propose an active sample identification scheme to filter out non-reliable and redundant test data
  from model adaptation
- extend the label-dependent Fisher regularizer to test samples with pseudo label generation, which prevents drastic changes in important model weights
- demonstrate that the proposed EATA improves the efficiency of test-time adaptation and also alleviates the long-neglected catastrophic forgetting issue

**Main ideas**:

- sample adaptive entropy minimization loss, excluding two types of samples out of optimization: i) samples with high entropy values, since the gradients provided by those
  samples are highly unreliable; and ii) samples are very similar
- anti-forgetting regularizer to enforce the important weights of the model do not change a lot during the adaptation

![1658457169321](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1658457169321.png)



#### [**NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation**](https://openreview.net/forum?id=E9HNxrCFZPV) (NIPS 2022) *single sample*

**Motivation:** Previous TTA schemes assume that the test samples are independent and identically distributed (i.i.d.), even though they are often temporally correlated (non-i.i.d.) in application scenarios, e.g., autonomous driving. Most existing TTA methods fail dramatically under such scenarios.

**Contributions:**

- NOTE is a batch-free inference algorithm, requiring a single instance for inference
- NOTE requires only a single forwarding pass and updates only the normalization statistics and affine parameters in IABN
- NOTE merely stores predicted labels of test data tu run PBRS

**Main idea:**

![Screen Shot 2023-02-09 at 16.16.21](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 16.16.21.png)

- Adapting BN layers by re-calibrating channel-wise statistics for normalization or adapting the affine parameters is more likely to remove instance-wise variations useful to predict labels and include a bias in p(y) rather than uniform
- Instance-Aware Batch Normalization (IABN): corrects the original statistics (mean and std) only in cases when IN statistics significantly differ from new BN statistics with a hyperparameter $\alpha$ to determine the confidence leverl of the BN statistics. 
- Prediction-Balanced Reservior Sampling (PBRS): mimics i.i.d. samples from temporally correlated streams with a small memory, combines time-uniform sampling (Reservior Sampling) and prediction-uniform sampling to simulate i.i.d. samples from non-i.i.d. streams.

**Limitations:** NOTE requires the backbone networks to equipped with BN layers. 



#### [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](https://openreview.net/forum?id=EQfeudmWLQ) (ICLR 2023) *small batch size even single*

**Motivation:** re-estimating normalization statistics using test data depends on impractical assumptions that a test batch should be large enough and be drawn from i.i.d. stream, and we observed that the previous methods with TBN (direct use of test batch statistics) show critical performance drop without the assumptions. CBN (running mean and variance obtained from source data) and TBN are in a trade-off relationship (Fig. 1), in the sense that one of each shows its strength when the other falls apart.

**Contributions:**

- a novel domain-shift aware test-time normalization (TTN) layer that combines source and test batch statistics using channel-wise interpolating weights considering the sensitivity to domain shift in order to flexibly adapt to new target domains while preserving the well-trained source knowledge
- the broad applicability of our proposed TTN, which does not alter training or test- time schemes, we show that adding TTN to existing TTA methods significantly improves the performance across a wide range of test batch sizes (from 200 to 1) and in three realistic evaluation scenarios; stationary, continuously changing, and mixed domain adaptation
- extensive experiments on image classification using CIFAR-10/100-C, and ImageNet-C and semantic segmen- tation task using CityScapes, BDD-100K, Mapil- lary GTAV, and SYNTHIA

**Problem settings:**

- various test batch sizes (especially small batch size)
- each test batch is drawn by a single target domain, one of the multiple target domains, or mixture of target domains.

**Methods:**

![Screen Shot 2023-02-19 at 20.16.32](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-19 at 20.16.32.png)

![Screen Shot 2023-02-19 at 20.21.01](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-19 at 20.21.01.png)

- TTN interpolates CBN and TBN with a learnable weight alpha, while using the same affine parameters. different mixing ratios for every layer and channel.

  ![Screen Shot 2023-02-19 at 20.24.26](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-19 at 20.24.26.png)

- post training to optimize alpha. all parameters except alpha are frozen and have access to the labeled source data.

  - obtain prior knowledge of alpha by identifying which layers and their channels are sensitive to domain shifts.

    - simulate domain shift by augmenting the clean image

    - difference between standardized features of clean image and paired augmented image represents domain discrepancy. if large, the parameter at (layer, channel) is sensitive to domain shift.

    - compare the gradients of affine parameters to measure the dissimilarity of features

      ![Screen Shot 2023-02-19 at 20.30.20](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-19 at 20.30.20.png)

  - optimize alpha with prior knowlede and an additional objective term using augmented data.

    - replace CBN with TTN with alpha initialized as prior knowledge
    - CE loss o make consistent predictions, MSE=||alpha - prior alpha|| as a constraint to prevent alpha from moving too far.



#### [MECTA: Memory-Economic Continual Test-Time Model Adaptation ](https://openreview.net/forum?id=N92hjSf5NNh) (ICLR 2023)



#### [DELTA: DEGRADATION-FREE FULLY TEST-TIME ADAPTATION ](https://openreview.net/forum?id=eGm22rqG93) (ICLR 2023)



#### [Improving test-time adaptation via shift-agnostic weight regularization and nearest source prototypes](https://arxiv.org/abs/2207.11707) (ECCV 2022)



### Self-training  

#### [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://proceedings.mlr.press/v119/sun20b.html)(ICML 2020) *single sample with augmentation*

**Motivation:** the unlabeled test sample presented at test time gives us a hint about the distribution from which it was drawn. to take the advantage of this hint, we allow the model parameters to depend on the test sample instead of its unknown label.

**Contributions:** 

- propose test-time training (TTT) method.
- TTT shows good performance on corrupted images, video frames and unknown distribution shifts.
- TTT-online v.s. unsupervised DA: TTT-online has the flexibility to forget the training distribution representation, which is shown to be not harmful and perhaps should even be taken advantage of.

**Main ideas:**

- TTT considers two tasks - a main task and an auxiliary SSL task.
- self-supervised task - rotation prediction task
- model is a Y-structure with a shared feature extractor and two branches - main task branch and self-supervised task branch
- training is done in multi-task learning fashion - losses for both tasks are added together
- test-time shared feature extractor fine-tuning - minimizing the auxiliary task loss on test sample

##### [Inference Stage Optimization for Cross-scenario 3D Human Pose Estimation](https://neurips.cc/virtual/2020/protected/poster_1943102704f8f8f3302c2b730728e023.html)(NIPS 2020)

has very similar idea as TTT(ICML 2020) with 3D human pose estimation specific technologies.

##### [Learning to Generalize One Sample at a Time with Self-Supervision](https://arxiv.org/abs/1910.03915)(ICCV 2019 submission)

given source domain class annotations, use an auxiliary self-supervised network to improve model generalization across multiple sources by solving a jigsaw puzzle (shuffling image patches)



#### [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive? ](https://openreview.net/forum?id=86NHK__yFDl)(NIPS 2021) *single sample with augmentation*

**Motivation:** the failure of TTT with self-supervised learning is largely attributed to the unconstrained update from SSL task that interfere with the main task.

**Contributions:**

- an in-depth look at TTT with emphasis on its limitations
- test-time feature alignment, batch-queue decoupling, and contrastive test-time training
- theoretical analysis of test accuracy after adaptation
- improved version of test-time training - TTT++

**Main ideas:**

![1650808396289](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650808396289.png)

- without constraints, updated encoder may severely overfit to SSL task, and deteriorate the accuracy on main task.

- add a loss term: impose a constraint over the feature space to make the feature distribution of test examples remain close to that in training domain - classical divergence measures (squared distance of mean and covariance matrix).
- maintain a dynamic queue of encoded features for limited computational resources.
- adopt contrastive learning in TTT: contrastive learning in SSL.

**Limitations:** 

- more complicated summarizations
- theoretical assumptions v.s. empirical settings



#### [**OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training**](https://openreview.net/forum?id=YPoRoad6gzY) (NIPS 2022) *single sample with generated pseudo-sample*

**Motivation:** previous methods try to explore common features among the training forgeries for better classification. But the test data often exhibit different characteristics, and the learned common features may not be shared by them. 

**Contributions:**

- A new learning paradigm specially-designed for the generalizable deepfake detection task 
- Only one-step gradient descent for better computational efficiency

**Main ideas:**

![Screen Shot 2023-02-10 at 09.27.50](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-10 at 09.27.50.png)

- online test-time training: first generating a pseudo-training sample then use it to form a mini-training set and update model parameters via one-shot training.
  - for every test sample, randomly select a template image from training dataset and align them in geometry based on landmarks, then extract a convex hull and deformed final mask, finally use alpha or Poisson blending to obtain pseudo training sample.
- offline meta-training



#### [Contrastive test-time adaptation](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf) (CVPR 2022)



### Entropy optimization

#### [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://proceedings.mlr.press/v119/liang20a.html)(ICML 2020)

**Motivation:** existing DA methods need to access the source data during learning to adapt, which is not efficient for data transmission and may violate the data privacy policy.

**Contributions:** 

- Source HypOthesis Transfer (SHOT)
- a novel self-supervised pseudo-labeling method to prevent wrong alignment
- SHOT achieves SOTA in vanilla closed-set DA and a variety of other unsupervised DA like partial-set and open-set cases.

**Main ideas:**

- address the unsupervised DA task with only a pre-trained source model and without access to source data. in particular a K-way classification.

![1650958784220](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650958784220.png)

- use the same classifier module - the source hypothesis encodes the distribution information of unseen source data.
- adopt information maximization (IM) loss (entropy + diversity) - the idea target outputs should be similar to one-hot encoding but differ from each other.
- apply pseudo-labeling for each unlabeled data to better supervise the target data encoding training and prevent wrong labeling: a procedure similar to weighted k-means clustering

![1650959614110](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650959614110.png)

**Limitations:**

- only classification is considered



#### [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c)(ICLR 2021)

**Motivation:** entropy has connections with error and shift - lower entropy have lower error rates; more corruption causes more loss and entropy, entropy can estimate the degree of shift without training data or labels.

**Contributions:** 

- fully test-time adaptation - only target data and no source data
- tent: a test-time entropy minimization schema
- tent is not only robust to corruptions, but also capable of online and source-free adaptation for digit classification and semantic segmentation.

**Main ideas:**

![1650806015096](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650806015096.png)

- the model must be probabilistic and differentiable.
- measure Shannon entropy, and jointly optimize batched predictions over parameters shared across the batch.
- optimizer only updates normalization layers channel-wisely, and the remaining parameters are fixed. normalization and transformation for shifts and scales.

![1650806560947](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650806560947.png)

**Limitations:**

- does not improve real-world  natural but unknown shifts.
- much of the model stays fixed.
- limited in scope.
- hard to compute in regression case.



#### [**Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models**](https://openreview.net/forum?id=e8PVEkSa4Fq) (NIPS 2022)



#### [Towards Stable Test-time Adaptation in Dynamic Wild World ](https://openreview.net/forum?id=g2YraF75Tj) (ICLR 2023 oral) *small batch size even single sample*

**Motivation:** 

- TTA may fail to improve or even harm the model performance when test data have: 1) mixed distribution shifts, 2) small batch sizes, and 3) online imbalanced label distribution shifts, which are quite common in practice. 
- The batch norm layer is a crucial factor hindering TTA stability
- TTA with group and layer norms does not always succeed and still suffers many failure cases
- Certain noisy test samples with large gradients may disturb the model adaption and result in collapsed trivial solutions, *i.e.*, assigning the same class label for all samples

**Problem setting:** fully TTA

- it does not alter training and can adapt arbitrary pre-trained models to the test data without access to original training data
- it may rely on fewer backward passes (only one or less than one) for each test sample than TTT

**Contributions:**

- Empirically verify that batch-agnostic norm layers (*i.e.*, GN and LN) are more beneficial than BN to stable test-time adaptation under wild test settings, *i.e.*, mix domain shifts, small test batch sizes and online imbalanced label distribution shifts
- Address the model collapse issue of test-time entropy minimization on GN/LN models by proposing a sharpness-aware and reliable (SAR) optimization scheme, which jointly minimizes the entropy and the sharpness of entropy of those reliable test samples
- Experiments on ImageNet-C under online imbalanced label distribution shifts, mixed distribution shifts, and batch size = 1.

**Main ideas:**

- BN can hinder model performance: Failure of TTA will result in problematic mean and variance estimation:

  - Simply estimating shared BN statistics of multiple distributions from mini-batch test samples unavoidably obtains limited performance
  - it is hard to use very few samples (*i.e.*, small batch size) to estimate statistics accurately
  - the imbalanced label shift will also result in biased BN statistics towards some specific classes in the dataset

- online entropy minimization tends to predict all samples to the same class:

  - Along with the model starts to collapse the l2-norm of gradients of all trainable parameters suddenly increases and then degrades to almost 0.
  - This indicates that some test samples produce large gradients that may hurt the adaptation and lead to model collapse![Screen Shot 2023-02-17 at 11.35.52](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-17 at 11.35.52.png)

- Reliable entropy minimization: (remove samples in Area 1&2)

  - Directly filtering samples with gradients norm is infeasible since the gradients norms for different models and distribution shift types have different scales, and thus it is hard to devise a general method to set the threshold for sample filtering or gradient clipping
  - Remove samples with large gradients based on their entropy. The entropy depends on the model’s output class number C and it belongs to (0, ln C ) for different models and data. In this sense, the threshold for filtering samples with entropy is easier to select

- Sharpness-aware entropy minimization: 

  - Make the model insensitive to the large gradients contributed by samples in Area 4
  - Encourage the model to go to a flat area of the entropy loss surface. The reason is that a flat minimum has good generalization ability and is robust to noisy/large gradients, *i.e.*, the noisy/large updates over the flat minimum would not significantly affect the original model loss, while a sharp minimum would
  - follow SAM

- Model recovery scheme: record a moving average em of entropy loss values and reset Θ ̃ to be original once em is smaller than a small threshold e0, since models after occurring collapse will produce very small entropy loss.

- only optimize affine parameters in norm layers.

  ![Screen Shot 2023-02-17 at 11.42.43](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-17 at 11.42.43.png)



#### [Temporal Coherent Test Time Optimization for Robust Video Classification ](https://openreview.net/forum?id=-t4D61w4zvQ) (ICLR 2023)

**Motivation:** Image-based corruption robustness techniques cannot generalize to video-based corruption robustness task due to:

- corruptions in video can change temporal information
- video input data has a different format from image data
- image-based techniques ignore the huge information hidden in temporal dimension

**Contributions:**

- first attempt to study test-time optimization techiques for video classification corruption robustness
- a novel test-time optimization framework TeCo for video classification utilizing spatio-temporal information in training and test data
- outperforms baseline on Mini Kinetics-C and Mini SSV2-C

**Main ideas:**

![Screen Shot 2023-02-10 at 09.52.51](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-10 at 09.52.51.png)



### Supplementary augmentation 

#### [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://openreview.net/forum?id=e_yvNqkJKAW)(NIPS 2021)

**Motivation:** methods using objective function during test time and SGD optimization (e.g. _Tent_) have the problem of increased computational costs and harmed inference throughput. it also could face catastrophic failure due to limited test data.

**Contributions:** 

- optimization-free test-time templates adjuster (T3A) with procedure making the adjusted decision boundary avoid the high-data density region on the target domain and reduce the ambiguity (prediction entropy) of predictions.
- T3A can adapt different DA algorithms for training phase, and different backbone networks.
- T3A is computational light.

**Main ideas:**

- replace the output layer of the predictor (a linear classifier) trained on source-domain to a pseudo-prototypical classifier.

- each element in the weights matrix ($\omega$) of output layer works as the template of representation for each class, and prediction is done by measuring the distance (dot product) between the template and the representation of the input data.

- T3A adjust the templates in test-time 

  - augment the input data with prototype label obtained using original $\omega$ 

  - add the augmented data into support set

    ![1650892890729](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650892890729.png)

  - compute the adjusted templates as the centroids of samples in support set with corresponding class label

    ![1650892915699](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650892915699.png)

- the support set is not discarded when new test data arrives.

- use prediction entropy to filter unreliable pseudo-labeled data.

**Limitations:**

- only linear classifier is considered
- doesn't fully utilize the test data since the wrongly pseudo-labeled data is simply filtered out



#### [Ensembling with Deep Generative Views](https://arxiv.org/abs/2104.14551)(CVPR 2021)

**Motivation:** GANs can be viewed as a type of "interpolating mechanism" that can blend and recombine images in a continuous manner $\rightarrow$ use GAN outputs as test-time augmentation for classification tasks.

**Contributions:**

- test-time input data ensembling with GAN samples helps in several classification tasks
- training on generated samples further improves classifier's performance.

**Main ideas:**

- leverage a generative model to synthesis useful variations of a given image

- perturbing latent codes: object center alignment $\rightarrow$ encoder training $\rightarrow$ Isotropic Gaussian + PCA directions + style-mixing

- ensemble the end classifier decision

  ![1651029928229](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651029928229.png)

**Limitations:**

- GAN reconstruction quality and efficiency.
- classifiers are sensitive to GAN reconstructions $\rightarrow$ appropriate weighting between real images and GAN outputs is needed.
- domain gap of real images and GAN outputs are relatively small.



#### [Adaptive Methods for Real-World Domain Generalization](https://arxiv.org/abs/2103.15796)(CVPR 2021)

**Motivation:** domain invariance methods may not guarantee good performance if the distribution of domains has high variance.

**Contributions:**

- adapt low-shot prototypical learning to construct domain embeddings from unlabeled samples of each domain, then use embeddings as supplementary signals to train adaptive classifiers.
- theoretical results based on framework of kernel mean embeddings, and derive generalization bounds on the average risk for adaptive classifiers.
- the first large-scale, real-world domain generalization benchmark, Geo-YFCC.

**Main ideas:**

- _adaptivity gap_: each domain can be seen as a sampling from a mother distribution, the universal classifier minimizes the universal ERM, leading to a distance from the optimal classifier for a specific domain.

- first step - computing domain embeddings: maps any domain to a finite vector using prototypical network to learn kernel mean embeddings (expressivity and consistency)

- second step - ERM using augmented inputs: learn a neural network directly over the input augmented with a suitable domain embedding (two networks, one for input feature learning, another concatenate feature and prototype to predict label)

  ![1651028168195](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651028168195.png)

**Limitations:**

- needs access to sufficient source domains
- computation heavier



#### [**MEMO: Test Time Robustness via Adaptation and Augmentation**](https://openreview.net/forum?id=XrGEkCOREX2) (NIPS 2022) *single sample with augmentation*

**Motivation:** devise methods that make no assumptions about the model training process and are broadly applicable at test time.

**Contributions:** 

- Marginal Entropy Minimization with One test point (MEMO): minimize the maginal entropy of the model's predictions across the augmented versions of the test point
- MEMO consistently improves the performance of ResNet and vision transformer models onCIFAR-10-C, CIFAR-10.1, ImageNet-C, ImageNet-R and ImageNet-A.
- MEMO encourages both invariane across augmentations and confident predictions.

**Main ideas:**

![Screen Shot 2023-02-09 at 16.50.53](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 16.50.53.png)

- MEMO: model prediction should be invariant across augmented versions of the test point (marginal output distribution), and be confident in its predictions (entropy minimization)

  ![Screen Shot 2023-02-09 at 16.56.51](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 16.56.51.png)

​		![Screen Shot 2023-02-09 at 16.57.36](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 16.57.36.png)		

- MEMO can be composed with prior methods for training robust models and adapting model statistics.

**Limitations:**

- MEMO is more computationally expensive than standard model inference
- MEMO tents to lead to degenerate solutions, e.g. a constant label with maximal confidence regardless of the input.



#### [Twofer: Tackling Continual Domain Shift with Simultaneous Domain Generalization and Adaptation](https://openreview.net/forum?id=L8iZdgeKmI6) (ICLR 2023) 

**Motivation:** improve model performance *before and during* the training stage of each previously-unseen target domain (i.e., in the *Unfamiliar Period*), while also main- tains good performance in the time periods *after* the training

**Contributions:**

- tackle an important problem in practical scenarios with continual domain shifts, i.e., to improve model performance *before and during* training on a new target domain, in what we call the *Unfamiliar Period*, and try to achieve good model performance *after* training, providing the model with capabilities of target domain adaptation and forgetting alleviation.
- a novel framework Twofer includes a training- free data augmentation module that generates more data for improving model’s generalization ability, a new pseudo labeling mechanism to provide more accurate labels for domain adaptation, and a prototype contrastive alignment algorithm that effectively aligns domains to simultaneously improve the generalization ability and achieve target domain adaptation and forgetting alleviation.

**Main ideas:**

![Screen Shot 2023-02-10 at 10.00.55](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-10 at 10.00.55.png)



#### [Continual test-time domain adaptation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf) (CVPR 2022)



### Variational learning

#### [Bayesian Adaptation for Covariate Shift](https://openreview.net/forum?id=15HPeY8MGQ)(NIPS 2021)

**Motivations:** 

- design a test-time adaptation procedure that can not only improve predictive accuracy under distribution shift, but also provide reliable uncertainty estimates. 
- what precisely unlabeled test data under covariate shift can actually tell us about the optimal dataset is sill unclear.
- adapting via entropy minimization without Bayesian marginalization can lead to overconfident uncertainty estimates.

**Contributions:**

- Bayesian Adaptation for Covariate Shift (BACS) which approximates Bayesian inference.
- BACS outperforms prior adaptive methods in terms of accuracy and calibration under distribution shift.

**Main ideas:**

- _assumption_: the inputs that will be shown to our classifier have an unambiguous and clear labeling.
- formalize the assumption in terms of a prior belief about the _conditional entropy_ of labels conditioned on the inputs, then extend the probabilistic model to covariate shift - _entropy minimization_
- avoid explicit generative modeling by using a plug-in empirical Bayes approach.
- learn approximate posterior in training time, then use this approximation in place of the training set in test time.
- ensembling approach

![1650965285751](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650965285751.png)

**Limitations:**

- requires effective techniques for estimating the parameter posterior from the training set.



#### [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU)(ICLR 2022) *single sample with meta-learning*

**Motivations:**

- domain-invariant learning or domain augmentation will always lead to an "adaptivity gap" when applied to target domains without  further adaptation.
- previous test-time adaptation (fine-tuning or extra network) usually need batches of target data to achieve good performance.

**Contributions:**

- _learning to generalize on single test samples_ by considering a single sample a domain by itself under a meta-learning framework.
- does not need any fine-tuning or extra network training, but only a feed-forward computation at test time.

**Main ideas:**

- a meta-learning framework with episodic training - divide the source domain(s) into several meta-source domains and a meta-target domain in each training iteration.

- probabilistic framework - conditional predictive log-likelihood to predict class label; _variational learning_ to approximate the meta-prior distribution

- incorporate the unlabeled sample into the variational distribution to approximate the true posterior distribution

  ![1650967317782](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1650967317782.png)

- divide model into a feature extractor and a classifier - feature extractor is shared across domains, classifier is trained to be adapted.

  ![1651021701078](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651021701078.png)

  ![1651021721366](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651021721366.png)

**Limitations:**

- homogeneous domain generalization where all domains share the same label space.
- fail in the cases that contains object in different categories or multiple objects from the same category.

#### [**Test-Time Training with Masked Autoencoders**](https://openreview.net/forum?id=SHMi1b7sjXk) (NIPS 2022) *single sample*

**Motivation:** 

- The self-supervised task in TTT cannot be too easy or too hard, otherwise the test input will not provide useful signal. 
- A fundamental property shared by natural images – spatial smoothness, i.e. the local redundancy of information in the xy space.
-  Spatial autoencoding – removing parts of the data, then predicting the removed content – forms the basis of some of the most successful self-supervised tasks.

**Contributions:**

- masked autoencoders (MAE) is well suited for test-time training.
- substantial improvement on four datasets for object recognition, ImageNet-C, ImageNet-A, ImageNet-R, Portrait dataset.

**Main ideas:**

- substitite MAE for the self-supervised part of TTT.
- feature extractor f, self-supervised head g and main task head h. f is encoder of MAE, g is decoder, both ViTs. h a linear projection.



#### [**Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation**](https://openreview.net/forum?id=wiHzQWwg3l) (NIPS 2022)



### Feature Alignment

#### [Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering](https://openreview.net/forum?id=W-_4hgRkwb) (NIPS 2022)

**Motivation:** There is a confusion over the experimental settings, thus leading to unfair comparisons. Among the multiple protocols, we adopt a realistic sequential test-time training (sTTT) protocol.

**Contributions:**

- categorization of TTT protocols by two key factors: one/multiple-pass adaptation, modifications of source domain training loss.
- adopt a realistic TTT setting, sTTT, and propose TTAC by matching the statistics of the target clusters to the source ones. The target statistics are updated through moving averaging with filtered pseudo labels.
- The proposed method is complementary to existing TTT method and is demonstrated on six TTT datasets, CIFAR-10/100-C, CIFAR10.1, VisDA-C, ImageNet-C, ModelNet40-C.

**Main ideas:**

![Screen Shot 2023-02-09 at 17.11.30](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 17.11.30.png)

- Anchored clustering: category-wise alignment that allocates the same number of clusters in both source and target domains, assigns each target cluster to one source cluster, then minimizes the KL-divergence between each pair of clusters.

- Clustering through pseudo-labelling: calculate the temporal exponential moving average posteriors, filters the posterior that deviates from historical value too much or has too small confidence, and update the component Gaussian only with leftover target samples.

- Global feature alignment: minimize the KL-divergence of marginal source and target distributions.

- Efficient iterative updating: iteratively update the running statistics of a fixed-length queue.

  ![Screen Shot 2023-02-09 at 17.24.57](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 17.24.57.png)



#### [Test-Time Adaptation via Self-Training with Nearest Neighbor Information](https://openreview.net/forum?id=EzLtB4M1SbM) (ICLR 2023)

**Motivation:** Self-training methods use predictions from the classifier as pseudo labels for the test data and fine-tune the classifier to make it fit to the pseudo labels. These methods have a limitation that the fine-tuned classifier can overfit to the inaccurate pseudo labels, resulting in confirmation bias.

**Contributions:**

- new test-time adaptation method, which is simple yet effective in mitigating the confirmation bias problem of self-training, by adding adaptation modules on top of the feature extractor.
- investigate the effectiveness of TAST on two standard benchmarks, domain generalization and image corruption.

**Main ideas:**

![Screen Shot 2023-02-09 at 17.35.07](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 17.35.07.png)

- add feature adaptation modules on top of the feature extractor: ensemble scheme to alleviate the performance degradation of trained classifier caused by the random initialization of the adaptation module.
- generates pseudo label distributions with the nearest neighbor information: 
  - first update the support set S and filter out the unconfident examples from the support set
  - compute the prototype representations in the embedding space of adaptation module with a support set
  - With the prototypes, compute the prototype-based predicted class distribution of the nearby support examples in the embedding space
  - With the nearest neighbor information, TAST generates a pseudo label distribution by aggregating prototype-based predicted class distribution of the nearby support examples
- fine-tune the adaptation module by minizing the cross-entropy loss beween the predicted class distribution of the test example and the nearest neighbor-based pseudo label distribution
- Iterate pseudo-labelling and fine-tuning processes for T steps per batch.



### Meta-Learning

#### [**Meta-DMoE: Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts**](https://openreview.net/forum?id=_ekGcr07Dsp) (NIPS 2022)



### Pseudo-labels

#### [**Test Time Adaptation via Conjugate Pseudo-labels**](https://openreview.net/forum?id=2yvUYc-YNUH) (NIPS 2022)



#### [Towards Understanding GD with Hard and Conjugate Pseudo-labels for Test-Time Adaptation](https://openreview.net/forum?id=FJXf1FXN8C) (ICLR 2023)





# Data-centric

#### [A Fourier-based Framework for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf) (CVPR 2021)

**Motivation:** a well-known property of Fourier Transformation - the phase component of Fourier spectrum preserves high-level semantics of the original signal, while the amplitude component contains low-level statistics.

**Contributions:**

- Fourier Augmented Co-Teacher (FACT) framework: Fourier-based data augmentation + co-teacher regularization
- in-depth analysis about the rationales behind the hypothesis and method

**Main ideas:**

![1651728264404](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651728264404.png)

- Fourier-based data augmentation: mix the amplitude information while keeping the phase information unchanged.

- co-teacher regularization: dual-formed consistency loss to reach consensuses between predictions derived from augmented and original inputs, which is implemented with a momentum-updated teacher model.
  $$
  \theta_{tea}=m\theta_{tea}+(1-m)\theta_{stu}
  $$

  $$
  L^{a2o}_{cot}=KL(\sigma(f_{stu}(\hat{x}_i^k)/T)||\sigma(f_{tea}(x_i^k)/T))
  $$

  $$
  L^{o2a}_{cot}=KL(\sigma(f_{stu}(x_i^k)/T)||\sigma(f_{tea}(\hat{x}_i^k)/T))
  $$




#### [Causally Invariant Predictor with Shift-Robustness](https://arxiv.org/pdf/2107.01876) (2021)

data regeneration by soft intervention on variation factors while estimating causally invariant predictor



#### [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv) (ICLR 2023)

**Motivation:** 

- A single sample would not be able to adjust the whole model due to the large number of model parameters and the limited information contained in the sample. This makes it challenging for model adaptation methods to handle large domain gaps. Instead, we propose to adapt each target sample to the source distributions, which does not require any fine-tuning or parameter updates of the source model.
- energy-based models flexibly model complex data distributions and allow for efficient sampling from the modeled distribution by Langevin dynamics

**Main ideas:**

- following the multi-source domain generalization setting

- a new discriminative energy-based model jointly modeling the feature distribution of input data and the conditional distribution on the source domains as the composition of a classifier and a neural-network-based energy function in the data space, which are trained simultaneously on the source domains. The trained model iteratively updates the representation of each target sample by gradient descent of energy minimization through Langevin dynamics, which eventually adapts the sample to the source data distribution.

  ![Screen Shot 2023-02-09 at 18.07.45](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 18.07.45.png)

  ![Screen Shot 2023-02-09 at 18.10.05](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 18.10.05.png)

- In order to maintain the category information of the target samples during adaptation and promote better classification performance, we further introduce a categorical latent variable in our energy-based model. Our model learns the latent variable to explicitly carry categorical information by variational inference in the classification model. We utilize the latent variable as conditional categorical attributes like in compositional generation to guide the sample adaptation to preserve the categorical information of the original sample.

  ![Screen Shot 2023-02-09 at 18.11.03](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-09 at 18.11.03.png)

- At inference time, we simply ensemble the predictions obtained by adapting the unseen target sample to each source domain as the final domain generalization result.

**Limitations:**

- iterative adaptation introduces an extra time cost



# Source-Free Domain Adaptation

### Pseudo-labelling

#### [Progressive Domain Adaptation from a Source Pre-trained Model](https://gaokeji.info/abs/2007.01524v3) (2020)

**Main ideas:**

- Leverage a pre-trained model from the source domain and progressively update the target model in a self-learning manner. 
- We observe that target samples with lower self-entropy measured by the pre-trained source model are more likely to be classified correctly. From this, we select the reliable samples with the self-entropy criterion and define these as class prototypes. 
- We then assign pseudo labels for every target sample based on the similarity score with class prototypes.
- To reduce the uncertainty from the pseudo labeling process, we propose set-to-set distance-based filtering which does not require any tunable hyperparameters. 
- Finally, we train the target model with the filtered pseudo labels with regularization from the pre-trained source model. 

#### [Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation](https://arxiv.org/abs/2106.15326)(IJCAI 2021)

**Main ideas:**

- mine the hidden knowledge in the source model and exploit it to generate source avatar prototypes (i.e., representative features for each source class) as well as target pseudo labels for domain alignment.
- Contrastive Prototype Generation and Adaptation (CPGA) method. Specifically, CPGA consists of two stages: (1) prototype generation: by exploring the classification boundary information of the source model, we train a prototype generator to generate avatar prototypes via contrastive learning. (2) prototype adaptation: based on the generated source prototypes and target pseudo labels, we develop a new robust contrastive prototype adaptation strategy to align each pseudo-labeled target data to the corresponding source prototypes. 

#### [Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data](https://proceedings.neurips.cc/paper/2021/hash/1dba5eed8838571e1c80af145184e515-Abstract.html) (NIPS 2021)

**Main ideas:**

- historical contrastive learning (HCL) technique that exploits historical source hypothesis to make up for the absence of source data in UMA
- historical contrastive instance discrimination (HCID) that learns from target samples by contrasting their embeddings which are generated by the currently adapted model and the historical models. With the historical models, HCID encourages UMA to learn instance-discriminative target representations while preserving the source hypothesis.

- historical contrastive category discrimination (HCCD) that pseudo-labels target samples to learn category-discriminative target representations. Specifically, HCCD re-weights pseudo labels according to their prediction consistency across the current and historical models.

#### [A Free Lunch for Unsupervised Domain Adaptive Object Detection without Source Data](https://www.aaai.org/AAAI21Papers/AAAI-4761.LiX.pdf) (AAAI 2021)

**Main ideas:**

- source data-free domain adaptive object detection (SFOD) framework via modeling it into a problem of learning with noisy labels.
- self-entropy descent (SED) is a metric proposed to search an appropriate confidence threshold for reliable pseudo label generation without using any handcrafted labels.
- After a thorough experimental analysis, false negatives are found to dominate in the generated noisy labels.



#### [When Source-Free Domain Adaptation Meets Learning with Noisy Labels](https://openreview.net/forum?id=u2Pd6x794I) (ICLR 2023)



### Adversarial manner

#### [Casting a BAIT for Offline and Online Source-free Domain Adaptation](https://arxiv.org/abs/2010.12427) (2020)

**Main ideas:**

- we introduce a second classifier, but with another classifier head fixed.
- When adapting to the target domain, the additional classifier initialized from source classifier is expected to find misclassified features. Next, when updating the feature extractor, those features will be pushed towards the right side of the source decision boundary, thus achieving source-free domain adaptation. 

#### [Model Adaptation: Unsupervised Domain Adaptation Without Source Data](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.html) (CVPR 2020)

**Main ideas:**

![1651741668722](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1651741668722.png)

- collaborative class conditional generative adversarial net: the prediction model is to be improved through generated target-style data, which provides more accurate guidance for the generator
- a weight constraint that encourages similarity to the source model.
- A clustering-based regularization is also introduced to produce more discriminative features in the target domain.



### Generative manner

#### [Domain Impression: A Source Data Free Domain Adaptation Method](https://openaccess.thecvf.com/content/WACV2021/html/Kurmi_Domain_Impression_A_Source_Data_Free_Domain_Adaptation_Method_WACV_2021_paper.html) (WACV 2021)

**Main ideas:**

- Our proposed approach is based on a generative framework, where the trained classifier is used for generating samples from the source classes.
- We learn the joint distribution of data by using the energy-based modeling of the trained classifier. 
- At the same time, a new classifier is also adapted for the target domain. 

#### [SoFA: Source-Data-Free Feature Alignment for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content/WACV2021/html/Yeh_SoFA_Source-Data-Free_Feature_Alignment_for_Unsupervised_Domain_Adaptation_WACV_2021_paper.html) (WACV 2021)

**Main ideas:**

- The source model is used to predict the labels for target data, and we model the generation process from predicted classes to input data to infer the latent features for alignment.
- a mixture of Gaussian distributions is induced from the predicted classes as the reference distribution. The encoded target features are then aligned to the reference distribution via variational inference to extract class semantics without accessing source data.



### Batch Normalization

#### [Source-free Domain Adaptation via Distributional Alignment by Matching Batch Normalization Statistics](https://arxiv.org/abs/2101.10842) (2021)

**Main ideas:**

- utilizing batch normalization statistics stored in the pretrained model to approximate the distribution of unobserved source data.
- we fix the classifier part of the model during adaptation and only fine-tune the remaining feature encoder part so that batch normalization statistics of the features extracted by the encoder match those stored in the fixed classifier.
- we also maximize the mutual information between the features and the classifier's outputs to further boost the classification performance.



### Feature Alignment

#### [Source-Free Domain Adaptation to Measurement Shift via Bottom-Up Feature Restoration](https://arxiv.org/pdf/2107.05446.pdf) (ICLR 2022)

**Main ideas:**

- a particularly pervasive type of domain shift called measurement shift which can
  be resolved by restoring the source features rather than extracting new ones
- Feature Restoration (FR) wherein we: (i) store a lightweight and flexible approximation of the feature distribution under the source data (*soft binning*); and (ii) adapt the feature-extractor such that the approximate feature distribution under the target data realigns with that saved on the source (*minimizing symmetric KL divergence*).
- a bottomup training scheme which boosts performance, which we call Bottom-Up Feature
  Restoration (BUFR).

#### [**Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation**](https://openreview.net/forum?id=ZlCpRiZN7n) (NIPS 2022)



#### [**Divide and Contrast: Source-free Domain Adaptation via Adaptive Contrastive Learning**](https://openreview.net/forum?id=NjImFaBEHl) (NIPS 2022)

**Motivations:** 

- Existing techniques mainly leverage self-supervised pseudo-labeling to achieve class-wise *global* alignment or rely on *local* structure extraction that encourages the feature consistency among neighborhoods.
- the “global” approach is sensitive to noisy labels while the “local” counterpart suffers from the source bias

**Contributions:**

- A novel divide-and-contrast paradigm for SFUDA that can fully exploit both the global and local structures of target data via data segmentation and customized learning strategies for data subsets
- a unified adaptive contrastive learning framework that achieves class-wise adaptation for source-like samples and local consistency for target-specific data
- a memory bank based MMD loss to reduce the batch bias in batch training and an improved MMD loss with non-negative exponential form
- new state-of-the-art performance on VisDA, Office-Home, and DomainNet

**Main ideas:**

![Screen Shot 2023-02-10 at 10.17.58](/Users/wangzige/Library/Application Support/typora-user-images/Screen Shot 2023-02-10 at 10.17.58.png)

- achieves preliminary class-wise adaptation by Lself , which fits the model to pseudo-labels
- leverages adaptive contrastive loss Lcon to jointly achieve robust class-wise adaptation for source-like samples and local consistency regularization for target-specific samples
  - select confident samples with prediction probability greater than a threshold as source-like samples.
- minimizes the discrepancy between the source-like set and target-specific outliers by L*EMMD*



### Variational Learning

#### [**Variational Model Perturbation for Source-Free Domain Adaptation**](https://openreview.net/forum?id=yTJze_xm-u6) (NIPS 2022)





# Alignment

#### [Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift](https://arxiv.org/pdf/2003.04475v2.pdf) (NIPS 2020)

- generalized label shift

![1655122776783](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655122776783.png)

- importance weights of the target and source label distributions

![1655122819384](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655122819384.png)

- estimation of importance weights

![1655123495395](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655123495395.png)

- distributional alignment

![1655123582695](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655123582695.png)

- importance-weighted domain adversarial network

![1655123974245](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655123974245.png)

- classifier loss

![1655124028833](C:\Users\83947\AppData\Roaming\Typora\typora-user-images\1655124028833.png)





#### [Confident Anchor-Induced Multi-Source Free Domain Adaptation](https://openreview.net/forum?id=EAdJEN8xKUl)



#### [**Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation**](https://arxiv.org/pdf/2110.04202.pdf)(NIPS 2021)



#### [Generalized Source-free Domain Adaptation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2108.01614.pdf)(ICCV 2021)



