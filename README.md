



# Summary of Understanding Self-Training for Gradual Domain Adaptation

Machine learning systems must adapt to data distributions that evolve over time, in applications ranging from sensor networks and self-driving car perception modules to brain-machine interfaces. We consider gradual domain adaptation, where the goal is to adapt an initial classifier trained on a source domain given only unlabeled data that shifts gradually in distribution towards a target domain. 

We prove the first non-vacuous upper bound on the error of self-training with gradual shifts, under settings where directly adapting to the target domain can result in unbounded error. The theoretical analysis leads to algorithmic insights, highlighting that regularization and label sharpening are essential even when we have infinite data, and suggesting that self-training works particularly well for shifts with small Wasserstein-infinity distance. Leveraging the gradual shift structure leads to higher accuracies on a rotating MNIST dataset and a realistic Portraits dataset.

- [1. Overview](https://github.com/mingruizhang97/gradual_self_train-project#1-overview)
- [2. Theoretical Analysis](https://github.com/mingruizhang97/gradual_self_train-project#2-theoretical-analysis)
- [3. Experiments](https://github.com/mingruizhang97/gradual_self_train-project#3-experiments)
- [4. My little experiment project](https://github.com/mingruizhang97/gradual_self_train-project#4-my-little-experiment-project)
- [5. Discussion](https://github.com/mingruizhang97/gradual_self_train-project#5-discussion)
- [6. Conclusion](https://github.com/mingruizhang97/gradual_self_train-project#6-conclusion)
- [7. Reference](https://github.com/mingruizhang97/gradual_self_train-project#7-reference)

# 1. Overview

## What is Domain Adaptation?
**Domain adaptation** is the ability to apply an algorithm trained in one or more "source domains" to a different but related "target domain". It generally seeks to learn a model from a source labeled data that can be generalized to an unlabeled target domain by minimizing the difference between domain distributions. The key challenge for domain adaptation theory is when the source and target supports do not overlap, which are typical in the modern high-dimensional regime.

Domain adaptation is a special case of transfer learning. Transfer learning refers to a class of machine learning problems where either the tasks and/or domains may change between source and target while in domain adaptations only domains differ and tasks remain unchanged.[1] 
<p align="center">
<img title="a title" alt="Alt text" src="https://github.com/mingruizhang97/gradual_self_train-project/blob/main/domain_adaptation%20graph.jpg" width = '600' height = '400'>
</p>


## What is Self-training?
**Self-training**, which is also known as self-learning, self-labeling, or decision-directed learning, is probably the earliest idea about using unlabeled data in classification. This is a wrapper-algorithm that repeatedly uses a supervised learning method. It starts by training on the labeled data only. In each step, a part of the unlabeled points is labeled as "pseudolabel" according to the current decision function; then the supervised method is retrained using the previous predictions(pseudolabeled samples)which have been classified with confidence as additional labeled points.[2]
## Motivation on Self-training for Gradual Domain Adaptation
Traditional machine learning aims to learn a model on a set of training samples to find an objective function with minimum risk on unseen test data. However, it assumes that both training and test data are drawn from the same distribution and share similar joint probability distributions. This assumption can be easily violated in the real-world applications. For example, sensor measurements drift over time due to sensor aging, self-driving car vision modules have to deal with evolving road conditions, and neural signals received by brain-machine interfaces change within the span of a day. In these examples, the domain shift doesn't happen at one time and repeatedly gathering large sets of labeled examples to retrain the model can be quite time and cost consuming. So we would like to see if self-training for gradual domain adaptation can solve the problem.
## Related Work
Hoffman et al.[3] , Michael et al.[4], Markus et al., Bobu et al.[5] among others propose
approaches for gradual domain adaptation. This setting differs from online learning,
lifelong learning, and concept drift, since we only have unlabeled data from shifted distributions. However most of these work are application or domain-based. To the best of our knowledge, in the paper, the author say that we are the $\textbf{first}$ to develop a theory for gradual domain adaptation, and investigate when and why the gradual structure helps.

# 2. Theoretical Analysis
## Problem General Setup
**Gradually shifting distributions:**
- Task: Consider a binary classification task of predicting labels $y \in \lbrace -1,1 \rbrace$ from input features $x \in R^{d}$. 
- Distributions: There are joint distributions over the inputs and labels, $R^{D} \times \lbrace -1,1 \rbrace : P_0,P_1,...,P_T$, where $P_0$ is the source domain, $P_T$ is the target domain, and $P_1,...,P_{T-1}$ are intermediate domains.
- Shift is gradual: Define $\rho(P,Q)$ as a distance function between distributions $P$ and $Q$. Assume that for some $\epsilon > 0$, $\rho (P_t,P_{t+1}) < \epsilon$ for all $0 \leq t \leq T$.
- Samples: There are $n_0$ labeled examples $S_0 = \lbrace x_i^{(0)}, y_i^{(0)} \rbrace_{i=1}^{n_0}$  sampled independently from the source $P_0$ and $n$ unlabeled examples $S_t = \lbrace x_i^{(t)} \rbrace_{i=1}^{n}$ sampled independently from $P_t$ for each $1 \leq t \leq T$.

**Models and objectives:**
- Model definition: We have a model family $\Theta$, where a model $M_{\theta}: R^d ->R$ outputs a score representing its confidence that the label $y$ is $1$ for the given example.
- Model prediction: Assume $sign(M_{\theta}(x))$ is the predict function for an input $x$, where $sign(r) = 1$ if $r\geq 0$ and $sign(r) = -1$ if $r < 0$.
- Model evaluation: Use 0-1 loss as the evaluation metric which evaluates models on the fraction of times they make a wrong prediction.
$$Err(\theta,P) = \underset{X,Y \sim P}{\mathbb{E}} [sign(M_{\theta}(X)\neq Y)]$$
The goal for the problem is to find a classifier $\theta$ that gets high accuracy on the target domain $P_T$—— that is, low $Err(\theta, P_T)$.
- Loss function: Select a loss function $l: R\times \lbrace-1,1\rbrace -> R^+$ which takes a prediction and label, and outputs a non-negative loss value, and we begin by training a source model $\theta_0$ that minimizes the loss on labeled data in the source domain:
$$\theta_0=\underset{\theta^{'} \in \Theta}{argmin \frac{1}{n_0}} \sum_{(x_i,y_i)\in S_0}^n l(M_{\theta^{'}}(x_i),y_i)$$
- Self-training: Use unlabeled data to adapt a model. Given a model $\theta$ and unlabeled data $S$, $ST(\theta, S)$ denotes the output of self-training. Self-training pseudolabels each example in $S$ using $M_{\theta}$, and then selects a new model $\theta^{'}$ that minimizes the loss on this pseudolabeled dataset. Formally,
$$ST(\theta, S) = \underset{\theta^{'} \in \Theta}{argmin \frac{1}{|S|}} \sum_{(x_i)\in S}^n l(M_{\theta^{'}}(x_i),sign(M_{\theta}(x_i)))$$
The behavior of self-training when run on infinite unlabeled data from a probability distribution $P$ can be described as 
$$ST(\theta, S) = \underset{\theta^{'} \in \Theta}{argmin} \underset{X \sim P}{\mathbb{E}} [l(M_{\theta^{'}}(X),sign(M_{\theta}(X)))]$$

**Baseline methods:**
- **Non-adaptive baseline:** Directly use $\theta_0$ on the target domain. It will incur error $Err(\theta_0, P_T)$.
- **Direct adaptation to target baseline:** Take the source model $\theta_0$ and self-trains on the target data $S_T$, and is denoted by $ST(\theta_0, S_T)$.
- **Gradual self-training:** In gradual self-training, we self-train on the finite unlabeled examples from each domain successively. That is, for $i\geq 1$, we set:
$$\theta_i = ST(\theta_{i-1},S_i)$$
$ST(\theta_0,(S_1,...,S_Y)) = \theta_T$ is the output of gradual self-training, which we evaluate on the target distribution $P_T$.
## Assumption
- **Models:** Consider regularized linear models that have weights with bounded $l_2$ norm: $\Theta_R = (w,b):\lbrace w\in R^{d} , b\in R , \lVert w \rVert_2 \leq R \rbrace$ for some fixed $R > 0$. Given $(w,b) \in \Theta_R$, the model's output is $M_{w,b}(x) = w^T x+b$.
- **Margin loss function:**  A margin loss encourages a model to classify points correctly and confidently by keeping correctly classified points far from the decision boundary. Consider the hinge loss $h$ and ramp loss $r$:
$$h(m) = max(1-m,0)$$
$$r(m) = min(h(m),1)$$
In our experiment, we take ramp loss as the loss function: $l_r (\hat{y},y)=r(y\hat{y})$, where $\hat{y} \in R$ is a model's prediction, and $y\in \lbrace -1,1\rbrace$ is the true label. We denote the population ramp loss as:
$$L_r(\theta,P) = \underset{X,Y \sim P}{\mathbb{E}}[l_r(M_{\theta}(X),Y)]$$
Given a finite sample $S$, the empirical loss is:
$$L_r(\theta,S) = \frac{1}{|S|} \underset{x,y\in S}{\sum}[l_r(M_{\theta}(x),y)]$$
- **Distributional distance:** Consider the Wasserstein-infinity distance($W_{\infty}$) as the distributional distance. It computes the distance between distributions. Intuitively, $W_{\infty}$ moves points from distribution $P$ to $Q$ by distance at most $\epsilon$ to match the distributions. Formally, given probability measures $P,Q$ on $\chi$:
$$W_{\infty}(P,Q) = inf(\underset{x\in R^d}{sup} \lVert f(x) - x\rVert_2: f:R^d->R^d, f_{push forward}P=Q)$$
In our case, we require that the conditional distributions do not shift too much. Given joint probability measures $P,Q$ on the inputs and labels $R^d \times \lbrace -1,1\rbrace$, the distance is:
$$\rho(P,Q) = max(W_{\infty}(P_{X|Y=1},Q_{X|Y=1}), W_{\infty}(P_{X|Y=-1},Q_{X|Y=-1}))$$
- **$\alpha^{\*}$-separation assumption:** Assume every domain admits a classifier with low loss $\alpha^{\*}$, that is there exists $\alpha^{\*} \geq 0$ and for every domain $P_t$, there exists some $\theta_t\in\Theta_R$ with $L_r(\theta_t,P_t) \leq \alpha^{\*}$.
- **Bounded data assumption:** Data is not too large on average: $\underset{x\sim P}{\mathbb{E}} [\lVert X\rVert_2^2] \leq B^2$, where $B >0$.
- **No label shift assumption:** Assume that the fraction of $Y=1$ labels does not change: $P_t (Y)$ is the same for all $t$.
## Essential Findings
### **Direct adaptation baseline fails:**
Direct adaptation baseline may get high ramp loss on the target domain even if it gets $0$ ramp loss on the source domain. It is because the distribution shift from source $P_0$ to the target $P_T$ can be large though the distribution shift from $P_t$ to $P_{t+1}$ is small. Formally:

Even under that $\alpha^{\*}$-separation, no label shift, gradual shift, and bounded data assumptions, there exists distributions $P_0,P_1,P_2$ and a source model $\theta \in \Theta_R$ that get $0$ loss on the source $(L_r(\theta,P_0)=0)$, but high loss on the target: $L_r(\theta,P_2) = 1$. Self-training directly on the target does not help: $L_r(ST(\theta,P_2),P_2) = 1$. This holds true even if every domain is separable, so $\alpha^{\*} = 0$.

### **Gradual self-training improves error:** 
- From this paper, they find out that the empirical ramp loss of the current pseudolabeled dataset has a upper bound through gradual self-training process. It means that if we have a model $\theta$ that gets low loss and the distribution shifts slightly, self-training gives us a model $\theta_{'}$ that does not do too badly on the new distribution. Formally:
Given $P,Q$ with $\rho(P,Q) = \rho < \frac{1}{R}$ and marginals on $Y$ are the same so $P(Y) = Q(Y)$. Suppose $P,Q$ satisfy the bounded data assumption, and we have initial model $\theta$, and $n$ unlabeled samples $S$ from $Q$, and we set $\theta^{'}= ST(\theta,S)$. Then with probability at least $1-\delta$ over the sampling of $S$, letting $\alpha^{\*} = min_{\theta^{\*} \in \Theta_R} L_r(\theta^{\*},Q)$:
$$L_r(\theta^{'},Q) \leq \frac{2}{1-\rho R}L_r(\theta,P)+\alpha^{\*}+\frac{4BR+\sqrt{2\log2/\delta}}{\sqrt{n}}$$

- After $T$ time steps, the error of gradual self-training is $\lesssim \exp(cT)\alpha_0$ for some constant $c$, if the originial error is $\alpha_0$. It means that this gradual structure allows some control of the error unlike direct adaptation where the accuracy on the target domain can be $0%$ if $T \geq 2$. Formally:
Under the $\alpha^{\*}$-separation, no label shift, gradual shift, and bounded data assumptions, if the source model $\theta_0$ has low loss $\alpha_0 \geq \alpha^{\*}$ on $P_0$ and $\theta$ is the result of gradual self-training: $\theta = ST(\theta_0,(S_1,...,S_n))$, letting $\beta = \frac{2}{1-\rho R}$:
$$L_r(\theta,P_T) \leq \beta^{T+1}(\alpha_0+\frac{4BR+\sqrt{2\log2T/\delta}}{\sqrt{n}})$$ 

- Gradual self-training in this setting is tight even with infinite unlabeled examples. The error still has an exponential growth under this case. Formally:
Even under the $\alpha^{\*}$-separation, no label shift, gradual shift, and bounded data assumptions, given $0 \leq \alpha_0 \leq \frac{1}{4}$, for every $T$ there exists distributions $P_0,..., P_{2T}$, and $\theta_0 \in \Theta_R$ with $L_r(\theta_0,P_0) \leq \alpha^{\*}$, but if $\theta^{'} = ST(\theta_0,(P1,...,P_{2T}))$ then $L_r(\theta^{'},P_{2T}) \geq min(0.5, \frac{1}{2}2^T \alpha_0)$. Note that $L_r$ is always in $[0,1]$.
In the paper, they suggest that if we want to sub-expotential bounds, we can either make additional assumptions on the data distributions, or devise alternative algorithms to achieve better bounds.
### **Essential ingredients for gradual self-training:**
- **Regularization:**
If we self-train without regularization, the optimal model for the pseudolabeled dataset is the original model since the original model gives the pseudolabel to this unlabeled dataset. More specific to our setting, our bounds require regularized models because regularized models classify the data correctly with a margin, so even after a mild distribution shift we get most new examples correct. Note that in traditional supervised learning, regularization is usually required when we have few examples for better generalization to the population, whereas in our setting regularization is important for maintaining a margin even with infinite data.
- **Label sharpening:**
In our setting, we pseufolabel examples as $-1$ or $1$ ("hard" labels), based on the output of the classifier, which consider as label sharpening. Some previous work uses "soft" labels, where for each example they assign a probability of the label being  $-1$ or $1$, and train using a logistic loss. Self-traning then picks the model which optimizes the logistic loss. However, this form of self-training may never update the parameters the original model minimizes the logistic loss. This suggests that we "sharpen" the soft labels to encourage the model to update its parameters.
- **Ramp loss versus hinge loss:**
Although hinge loss is very popular and tends to work better in practice since it is easier to optimize and is convex for linear models, it is not suitable to be a loss function  in our case. In the paper, they analyze that we cannot control the error of gradual self-training with the hinge loss even if we had infinite examples, so the ramp loss is important to get loss upper bounded.
- **Self-training won't improve the performance without domain shift:**
If we have no distribution shift ($P_0=...=P_T$), the error can only grow linearly. Given a classifier with loss $\alpha_0$, if we do gradual self-training the loss is at most $\alpha_0 T$. Formally:
Given $\alpha_0 > 0$, distributions $P_0 =...=P_T$, and model $\theta_0 \in \Theta_R$ with $L_r(\theta_0,P_0) \leq \alpha_0$, $L_r(\theta^{'},P_T) \leq \alpha_0(T+1)$ where $\theta^{'} = ST(\theta_0,(P_1,...,P_T))$.
However, if we use non-adaptive baseline under this case, it only has error $\alpha_0$. Therefore, self-training will not outperform when there is no domain shift.



# 3. Experiments
## Datasets
- **Gaussian:** Synthetic dataset where the distribution $P_t(X|Y )$for each of two classes is a ddimensional Gaussian, where $d = 100$. The means and covariances of each class vary over time. The model gets $500$ labeled samples from the source domain, and $500$ unlabeled samples from each of $10$ intermediate domains. This dataset resembles our Gaussian setting but the covariance matrices are not isotropic, and the number of labeled and unlabeled samples is finite and on the order of the dimension $d$.
- **Rotating MNIST:** Rotating MNIST is a semi-synthetic dataset where we rotate each MNIST image by an angle between $0$ and $60$ degrees. We split the $50,000$ MNIST training set images into a source domain (images rotated between $0$ and $5$ degrees), intermediate domain (rotations between $5$ and $60$ degrees), and a target domain (rotations between $55$ degrees and $60$ degrees). Note that each image is seen at exactly one angle, so the training procedure cannot track a single image across different angles.
- **Portraits:** A real dataset comprising photos of high school seniors across years. The model’s goal is to classify gender. We split the data into a source domain (first $2000$ images), intermediate domain (next $14000$ images), and target domain (next $2000$ images).
## Results
Models are evaluated on **classification accuracy**.
### Results on different self-training methods
<p align="center">
<img title="a title" alt="Alt text" src="https://github.com/mingruizhang97/gradual_self_train-project/blob/main/Picture1.png" width = '500' height = '150'>
</p>

- **Source:** simply train a classifier on the labeled source examples. 
- **Target self-train:** repeatedly self-train on the unlabeled target examples ignoring the intermediate examples.
- **All self-train:** pool all the unlabeled examples from the intermediate and target domains, and repeatedly self-train on this pooled dataset to adapt the initial source classifier.
- **Gradual self-train:** sequentially use self-training on unlabeled data in each successive intermediate domain, and finally self-train on unlabeled data on the target domain, to adapt the initial source classifier.

### Results on different label and regularization methods
<p align="center">
<img title="a title" alt="Alt text" src="https://github.com/mingruizhang97/gradual_self_train-project/blob/main/Picture2.png" width = '500' height = '150'>
</p>

- **Soft Labels:** With regularization but with soft labels
- **Gradual ST:** Explicit regularization and hard labels
- **No Reg:** Without regularization but with hard labels 
### Results on different number of samples

<p align="center">
<img title="a title" alt="Alt text" src="https://github.com/mingruizhang97/gradual_self_train-project/blob/main/Picture3.png" width = '500' height = '150'>
</p>

- Dataset: a rotating MNIST dataset where we increase the sample sizes. The source domain $P_0$ consists of $N \in \lbrace2000, 5000, 20000 \rbrace$ images on MNIST. $P_t$ then consists of these same $N$ images, rotated by angle $3t$, for $0\leq t \leq 20$. 
The goal is to get high accuracy on $P_{20}$: these images rotated by $60$ degrees—the model doesn’t have to generalize to unseen images, but to seen images at different angles.
### Results on different distributional distance
They also take total-variant distance as distributional distance instead of Wasserstein-infinity distance. Gradual self-training with total-variant distance gets $33.5 \pm 1.5 \%$ accuracy on the target, while direct adaptation to the target gets $33.0 \pm 2.2\%$ over 5 runs.
# 4. My little experiment project
To find out the influence of distributional distance, Wasserstein-infinity distance in this case, I use the same rotating MNIST dataset but change the rotating degree on domains. 
|                |Source domain rotation |   Intermediate domain rotation                         |Target domain rotation
|----------------|-------------------------------|-----------------------------|----------|
|Original Setting| $0.0-5.0$       | $5.0-60.0$    |$55.0-60.0$
|New Setting1    | $0.0-10.0$      |$10.0-80.0$  | $80.0-90.0$
|New Setting2    |$0.0-5.0$| $5.0-55.0$    |$55.0-60.0$

New setting1 is to measure the performance of gradual self-training when distributional distance increases. New setting2 is to measure the whether the overlap degree between intermediate domain and target domain on original setting can improve the performance in some ways which we need to get rid of in the experiment.
- Result on original setting

|                |Accuracy on target domain(%)                                               |
|----------------------------------------------|-----------------------------|
|SOURCE             |    31.90      |
|TARGET ST          |33.47           |       |
|ALL ST             |36.94|
|GRADUAL ST         |88.59|
- Result on New setting1

|                |Accuracy on target domain(%)                                               |
|----------------------------------------------|-----------------------------|
|SOURCE             |    14.70      |
|TARGET ST          |14.18           |       |
|ALL ST             |15.46|
|GRADUAL ST         |60.31|

- Result on New setting2

|                |Accuracy on target domain(%)                                               |
|----------------------------------------------|-----------------------------|
|SOURCE             |   33.90      |
|TARGET ST          |34.23           |       |
|ALL ST             |39.70|
|GRADUAL ST         |87.51|

From the result above, we can find out that when we increase the Wasserstein-infinity distance between distributions, the gradual self-training can still outperform among other methods but has poorer performance compared to that on smaller distance. It is intuitive if we remember that the upper bound of error is related to and has a negative correlation with distributional distance constant $\rho$. Overlap data distribution on the intermediate domain and target domain can have small positive impact on the performance which may lead to less rigorous experiment result on the original paper.

The original code that I use comes from the github repository of this paper:  https://github.com/p-lambda/gradual_domain_adaptation

# 5. Discussion
From the above experiments, the model of gradual self-training indeed outperforms the other model methods among all three datasets. Regularization and "hard" labels can both improve the performance of gradual self-training, especially in MNIST dataset. When the dataset has more samples, regularization is still important and model with regularization can have higher accuracy even dataset has more data. It is different from what we have known for traditional supervised machine learning where regularization is used for generalization and its effect may gradually wave when the number of data increases. If they use total-variant distance as distributional distance, the gradual self-training has similar performance as direct adaptation.
# 6. Conclusion
In this paper, they propose a general theory structure of self-training for gradual domain adaptation. To analyze it in a deeper place, they make more assumptions including defining regularized linear models as base models, ramp loss as loss function, Wasserstein-infinity distance as the distributional distance, etc.  Based on assumptions, they find out that direct adaptation may have a high ramp loss on the target domain even if it gets $0$ ramp loss on the source domain. The ramp loss of gradual self-training is upper bounded. After $T$ steps, the loss has an exponential growth even with infinite unlabeled examples. The gradual self-training has three essential ingredients: regularization, label sharpening and ramp loss. 

Although gradual self-training has better performance compared with other baseline methods, it still has limitations, which need to explore more in the future work. First, in this paper, they only analyze the gradual self-training under binary classification task with linear model. It is still unknown whether gradual self-training can outperform on multi-label classification task and other machine learning model classes. Second, although the error is upper-bounded, it is exponential growth which is expected to be lower in the future. Third, it is unclear the influence on the choice of distributional distance function as small Wasserstein-infinity distance can bring good performance to gradual self-training while gradual self-train with small total-variation distance has no better than direct adaptation. 

To sum up, gradual self-training is a new theory and has a lot of directions to explore and research.
# 7. Reference
[1] arXiv:2010.03978:[https://doi.org/10.48550/arXiv.2010.03978](https://doi.org/10.48550/arXiv.2010.03978)

[2] O. Chapelle, A. Zien, and B. Scholkopf. Semi-Supervised Learning. MIT Press, 2006.

[3] J. Hoffman, T. Darrell, and K. Saenko. Continuous manifold based adaptation for evolving visual domains. In Computer Vision and Pattern Recognition (CVPR), 2014.

[4] G. Michael, E. Dennis, K. B. Mara, B. Peter, and M. Dorit. Gradual domain adaptation for segmenting whole slide images showing pathological variability. In Image and Signal Processing, 2018.

[5] A. Bobu, E. Tzeng, J. Hoffman, and T. Darrell. Adapting to continuously shifting domains. In International Conference on Learning Representations Workshop (ICLR), 2018.
