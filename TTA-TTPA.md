# Test-Time Prior Adaptation 

## - `TTADC` [Ma et al., Proc. MICCAI 2022] **Test-time adaptation with calibration of medical image classification nets for label distribution shift** [[PDF]](https://arxiv.org/abs/2207.00769) [[G-Scholar]](https://scholar.google.com/scholar?cluster=7982883573733677737&hl=en) [[CODE]](https://github.com/med-air/TTADC)

<details open>
<summary><strong>📌 Abstract</strong></summary>
<br>
Class distribution plays an important role in learning deep classifiers. When the proportion of each class in the test set differs from the training set, the performance of classification nets usually degrades. Such a label distribution shift problem is common in medical diagnosis since the prevalence of disease varies over location and time.  In this paper, we propose the first method to tackle label shift for medical image classification, which effectively adapts the model learned from a single training label distribution to arbitrary unknown test label distribution.  Our approach innovates **distribution calibration** to learn multiple representative classifiers, which are capable of handling different one-dominating-class distributions. When given a test image, the diverse classifiers are dynamically aggregated via the **consistency-driven test-time adaptation**, to deal with the unknown test label distribution.  
We validate our method on two important medical image classification tasks including **liver fibrosis staging** and **COVID-19 severity prediction**. Our experiments clearly show the decreased model performance under label shift. With our method, model performance significantly improves on all the test datasets with different label shifts for both medical image diagnosis tasks.  
🔗 **Code**: [https://github.com/med-air/TTADC]  
  
contribution  
In this paper, to our best knowledge, we present the first work to effectively tackle the label distribution shift in medical image classification. Our method learns representative classifiers with distribution calibration, by extending the concept of balanced softmax loss [24,34] to simulate multiple distributions that one class dominates other classes. Compared with [34], our method can be more flexible and be more targeted for ordinal classification, as our one-dominating-class distributions can represent more diverse label distributions and we use ordinal encoding instead of one-hot encoding to train the model. Then, at model deployment to new test data, we dynamically combine the representative classifiers by adapting their outputs to the label distribution of test data. The test-time adaptation is driven by a consistency regularization loss to adjust the weights of different classifier. We evaluate our method on two important medical applications of liver fibrosis staging and COVID-19 severity prediction. With our proposed method, the label shift can be largely mitigated with consistent performance improvement.

<img src="images/TTPA/TTADC.png" alt="TTADC Overview">
</details>
