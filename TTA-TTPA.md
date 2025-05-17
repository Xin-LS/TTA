# Test-Time Prior Adaptation 

## - `TTADC` [Ma et al., Proc. MICCAI 2022] **Test-time adaptation with calibration of medical image classification nets for label distribution shift** [[PDF]](https://arxiv.org/abs/2207.00769) [[G-Scholar]](https://scholar.google.com/scholar?cluster=7982883573733677737&hl=en) [[CODE]](https://github.com/med-air/TTADC)

<details open>
<summary><strong>📌 Abstract</strong></summary>
<br>
Class distribution plays an important role in learning deep classifiers. When the proportion of each class in the test set differs from the training set, the performance of classification nets usually degrades. Such a label distribution shift problem is common in medical diagnosis since the prevalence of disease varies over location and time.  In this paper, we propose the first method to tackle label shift for medical image classification, which effectively adapts the model learned from a single training label distribution to arbitrary unknown test label distribution.  Our approach innovates **distribution calibration** to learn multiple representative classifiers, which are capable of handling different one-dominating-class distributions. When given a test image, the diverse classifiers are dynamically aggregated via the **consistency-driven test-time adaptation**, to deal with the unknown test label distribution.  
We validate our method on two important medical image classification tasks including **liver fibrosis staging** and **COVID-19 severity prediction**. Our experiments clearly show the decreased model performance under label shift. With our method, model performance significantly improves on all the test datasets with different label shifts for both medical image diagnosis tasks.  
🔗 **Code**: [https://github.com/med-air/TTADC](https://github.com/med-air/TTADC)
<br>
<img src="images/TTPA/TTADC.png" alt="TTADC Overview">
</details>
