# Test-Time Prior Adaptation 

## - `TTADC` [Ma et al., Proc. MICCAI 2022] **Test-time adaptation with calibration of medical image classification nets for label distribution shift** [[PDF]](https://arxiv.org/abs/2207.00769) [[G-Scholar]](https://scholar.google.com/scholar?cluster=7982883573733677737&hl=en) [[CODE]](https://github.com/med-air/TTADC)
<details open>
<summary><strong>📌 Abstract, Contributions, Datasets & Visualization</strong></summary>
<br>
<h3>🧠 <strong>Abstract</strong></h3>
<p>
Class distribution plays an important role in learning deep classifiers. When the proportion of each class in the test set differs from the training set, the performance of classification nets usually degrades. Such a label distribution shift problem is common in medical diagnosis since the prevalence of disease varies over location and time.  
</p>
<p>
In this paper, we propose the first method to tackle label shift for medical image classification, which effectively adapts the model learned from a single training label distribution to arbitrary unknown test label distribution. Our approach innovates <strong>distribution calibration</strong> to learn multiple representative classifiers, which are capable of handling different one-dominating-class distributions. When given a test image, the diverse classifiers are dynamically aggregated via the <strong>consistency-driven test-time adaptation</strong>, to deal with the unknown test label distribution.  
</p>
<p>
We validate our method on two important medical image classification tasks including <strong>liver fibrosis staging</strong> and <strong>COVID-19 severity prediction</strong>. Our experiments clearly show the decreased model performance under label shift. With our method, model performance significantly improves on all the test datasets with different label shifts for both medical image diagnosis tasks.  
</p>
<p>🔗 <strong>Code</strong>: <a href="https://github.com/med-air/TTADC" target="_blank">https://github.com/med-air/TTADC</a></p>
<br>
<h3>🎯 <strong>Contributions</strong></h3>
<p>
In this paper, to our best knowledge, we present the first work to effectively tackle the label distribution shift in medical image classification. Our method learns representative classifiers with distribution calibration, by extending the concept of balanced softmax loss [24,34] to simulate multiple distributions that one class dominates other classes. Compared with [34], our method can be more flexible and be more targeted for ordinal classification, as our one-dominating-class distributions can represent more diverse label distributions and we use ordinal encoding instead of one-hot encoding to train the model. Then, at model deployment to new test data, we dynamically combine the representative classifiers by adapting their outputs to the label distribution of test data. The test-time adaptation is driven by a consistency regularization loss to adjust the weights of different classifier. We evaluate our method on two important medical applications of liver fibrosis staging and COVID-19 severity prediction. With our proposed method, the label shift can be largely mitigated with consistent performance improvement.
</p>
<br>
<h3>📂 <strong>Datasets</strong></h3>
<p>
For the liver fibrosis staging task, we use an in-house abdominal CT dataset collected from three centers with varying label distributions, including 823 cases from our center, 99 from external center A, and 50 from external center B. The ground truths are obtained from liver biopsy pathology results. The disease is categorized into five stages: F0 (no fibrosis), F1 (portal fibrosis without septa), F2 (with few septa), F3 (numerous septa without cirrhosis), and F4 (cirrhosis). Liver regions were segmented using an existing clinical tool and used as the classification region of interest. The CT slices have a thickness of 5 mm and an in-plane resolution of 512 × 512.
</p>
<p>
For the COVID-19 severity prediction task, we use the public chest CT dataset iCTCF [17], which contains 969 training cases from HUST-Union Hospital and 370 test cases from HUST-Liyuan Hospital. The severity of COVID-19 is divided into six levels: S0 (control), S1 (suspected), S2 (mild), S3 (regular), S4 (severe), and S5 (critical). The preprocessing and lung segmentation steps follow the same procedure as a recent study [2].
</p>
<br>
<h3>🖼️ <strong>Method Overview</strong></h3>
<p align="center">
  <img src="images/TTPA/TTADC.png" alt="TTADC Overview" width="80%">
</p>
</details>


