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

🔗 <strong>Code</strong>: <a href="https://github.com/med-air/TTADC" target="_blank">https://github.com/med-air/TTADC</a>

<br>

<h3>🎯 <strong>Contributions</strong></h3>

<p>
To our best knowledge, this is the first work to effectively tackle the label distribution shift in medical image classification. Our method learns representative classifiers through distribution calibration by extending balanced softmax loss [24,34] to simulate multiple one-dominating-class distributions. Unlike [34], our approach is more flexible and suitable for ordinal classification, as we use ordinal encoding instead of one-hot encoding and generate more diverse distributions. During inference, we dynamically combine these classifiers based on the test label distribution via a consistency regularization loss. This method significantly mitigates label shift, as validated on liver fibrosis staging and COVID-19 severity prediction tasks.
</p>

<br>

<h3>📂 <strong>Datasets</strong></h3>

<p>
For the liver fibrosis staging task, we use an in-house abdominal CT dataset from three centers (823 cases from our center, 99 from center A, and 50 from center B). The staging is based on biopsy pathology and includes five classes: F0–F4. Liver regions are segmented using existing tools, and the CT images have a slice thickness of 5 mm and resolution of 512×512.
</p>

<p>
For COVID-19 severity prediction, we use the public chest CT dataset iCTCF [17], with 969 training cases from HUST-Union Hospital and 370 test cases from HUST-Liyuan Hospital. Severity is divided into six levels (S0–S5). Preprocessing and lung segmentation follow [2].
</p>

<br>

<h3>🖼️ <strong>Method Overview</strong></h3>

<p align="center">
  <img src="images/TTPA/TTADC.png" alt="TTADC Overview" width="80%">
</p>

</details>
