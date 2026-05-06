# Fair Conformal Prediction for Individuals and Subgroups in Natural and Adversarial Settings: Theoretical Guarantees and Impossibility Results

[![Paper](https://img.shields.io/badge/Paper-COPA%202026-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**Alberto Carlevaro<sup>1</sup>**, **Luca Oneto<sup>2</sup>**, **Davide Anguita<sup>2</sup>**, and **Fabio Roli<sup>2,3</sup>**

1. Aitek S.p.A., Via della Crocetta 15, 16122 Genova, Italy
2. DIBRIS, University of Genoa, Via Opera Pia 11a/13, 16145 Genova, Italy
3. DIEE, University of Cagliari, Via Marengo, Cagliari, 09123, Italy

---

## Abstract

Conformal prediction provides a principled framework to construct prediction sets with distribution-free coverage guarantees under the assumption of exchangeability. However, coverage alone is often insufficient in high-stakes applications, where additional requirements such as fairness and robustness are crucial.

In this work, we study the problem of constructing conformal predictors that simultaneously satisfy:
(i) coverage guarantees,
(ii) group-level fairness, in terms of Equalized Coverage,
(iii) individual-level fairness, in terms of Counterfactual Fairness, and
(iv) robustness to adversarial perturbations.

We show that these objectives are, in general, incompatible, establishing novel impossibility results. We then propose new conformal prediction methods that achieve a principled trade-off between these properties in both natural and adversarial settings. In particular, we introduce fairness-oriented adversarial attacks that explicitly target violations of fairness criteria, and we design defense mechanisms that restore coverage and fairness guarantees under realistic threat models.

Empirical evaluations on real-world datasets demonstrate the effectiveness of the proposed framework and highlight the limitations of existing approaches.

---

## How to Use

The code is organized into multiple components, reflecting the main settings analyzed in the paper.

Each dataset folder contains Jupyter Notebooks to reproduce the experiments.

* **NaturalSetting.ipynb**:
  Runs experiments in the standard (non-adversarial) setting.
  It evaluates:

  * Equalized Coverage
  * Equalized Set Size
  * Efficiency of prediction sets

  This notebook reproduces the main results of the **natural setting**.

* **AdversarialSetting.ipynb**:
  Runs experiments under adversarial perturbations.
  It includes:

  * Fairness-targeted attacks
  * Evaluation of fairness degradation
  * Robust conformal prediction (defense)

  This notebook reproduces the **adversarial experiments** in the paper.

---

## Libraries Used

The implementation relies on standard Python libraries for machine learning and experimentation, including:

* NumPy
* PyTorch / Scikit-learn
* Matplotlib

Adversarial attacks and conformal prediction methods are implemented within this repository.

Please install all dependencies before running the notebooks:

```bash
pip install -r requirements.txt
```

---

## Future Steps

* Extend the framework to **regression tasks**
* Study **multi-valued or continuous sensitive attributes**
* Develop **adaptive methods without access to sensitive attributes at test time**
* Investigate stronger and more adaptive adversarial attacks

---

## Contact

For any questions regarding the work or the code in this repository, feel free to contact:
📧 **Alberto Carlevaro** — [alberto.carlevaro@aitek.it](mailto:alberto.carlevaro@aitek.it)

---

## Acknowledgements

This work is partially supported by:

* Project **SERICS** (PE00000014) under the NRRP MUR program funded by the EU - NGEU
* Project **FAIR** (PE00000013) under the NRRP MUR program funded by the EU - NGEU
* **REXASI-PRO** H-EU project (HORIZON-CL4-2021-HUMAN-01-01, Grant Agreement ID: 101070028)
* **Fit4MedRob - Fit for Medical Robotics** Grant (PNC0000007)
* **ELSA – European Lighthouse on Secure and Safe AI** (Grant Agreement No. 101070617)

---

## How to cite this work

Still under submission.

```
