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

All experiments are provided in a single Jupyter notebook: `faircp.ipynb`. 
The notebook is fully self-contained and allows reproducing both the **natural** and **adversarial** settings described in the paper.

### Running Experiments

To run the experiments, simply follow these steps:

1.  **Open the notebook:**
    ```bash
    jupyter notebook faircp.ipynb
    ```

2.  **Select the desired dataset:**
    Modifiy the `dataset_name` variable in the configuration cell:
    ```python
    dataset_name = "student"
    ```
    | Available Datasets | | |
    | :--- | :--- | :--- |
    | `adult` | `compas` | `german` |
    | `student` | `arrhythmia` | `bank` |

3.  **Set the experiment parameters:**
    ```python
    alpha = 0.1        # miscoverage level
    eps = 0.25         # adversarial perturbation strength
    lambd = 0.6        # fairness-attack trade-off
    ```

4.  **Run all cells.**

---

### What the Notebook Includes

The notebook automatically executes the following pipelines:

#### 1. Natural Setting
*   Standard conformal prediction evaluation.
*   **Metrics:** Coverage, Equalized Coverage, and Prediction set size.

#### 2. Adversarial Settings
*   Fairness-targeted adversarial attacks.
*   Evaluation under distribution shift.
*   Robust conformal prediction (defense).
*   **Scenarios considered:**
    *   **A1:** Attacks applied uniformly across the population.
    *   **A2:** Group-targeted attacks.

---

### Outputs

The notebook produces:
*   **Tables of results:** Detailed coverage and fairness metrics.
*   **Robustness comparisons:** Performance evaluation with and without defense mechanisms.
*   **Plots:** Visual representations showing the effect of adversarial perturbations.

> [!TIP]
> All results reported in the paper can be reproduced directly from this notebook.

---

### Notes

*   **Hardware:** The code automatically detects and uses GPU if available:
    ```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ```
*   **Dependencies:** No external fairness or adversarial libraries are required (all methods are implemented natively in this repository).
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
