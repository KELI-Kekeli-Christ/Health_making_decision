# Triagegeist Writeup — Triage Acuity Prediction from Structured Intake Data

*Template sections for the Kaggle Triagegeist competition writeup. Word count: ~1,350.*

---

## Clinical Problem Statement

This solution addresses a **narrow workflow gap**: the need for a **reproducible, data-driven estimate of triage acuity level** at the moment of patient intake in the emergency department (ED).

**Why it matters.** Triage nurses and physicians assign an acuity level (e.g. 1–5, ESI- or MTS-style) to each arriving patient under time pressure and with incomplete information. That level drives wait order and resource allocation. **Failure modes** are well documented: (1) **Undertriage** — assigning a lower urgency than warranted — delays care and can contribute to adverse outcomes; (2) **Inter-rater variability** — the same patient may receive different acuity scores depending on who performs triage; (3) **Cognitive load** — understaffing and high volume increase inconsistency. The people affected are **triage staff** (who make the decision), **patients** (whose wait and pathway depend on it), and **the rest of the ED** (through queue and flow).

This project does **not** aim to replace human triage or to “solve” ED crowding. It targets the **specific gap**: providing a **second signal** — a predicted acuity level from the same structured data available at triage — that clinicians can use to check consistency, flag possible undertriage, or support training. The outcome we predict is the **human-assigned triage acuity** (1 = most urgent, 5 = least urgent), not a downstream event such as death or ICU admission.

---

## Approach and Methodology

**Data.** We use the competition datasets: **train.csv** (~80k visits with `triage_acuity`), **test.csv** (~20k visits without labels), **chief_complaints.csv** (free-text and category), and **patient_history.csv** (comorbidities). Only the main **train** and **test** tables are used for modeling; chief complaints and patient history are loaded but not incorporated in this baseline, so the model relies on data that are available at triage in structured form (vitals, demographics, arrival context, chief complaint *category*, prior use, comorbidities count). This choice keeps the pipeline simple and avoids free-text processing for the baseline.

**Pipeline.**

1. **Load** — Read CSVs; detect Kaggle vs local paths for `DATA_DIR` and `OUT_DIR`.
2. **Exclude columns** — Drop `patient_id`, `triage_acuity`, and outcome-only columns (`disposition`, `ed_los_hours`) from the feature set so the model does not use information that would not be available or appropriate at triage prediction time.
3. **Categorical encoding** — All non-numeric columns (site_id, triage_nurse_id, arrival_mode, arrival_day, arrival_season, shift, age_group, sex, language, insurance_type, transport_origin, pain_location, mental_status_triage, chief_complaint_system) are encoded with **LabelEncoder** fit on the training set; unseen categories at test time are mapped to −1.
4. **Numeric handling** — Missing values in numeric columns are **imputed with the median** (computed on the training set for consistency). Any column still non-numeric after encoding (e.g. string dtype) is coerced with `pd.to_numeric(..., errors="coerce")`, then NaN → −1, and the full feature matrix is cast to **float32** for the model.
5. **Model** — **LightGBM** (LGBMClassifier) for multiclass classification (classes 1–5), with fixed hyperparameters (n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31, random_state=42). If LightGBM is unavailable, **RandomForestClassifier** is used as fallback.
6. **Evaluation** — **5-fold stratified cross-validation** on the training set; metric: **quadratic Cohen’s kappa**, which weights larger errors on the ordinal scale more heavily than adjacent-class errors and is appropriate for acuity as an ordinal outcome.
7. **Output** — Final model is fit on the full training set; predictions on the test set are clipped to [1, 5] and written to **submission.csv** with columns `patient_id` and `triage_acuity`.

**Rationale.** Tree-based models (LightGBM) handle mixed feature types and missingness after imputation without heavy feature engineering. Multiclass classification with kappa as the metric aligns with the ordinal nature of acuity and penalizes serious misclassification (e.g. 1 vs 5) more than near-neighbour errors (e.g. 2 vs 3). No ordinal-specific loss is used in this baseline.

---

## Results and Findings

**Metric.** Quadratic Cohen’s kappa in 5-fold stratified CV: **≈ 0.925** (single run with random_state=42).

**Interpretation.** Under the Landis–Koch interpretation, this falls in the “almost perfect” agreement range (>0.81) between predicted and human-assigned acuity on this dataset. So on average the model’s ordinal predictions align well with the labels in cross-validation.

**Clinical context.** A high kappa means the model often agrees with the triage nurse’s acuity level given the same structured inputs. It does **not** mean the model is “correct” in the sense of predicting who actually needed more urgent care; we are measuring agreement with the **human label**, which may itself be variable or biased.

**Limitations and failure modes (honest account).**

- **Redundancy with the target.** **news2_score** (National Early Warning Score 2) and **shock_index** are severity scores that are likely used (or highly correlated with what is used) when assigning acuity. Including them may allow the model to “echo” the same logic used to create the labels, inflating kappa. We did not run an ablation (e.g. dropping news2/shock_index) in this baseline; the reported kappa may overstate generalizable performance.
- **No external validation.** All evaluation is cross-validation on the same data source. Performance on other hospitals, regions, or time periods is unknown.
- **Target is human judgment.** We predict the nurse’s acuity, not an objective outcome (death, ICU admission, deterioration). So we are evaluating **consistency with current practice**, not clinical “correctness.”
- **Unseen categories.** Test-time categories not seen in training are encoded as −1, which may hurt performance for rare sites or complaint types.
- **Imputation.** Median imputation assumes missingness is at least not strongly informative; if missing vitals are associated with acuity, this could bias estimates.

A rigorous next step would be to report kappa **with and without** news2 and shock_index, and to analyse errors by acuity level and by subgroup (e.g. age, chief complaint category) to identify where the model fails.

---

## Limitations and Future Work

**What this solution does not do.**

- It does **not** replace triage or make a binding clinical decision.
- It does **not** explain why a given acuity was predicted (no interpretability layer).
- It does **not** use free-text chief complaints or structured patient_history in the model.
- It does **not** guarantee generalization to other EDs or countries without re-evaluation.

**Assumptions that may not hold in real use.**

- **Same distribution:** Train and test are assumed to come from a similar population and process; different sites or protocols could break this.
- **Data availability:** We assume the same structured fields are available at triage in the same form (e.g. news2 already computed).
- **Labels as gold standard:** We treat human-assigned acuity as the target; systematic undertriage in the labels would be learned by the model.
- **Missingness:** Median imputation may be inadequate if missingness is informative (e.g. unstable patients without complete vitals).

**What would be needed for clinical validation or deployment.**

- **External validation** on one or more EDs not used for training, and ideally prospective data.
- **Ablation studies** to quantify the contribution of news2 and shock_index and to report performance without them.
- **Outcome-based evaluation** where possible (e.g. correlation with disposition, ICU admission, or deterioration), not only agreement with the assigned acuity.
- **Fairness and safety analysis** (e.g. by age, sex, language, site) to detect differential performance or risk of exacerbating bias.
- **Integration and usability:** clear presentation of the prediction as an aid (e.g. “suggested acuity”), not as the final decision, and documentation of intended use and limitations for clinicians and governance.

---

## Reproducibility Notes

**Datasets.** All data are from the **Triagegeist competition** on Kaggle: [https://www.kaggle.com/competitions/triagegeist](https://www.kaggle.com/competitions/triagegeist). Files used: `train.csv`, `test.csv`, `chief_complaints.csv`, `patient_history.csv`. No external or proprietary datasets are required.

**Notebook.** The attached **Kaggle Notebook** runs **end-to-end without errors** when the competition dataset is attached as input. It is set to **public** at submission time. The notebook (1) loads data from `/kaggle/input/triagegeist` on Kaggle or from a local `data/` folder when run elsewhere, (2) runs EDA, preprocessing, 5-fold CV, and final fit, and (3) writes `submission.csv` to `/kaggle/working/` (Kaggle) or the current directory (local).

**Environment.**

- **Hardware:** Standard (Kaggle default or any machine that can run the notebook; no GPU required).
- **Key packages:** pandas, numpy, scikit-learn, lightgbm (optional; RandomForest used if LightGBM is missing). See `requirements.txt` in the project repository for versions.
- **Random seed:** `random_state=42` is set in LightGBM and in StratifiedKFold so that CV and training are reproducible.

**Reproducing the reported kappa.** Run the notebook from top to bottom with the competition data attached; the CV step prints the quadratic Cohen’s kappa (e.g. ~0.925). Re-running with the same seed should yield the same value. Submissions may vary if the competition test set or evaluation metric differ from this CV setup.
