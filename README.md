<!-- =========================
     HERO / TITLE
     (Centered, readable, screen-reader friendly)
========================= -->

<h1 align="center">Clinical Diabetes Risk & GLP-1 Dispensing Analysis</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python 3.10 or higher" />
  <img src="https://img.shields.io/badge/Analytics-EDA%20%7C%20Inference-informational" alt="Analytics: exploratory data analysis and inference" />
  <img src="https://img.shields.io/badge/Stats-OLS%20%7C%20Logit-success" alt="Statistics: OLS and logistic regression" />
  <img src="https://img.shields.io/badge/Domain-Healthcare-critical" alt="Domain: healthcare" />
  <img src="https://img.shields.io/badge/Portfolio-Ready-brightgreen" alt="Portfolio ready" />
</p>

<p align="center">
  <em>
    Clinical Diabetes & Digital Health Analytics — Exploratory Data Analysis,
    statistical foundations and real-world prescription trends in digital health.
  </em>
</p>

<p align="center">
  <a href="#executive-summary-what-this-project-shows">Executive summary</a> ·
  <a href="#1-general-description">Description</a> ·
  <a href="#analytical-overview">Analytical overview</a> ·
  <a href="#about-the-datasets-used">Datasets</a> ·
  <a href="#column-guide-plain-english">Column guide</a> ·
  <a href="#2-project-objectives">Objectives</a> ·
  <a href="#3-repository-structure">Structure</a> ·
  <a href="#4-technologies-used">Tech</a> ·
  <a href="#5-how-to-run-the-project">Run</a> ·
  <a href="#6-data-cleaning-and-transformation">Cleaning</a> ·
  <a href="#visual-highlights">Visual highlights</a> ·
  <a href="#8-interpretation-and-methodology">Methodology</a> ·
  <a href="#key-findings-interpreted-not-overstated">Key findings</a> ·
  <a href="#key-learnings">Key learnings</a> ·
  <a href="#11-references">References</a>
</p>

<hr>

<!-- =========================
     EXECUTIVE SUMMARY
========================= -->

## Executive summary (what this project shows)

This repository demonstrates a full, end-to-end analytical workflow using two *independent* datasets: a synthetic patient-level diabetes risk dataset and an NHS aggregated GLP-1 dispensing time series (2019–2023). The datasets are intentionally **not merged** (different granularity and no patient linkage), so results are interpreted as **associations**, not causal effects.

### 1) NHS GLP-1 dispensing trends (system-level signal)

Dispensing volumes show a clear upward trajectory across the study period, with molecule-specific patterns rather than a single uniform trend. I quantify these trajectories using OLS trend slopes at monthly resolution and report uncertainty via confidence intervals. Where the Breusch–Pagan test indicates heteroscedasticity, I rely on HC3 robust standard errors to avoid overconfident inference.  
**Interpretation boundary:** dispensing growth is a real-world utilisation signal, but it cannot be attributed to individual risk changes without linked data.

### 2) Diabetes risk profile (patient-level signal)

The synthetic cohort displays a substantial overweight/obesity burden, and diagnosed diabetes prevalence increases across age bands and BMI categories, consistent with clinical expectation. Glycaemic markers behave coherently (fasting glucose and HbA1c move together), providing a useful “face validity” check that the synthetic data behave plausibly.

### 3) Added clinical signal beyond the obvious

A central insight is that **waist-to-hip ratio (WHR)** adds information beyond BMI: central adiposity can indicate elevated metabolic risk even when BMI remains below obesity thresholds. A simple BMI–WHR discordance split highlights a subgroup that could be under-identified by BMI-only screening logic. This is presented as an analytical lens, not a diagnostic rule.

### 4) Modelling layer: interpretable, not “black box”

I fit:

- a multiple linear regression model to explain variation in the dataset’s diabetes risk score, and  
- a logistic regression model for diagnosed diabetes, reported as odds ratios with confidence intervals.  

The intent is interpretability and communication: coefficients and uncertainty are emphasised, and model diagnostics are included to show when standard assumptions may not hold.

### Overall

Overall, the analysis confirms clinically expected patterns while demonstrating a rigorous, reproducible analytical workflow (EDA → inference → diagnostics → interpretable models) rather than aiming to produce novel clinical claims.

<hr>

<!-- =========================
     1. GENERAL DESCRIPTION
========================= -->

## 1. General Description

This project combines two complementary analytical perspectives to explore diabetes risk and digital health.

On one side, it uses a **synthetic clinical dataset for diabetes risk prediction**, built from medically validated ranges and distributions. Although the data do not represent real patients, they are suitable for exploratory analysis and machine-learning experimentation without compromising privacy.

On the other side, it incorporates **real-world NHS reimbursed prescription data** for GLP-1 weight-loss and diabetes-related medications such as Ozempic, Saxenda and Victoza. These records are aggregated at organisational level and reflect real pharmacological activity without individual patient information.

The two datasets are **not merged**. They are analysed independently to avoid incorrect inferences, while together providing a broader analytical view that links simulated clinical risk profiles with real medication-usage trends, always respecting the scope and limitations of each data source.

Beyond descriptive trends, the analysis explores risk discordance, metabolic profiles and behavioural factors to contextualise pharmacological uptake within broader public-health dynamics.

<hr>

<!-- =========================
     ANALYTICAL OVERVIEW
========================= -->

## Analytical overview

The diagram below summarises how the two independent datasets are analysed
and conceptually integrated within this project.

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/f0a43930-1674-4ac3-b476-b58ebf30e2e8"
    alt="Analytical overview of diabetes risk and GLP-1 dispensing analysis"
    width="780"
  />
</p>

<p align="center">
  <em>
    Two independent datasets are analysed separately and interpreted together.
    No individual-level linkage is performed.
  </em>
</p>

<hr>

<!-- =========================
     DATASETS
========================= -->

## About the datasets used

This project is based on two complementary data sources:

1. **Synthetic clinical dataset — patient-level diabetes risk**  
   Source: Kaggle — *Health & Lifestyle Data for Diabetes Prediction*  
   https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction/data  

   Although the records do not represent real patients, the dataset follows medically validated ranges and distributions. This makes it suitable for exploratory analysis and machine-learning experimentation without compromising privacy.

2. **Real-world NHS dispensing dataset — GLP-1 medications (2019–2023)**  
   Source: Kaggle — *Weight Loss Medications*  
   https://www.kaggle.com/datasets/mpwolke/medications  

   The records reflect actual reimbursed prescriptions within the NHS, aggregated at organisational level and free of identifiable patient information.

**Important:** these datasets are analysed independently and are not merged. One is synthetic and population-wide, while the other captures real prescription activity over time. Together, they provide two complementary perspectives: simulated clinical profiles and real medication-usage trends.

<hr>

<!-- =========================
     COLUMN GUIDE
========================= -->

## Column guide (plain English)

Units are included where standard; thresholds are used for interpretability, not for clinical diagnosis.  
This section explains the **key columns used in the notebook** in non-technical terms.

### Dataset 1 — `diabetes_health_indicators_patientlevel_synthetic.csv` (synthetic patients)

**Demographics**
- `age` — Age in years.
- `gender` — Biological sex category (Male/Female/Other).
- `ethnicity` — Broad ethnic background category.
- `education_level` — Highest education attained.
- `income_level` — Income group category.
- `employment_status` — Employment category (e.g., Employed, Retired).

**Lifestyle**
- `smoking_status` — Smoking behaviour (Never/Former/Current).
- `alcohol_consumption_per_week` — Average number of alcoholic drinks per week.
- `physical_activity_minutes_per_week` — Weekly minutes of physical activity.
- `diet_score` — Diet quality score (higher values indicate healthier patterns).
- `sleep_hours_per_day` — Average sleep duration (hours/day).
- `screen_time_hours_per_day` — Daily screen time (hours/day).

**Medical history (binary flags)**
- `family_history_diabetes` — Family history of diabetes (0 = No, 1 = Yes).
- `hypertension_history` — History of hypertension (0 = No, 1 = Yes).
- `cardiovascular_history` — History of cardiovascular conditions (0 = No, 1 = Yes).

**Clinical measurements**
- `bmi` — Body Mass Index in kg/m² (body weight relative to height).
- `waist_to_hip_ratio` — Waist-to-hip ratio (central adiposity indicator).
- `systolic_bp` — Systolic blood pressure (mmHg).
- `diastolic_bp` — Diastolic blood pressure (mmHg).
- `heart_rate` — Resting heart rate (beats per minute).
- `cholesterol_total` — Total cholesterol (mg/dL).
- `hdl_cholesterol` — HDL (“good”) cholesterol (mg/dL).
- `ldl_cholesterol` — LDL (“bad”) cholesterol (mg/dL).
- `triglycerides` — Triglycerides (mg/dL).
- `glucose_fasting` — Fasting blood glucose (mg/dL).
- `glucose_postprandial` — Post-meal blood glucose (mg/dL).
- `insulin_level` — Blood insulin concentration (µU/mL).
- `hba1c` — HbA1c (%) as an indicator of longer-term blood glucose levels.

**Targets / outcomes used in modelling**
- `diabetes_risk_score` — Composite diabetes risk score provided in the dataset (used for regression).
- `diagnosed_diabetes` — Diabetes diagnosis flag (0 = No, 1 = Yes) (used for logistic regression).
- `diabetes_stage` — Stage label (e.g., No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational).

> Note: this dataset is synthetic and intended for analysis practice and modelling demonstrations. Values are designed to be clinically plausible but do not represent real individuals.

### Dataset 2 — `glp1_weight_loss_medications_dispensing_2019_2023.csv` (NHS dispensing)

Each row represents dispensing activity for a given month and presentation.

- `Year Month` — Month identifier in `YYYYMM` format (parsed to a proper date in the notebook).
- `Prescribed BNF Presentation Code` — Code for the prescribed presentation.
- `Prescribed BNF Presentation` — Text description of what was prescribed.
- `BNF Chemical Substance` — Active ingredient (molecule) name (e.g., Semaglutide, Liraglutide).
- `Dispensed (Reimbursed) BNF Presentation Code` — Code for the reimbursed dispensed presentation.
- `Dispensed (Reimbursed) BNF Presentation` — Text description of what was actually dispensed/reimbursed.
- `Items` — Count of dispensed items (a volume measure).
- `Total Quantity` — Total quantity dispensed (depends on presentation; used alongside `Items`).

**Added for accessibility**
- `Commercial Name` — Stakeholder-friendly brand name derived from `BNF Chemical Substance` (e.g., Ozempic, Victoza).  
  This supports readability for non-clinical audiences, while molecule-level analysis remains the clinically stable reference.

> Note: dispensing data are aggregated at organisational level and contain no patient-level linkage.

<hr>

<!-- =========================
     2. OBJECTIVES
========================= -->

## 2. Project Objectives

- Understand clinical, lifestyle and demographic factors associated with diabetes risk.
- Apply rigorous exploratory data analysis techniques to healthcare data.
- Analyse real-world GLP-1 medication dispensing trends over time.
- Demonstrate statistical reasoning beyond descriptive metrics.
- Build a portfolio-ready healthcare analytics project aligned with digital health roles.
- Explicitly highlight limitations, assumptions and ethical considerations.
- Explore discordance between standard clinical indicators (e.g. BMI vs central adiposity) to highlight hidden risk profiles.
- Contextualise pharmacological trends alongside behavioural and metabolic factors, including ethical considerations around non-pharmacological alternatives.

<hr>

<!-- =========================
     3. REPO STRUCTURE
========================= -->

## 3. Repository Structure

```
clinical-diabetes-risk-glp1-analysis/
├── data/                         # Original datasets (CSV)
├── notebooks/                    # Executed notebooks
│   └── clinical_diabetes_glp1_analysis.ipynb
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

<hr>

<!-- =========================
     4. TECHNOLOGIES
========================= -->

## 4. Technologies Used

- **Python** (NumPy, pandas, matplotlib, seaborn)
- **Statsmodels** (OLS, logistic regression, diagnostics)
- **Scikit-learn** (train/test split, linear regression, evaluation metrics)
- **Jupyter Notebook / Google Colab**
- **Excel** (initial inspection and enrichment)
- **GitHub**

<hr>

<!-- =========================
     5. RUN
========================= -->

## 5. How to Run the Project

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Place the datasets in the `data/` folder.
4. Open the notebook in Jupyter or Google Colab.
5. Run the notebook from top to bottom.

<hr>

<!-- =========================
     6. CLEANING
========================= -->

## 6. Data Cleaning and Transformation

Key preprocessing steps include:

- Parsing monthly time variables (`Year Month → datetime`).
- Renaming columns for clarity and clinical transparency.
- Creating derived features (BMI categories, activity levels, age bands).
- Converting variables to appropriate data types.
- Checking duplicates and missing values.
- Flagging outliers using IQR rules without automatic exclusion.
- Adding a **Commercial Name** column to the medication dataset to improve accessibility for non-clinical audiences, while retaining molecule-level analysis for clinical stability.

<hr>

<!-- =========================
     VISUAL HIGHLIGHTS
========================= -->

## Visual Highlights

The notebook includes **59 carefully selected visualisations**, each designed to answer a specific analytical or clinical question rather than maximise quantity. The most relevant visuals are summarised below.

### GLP-1 dispensing trends (system-level)

- **Monthly dispensing time series (Plots 4–6)**  
  Show the sustained growth in GLP-1 prescribing between 2019 and 2023, both overall and by molecule, allowing comparison of adoption trajectories.

- **Yearly totals and year-over-year growth (Plots 10–11)**  
  Quantify acceleration patterns and contextualise short-term fluctuations within longer-term trends.

- **Molecule share over time (Plot 16)**  
  Highlights how market composition evolves, separating absolute growth from substitution effects.

- **OLS trend slopes with confidence intervals (Plots 17–21)**  
  Move beyond descriptive trends by formally estimating growth rates, checking heteroscedasticity and applying robust standard errors where needed.

**Why these matter:** together, these plots distinguish real structural growth from noise, while demonstrating statistically rigorous trend analysis on aggregated healthcare data.

### Diabetes risk profile (patient-level)

- **Core distributions (Plots 22–26)**  
  Establish face validity of age, BMI, glycaemic markers and risk score using clinically interpretable cut-offs.

- **Adiposity vs risk relationships (Plots 36a–36b)**  
  Compare BMI and waist-to-hip ratio, showing that central adiposity captures additional risk not fully explained by BMI alone.

- **Glycaemic concordance (Plot 38)**  
  Confirms expected alignment between fasting glucose and HbA1c, validating internal consistency of the synthetic dataset.

- **Risk stratification visuals (Plots 42–45)**  
  Show how diabetes prevalence and risk score vary across BMI categories, age bands and diagnosis status.

**Why these matter:** the visuals confirm clinically expected patterns while also highlighting subtle risk discordance, reinforcing careful interpretation rather than simplistic threshold-based conclusions.

### Behavioural and metabolic context

- **Physical activity and diet vs glycaemic control (Plots 41, lifestyle boxplots)**  
  Demonstrate that behavioural factors retain explanatory power alongside clinical markers.

- **Lipid profile scatter (Plot 41b)**  
  Frames diabetes risk within a broader metabolic context rather than glucose-only thinking.

- **Correlation structure (Plots 46–47)**  
  Summarises multivariate relationships and helps prioritise variables for modelling.

**Why these matter:** they contextualise pharmacological uptake within lifestyle and metabolic patterns, supporting ethical and public-health-aware interpretation.

### Modelling diagnostics and interpretability

- **Regression diagnostics (Plots 19–21, 49–51, 55)**  
  Residuals, Q–Q plots and fitted-vs-observed visuals ensure assumptions are checked, not assumed.

- **Coefficient and odds ratio plots (Plots 56–57)**  
  Translate models into interpretable effect directions and magnitudes for non-technical audiences.

- **Predicted probability curve (Plot 58)**  
  Illustrates how risk changes smoothly across BMI rather than at arbitrary cut-offs.

### Integrated perspective

- **Compact integration plot (Plot 59)**  
  Links patient-level risk profiles with system-level prescribing context without violating data granularity or making causal claims.

**Overall**, the visual strategy prioritises **clarity, clinical interpretability and statistical robustness**.  
The analysis confirms **clinically expected patterns** while demonstrating a **rigorous, reproducible analytical workflow**, rather than seeking novel or overstated clinical claims.

<hr>

<!-- =========================
     8. METHODOLOGY
========================= -->

## 8. Interpretation and Methodology

All findings are interpreted as **associations, not causal relationships**.

- The diabetes dataset is synthetic and cross-sectional.
- The dispensing dataset is real but aggregated at organisational level.
- There is no individual-level linkage between datasets.

Statistical models are used to quantify relationships, test hypotheses and assess uncertainty, while explicitly avoiding causal claims.  
Rather than linking datasets mechanically, the project integrates them conceptually: patient-level metabolic risk patterns are interpreted alongside system-level prescribing responses, reflecting how clinical need, behavioural risk and policy-driven access interact in real healthcare settings.

<hr>

<!-- =========================
     KEY FINDINGS
========================= -->

## Key findings (interpreted, not overstated)

- Diabetes prevalence increases across age bands and BMI categories, supporting expected risk stratification patterns for screening and prevention.
- HbA1c and fasting glucose show strong concordance, providing a face-validity check for the synthetic cohort’s clinical realism.
- WHR captures a central-adiposity subgroup that BMI-only thresholds can under-represent, motivating more nuanced risk framing in analytics.
- GLP-1 dispensing volumes rise over time with molecule-specific trajectories; trend uncertainty is reported with robust inference when assumptions fail.

<hr>

<!-- =========================
     KEY LEARNINGS
========================= -->

## Key learnings

- **Granularity is a design choice, not a detail.** Working with patient-level risk data alongside an aggregated NHS time series reinforced that “joining” is not always appropriate; respecting data structure prevents misleading narratives.

- **Clinical interpretability beats feature bloat.** Small, purposeful engineered features (BMI categories, age bands, activity levels) improved communication and stratified analysis without turning the notebook into a feature-factory.

- **Diagnostics change conclusions, not just aesthetics.** Running assumption checks (e.g., heteroscedasticity testing and robust HC3 standard errors) highlighted how easily naïve inference can look stronger than it is.

- **Validation is not only predictive accuracy.** “Face validity” checks (e.g., expected HbA1c–glucose concordance, higher prevalence across age/BMI strata) are essential when using synthetic datasets.

- **Risk is multi-dimensional.** WHR-based discordance showed how central adiposity can add signal beyond BMI, supporting more nuanced screening logic (presented as an analytical insight, not a clinical threshold).

- **Storytelling is part of the analysis.** Clear plot titles, consistent units, and short plain-language interpretations made the notebook readable for technical and clinical audiences without oversimplifying.

<hr>

<!-- =========================
     REFERENCES
========================= -->

## 11. References

- Health & Lifestyle Data for Diabetes Prediction (Synthetic)  
  https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction/data

- Weight Loss Medications — NHS Prescriptions  
  https://www.kaggle.com/datasets/mpwolke/medications

- WHO guidelines on BMI and physical activity  
- Statsmodels and scikit-learn documentation

<hr>

<p align="center">
  <em>
    Data doesn’t heal people — decisions do.
    So I treat every chart like a clinical conversation: clarify uncertainty, surface trade-offs, and stay honest about what the data can’t say.
  </em>
</p>
