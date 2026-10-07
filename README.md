# 🏆 Spoken Audio Grammar Scoring Engine

> **SHL AI / Data Science Hiring Assessment 2026**  
> **Task:** Predict continuous Mean Opinion Score (MOS) Likert grammar ratings (1.0 to 5.0) from spoken English audio recordings.  
> **Evaluation Metric:** Root Mean Squared Error (RMSE) and Pearson Correlation ($r$).

---

## 📌 Executive Summary

This repository contains the complete, reproducible end-to-end pipeline for automated grammar evaluation of spoken English audio recordings. The solution combines **multimodal feature engineering** (acoustic prosody + linguistic error metrics + sentence embeddings + character/word n-grams) with a **5-fold stacking ensemble** and **dual foundation LLM rubric scoring** calibrated directly to human rater distributions.

### Key Performance Metrics
* **Compulsory Training Data RMSE:** `0.4919` (Pearson $r = 0.8896$)
* **5-Fold Cross-Validation OOF RMSE:** `0.6797` (Pearson $r = 0.7467$)
* **Base Model OOF Performance:**
  * LightGBM Regressor: `RMSE = 0.7015`, `r = 0.7287`
  * Bayesian Ridge: `RMSE = 0.7144`, `r = 0.7125`
  * Gradient Boosting: `RMSE = 0.7260`, `r = 0.7120`
  * Ridge Regressor: `RMSE = 0.7312`, `r = 0.7047`
  * ExtraTrees Regressor: `RMSE = 0.7388`, `r = 0.7077`

---

## 🏗️ Architecture Overview

```text
                        ┌───────────────────────────────┐
                        │      Spoken Audio (.wav)      │
                        └───────────────┬───────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                │                                               │
                ▼                                               ▼
   [ Automatic Speech Recognition ]             [ Torchaudio Acoustic Extraction ]
       Whisper-Large-v3 Transcripts                 87 Prosody & Spectral Features
                │                                    - 20 MFCCs (mean & std)
        ┌───────┼──────────────────────┐             - 20 Delta-MFCCs (mean & std)
        │       │                      │             - Pitch F0 Autocorrelation
        ▼       ▼                      ▼             - Voicing Ratio & RMS Energy
  [LanguageTool] [SBERT MiniLM-L6] [Sublinear N-grams]          │
   30 Grammar &   24 PCA Semantic   Word (1-2) + Char (3-5)     │
  Syntax Metrics    Embeddings       (8,000 sparse dims)        │
        │               │                      │                │
        └───────────────┼──────────────────────┴────────────────┘
                        ▼
       [ Unified Feature Matrix (8,141 dimensions) ]
                        │
                        ▼
       [ 5-Fold Multi-Model Stacking Ensemble ]
       LightGBM + Ridge + BayesianRidge + GBR + ExtraTrees
                        │ (Weight: 15%)
                        ├────────────────────────┐
                        │                        │
                        │                        ▼
                        │        [ Dual Foundation LLM Consensus ]
                        │        Gemini Flash Lite + Qwen-27B Rubrics
                        │        Calibrated to human Likert distribution
                        │                        │ (Weight: 85%)
                        ▼                        ▼
       ┌────────────────────────────────────────────────────────┐
       │     Acoustic-Linguistic Integration & Target Scaling   │
       │     MMSE Mean & Variance Alignment (μ=3.339, σ=0.800)  │
       └────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                   [ Final submission 1.csv ]
                   (216 rows, strictly bounded 1.0 - 5.0)
```

---

## 🔬 Methodology & Key Innovations

### 1. Data Cleaning & Outlier Filtering
* Identified and filtered out 37 corrupt training audio records (`audio_50*` sequence) erroneously assigned labels of `0.0` (outside the legitimate 1.0–5.0 Likert evaluation range).
* Resulted in **721 high-quality clean training records** ensuring stable gradients and preventing scale distortion.

### 2. Multimodal Feature Engineering (8,141 Dimensions)
* **Acoustic Prosody (87 features):** Extracted using `torchaudio` and `scipy`:
  * 20 MFCC coefficients (mean & standard deviation across frames).
  * 20 First-order Delta-MFCCs capturing speech dynamics and tempo.
  * Pitch $F_0$ contour via autocorrelation (mean, standard deviation, and voicing frame ratio).
  * RMS energy dynamics capturing volume stability and hesitation.
* **Linguistic & Grammatical Metrics (30 features):** Grammatical error frequency, spelling error counts, sentence complexity, syntactic depth, and readability indices.
* **Semantic Embeddings (24 features):** 384-dimensional dense representations from `sentence-transformers/all-MiniLM-L6-v2`, compressed via PCA preserving primary semantic variance.
* **Lexical N-Gram Representations (8,000 features):** Sublinear term-frequency word n-grams (1–2) and character n-grams (3–5) to identify subtle grammatical slips and morphological markers.

### 3. Multi-Model Stacking Ensemble
* Evaluated 5 diverse regression families using 5-Fold Cross Validation.
* Out-of-fold predictions were ensembled using **Sequential Least Squares Programming (SLSQP)** to find optimal constrained weights, achieving an OOF RMSE of **0.6797**.

### 4. Dual Foundation LLM Rubric Evaluation
* Evaluated full transcripts against the standardized Likert grammar scoring rubric using **Google Gemini Flash Lite** and **Qwen-27B**.
* Applied empirical linear transformation to align LLM scoring tendencies with the ground-truth human rater distribution.

### 5. Bayesian MMSE Calibration & Extremes Debiasing
* Blended 85% high-capacity LLM consensus with 15% acoustic prosody model to account for spoken fluency and delivery.
* Scaled target distribution to match true population parameters ($\mu = 3.339, \sigma = 0.800$), strictly clipping predictions to the valid `[1.0, 5.0]` interval.

---

## 📁 Repository Structure

```text
├── README.md                                 # Complete project documentation & methodology
├── grammar_scoring_engine_executed.ipynb     # Fully executed Jupyter Notebook with outputs & plots
└── submission 1.csv                          # Final test predictions (216 samples)
```

---

## 📊 Submission File Verification

The final submission file `submission 1.csv` satisfies all competition constraints:
* **Row Count:** Exactly `216` rows matching `test.csv`.
* **Columns:** `filename,label`
* **Score Bounds:** Minimum `1.8007`, Maximum `5.0000` (all within $[1.0, 5.0]$).
* **Mean Score:** `3.3390`
* **Standard Deviation:** `0.8003`
* **Missing Values:** `0` null / NaN entries.

---

## 🚀 How to Run / Reproduce

### Requirements
* Python 3.10+
* PyTorch & Torchaudio
* Scikit-Learn
* LightGBM
* Transformers & Sentence-Transformers

```bash
pip install torch torchaudio scikit-learn lightgbm sentence-transformers scipy pandas numpy
```

Open and run `grammar_scoring_engine_executed.ipynb` in any Jupyter environment. All intermediate outputs and final visualizations are pre-rendered in the notebook for immediate review.
