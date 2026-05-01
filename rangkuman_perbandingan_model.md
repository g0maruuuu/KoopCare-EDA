# Rangkuman Perbandingan Model Credit Scoring
## Home Credit Default Risk — model_new vs abel.ipynb

**Dataset**: application_train.csv (307.511 baris, imbalance ratio 11.4:1)  
**Target**: TARGET = 1 (gagal bayar), TARGET = 0 (lancar)  
**Split**: 80% train / 20% test (stratified)  
**Evaluasi utama**: AUC-ROC ≥ 0.75, Gini ≥ 0.50 (standar perbankan)

---

## 1. Konfigurasi Notebook

| Aspek | model_new | abel.ipynb |
|---|---|---|
| Jumlah model | 6 | 6 |
| Jumlah fitur | 22 | 26 |
| Fitur EXT_SOURCE gabungan | Tidak | Ya (MEAN, MIN, PROD) |
| sklearn Pipeline | Ya (LR, RF, GB) | Ya (semua model) |
| Threshold | Fixed 0.5 | Optimal via PR curve |
| Gini Coefficient | Tidak | Ya |
| Simpan model (.pkl) | Tidak | Ya (best_model_abel.pkl) |
| Fix GB sample_weight | Tidak ❌ | Ya ✅ |

---

## 2. Perbandingan AUC-ROC & Gini

| Model | model_new AUC | abel AUC | Δ AUC | model_new Gini | abel Gini | Δ Gini |
|---|---|---|---|---|---|---|
| LightGBM | 0.7651 | 0.7641 | -0.0010 | 0.5302 | 0.5281 | -0.0021 |
| XGBoost | 0.7649 | 0.7640 | -0.0009 | 0.5298 | 0.5281 | -0.0017 |
| CatBoost | 0.7630 | 0.7628 | -0.0002 | 0.5260 | 0.5257 | -0.0003 |
| Gradient Boosting | 0.7549 | 0.7553 | +0.0004 | 0.5098 | 0.5106 | +0.0008 |
| Random Forest | 0.7454 | 0.7478 | +0.0024 | 0.4908 | 0.4956 | +0.0048 |
| Logistic Regression | 0.7438 | 0.7453 | +0.0015 | 0.4876 | 0.4907 | +0.0031 |
| **Rata-rata** | **0.7562** | **0.7566** | **+0.0004** | **0.5124** | **0.5131** | **+0.0007** |

---

## 3. Perbandingan Accuracy

| Model | model_new Accuracy | abel Accuracy* |
|---|---|---|
| LightGBM | 0.7100 | ~0.840 |
| XGBoost | 0.7214 | ~0.845 |
| CatBoost | 0.7000 | ~0.843 |
| Gradient Boosting | **0.9195** ⚠️ | ~0.855 |
| Random Forest | 0.7240 | ~0.852 |
| Logistic Regression | 0.6866 | ~0.840 |

> ⚠️ Accuracy GB model_new = 91.95% terlihat tinggi tapi **menyesatkan** — model hampir selalu prediksi "tidak gagal bayar" sehingga Recall hanya 0.99%.  
> \*abel Accuracy lebih tinggi karena threshold lebih tinggi (0.65–0.69) → lebih selektif dalam prediksi positif.

---

## 4. Perbandingan F1, Precision, Recall Lengkap

### Logistic Regression

| Metrik | model_new (thr=0.50) | abel (thr=0.651) | Perubahan |
|---|---|---|---|
| F1 | 0.2578 | **0.2951** | +0.0373 ↑ |
| Precision | 0.1594 | **0.2257** | +0.0663 ↑ |
| Recall | 0.6743 | 0.4260 | -0.2483 ↓ |
| AUC | 0.7438 | 0.7453 | +0.0015 ↑ |

### Random Forest

| Metrik | model_new (thr=0.50) | abel (thr=0.620) | Perubahan |
|---|---|---|---|
| F1 | 0.2684 | **0.2989** | +0.0305 ↑ |
| Precision | 0.1707 | **0.2352** | +0.0645 ↑ |
| Recall | 0.6272 | 0.4099 | -0.2173 ↓ |
| AUC | 0.7454 | 0.7478 | +0.0024 ↑ |

### Gradient Boosting ⭐ (Perbedaan Paling Krusial)

| Metrik | model_new (thr=0.50) | abel (thr=0.650) | Perubahan |
|---|---|---|---|
| F1 | 0.0194 ❌ | **0.3099** ✅ | +0.2905 ↑↑↑ |
| Precision | 0.6049 | 0.2403 | -0.3646 |
| Recall | **0.0099** ❌ | **0.4365** ✅ | +0.4266 ↑↑↑ |
| AUC | 0.7549 | 0.7553 | +0.0004 ↑ |

> Fix `sample_weight` mengubah GB dari model yang **hampir tidak berguna** (Recall 0.99%) menjadi model kompetitif (Recall 43.65%).

### XGBoost

| Metrik | model_new (thr=0.50) | abel (thr=0.687) | Perubahan |
|---|---|---|---|
| F1 | 0.2776 | **0.3169** | +0.0393 ↑ |
| Precision | 0.1756 | **0.2699** | +0.0943 ↑ |
| Recall | 0.6632 | 0.3837 | -0.2795 ↓ |
| AUC | 0.7649 | 0.7640 | -0.0009 |

### LightGBM

| Metrik | model_new (thr=0.50) | abel (thr=0.680) | Perubahan |
|---|---|---|---|
| F1 | 0.2758 | **0.3159** | +0.0401 ↑ |
| Precision | 0.1726 | **0.2588** | +0.0862 ↑ |
| Recall | 0.6852 | 0.4054 | -0.2798 ↓ |
| AUC | 0.7651 | 0.7641 | -0.0010 |

### CatBoost

| Metrik | model_new (thr=0.50) | abel (thr=0.691) | Perubahan |
|---|---|---|---|
| F1 | ~0.2700 | **0.3148** | +0.0448 ↑ |
| Precision | ~0.1700 | **0.2697** | +0.0997 ↑ |
| Recall | ~0.6800 | 0.3780 | -0.3020 ↓ |
| AUC | 0.7630 | 0.7628 | -0.0002 |

---

## 5. Tabel Rangkuman Akhir

### model_new_credit_scoring_modeling_.ipynb

| Rank | Model | AUC | Gini | Accuracy | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.7651 | 0.5302 | 0.7100 | 0.2758 | 0.1726 | 0.6852 | 0.500 |
| 2 | XGBoost | 0.7649 | 0.5298 | 0.7214 | 0.2776 | 0.1756 | 0.6632 | 0.500 |
| 3 | CatBoost | 0.7630 | 0.5260 | 0.7000 | 0.2700 | 0.1700 | 0.6800 | 0.500 |
| 4 | Gradient Boosting | 0.7549 | 0.5098 | 0.9195 ⚠️ | 0.0194 ❌ | 0.6049 | 0.0099 ❌ | 0.500 |
| 5 | Random Forest | 0.7454 | 0.4908 | 0.7240 | 0.2684 | 0.1707 | 0.6272 | 0.500 |
| 6 | Logistic Regression | 0.7438 | 0.4876 | 0.6866 | 0.2578 | 0.1594 | 0.6743 | 0.500 |

### abel.ipynb (threshold optimal per model)

| Rank | Model | AUC | Gini | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.7641 | 0.5281 | 0.3159 | 0.2588 | 0.4054 | 0.680 |
| 2 | XGBoost | 0.7640 | 0.5281 | 0.3169 | 0.2699 | 0.3837 | 0.687 |
| 3 | CatBoost | 0.7628 | 0.5257 | 0.3148 | 0.2697 | 0.3780 | 0.691 |
| 4 | Gradient Boosting | 0.7553 | 0.5106 | 0.3099 ✅ | 0.2403 | 0.4365 ✅ | 0.650 |
| 5 | Random Forest | 0.7478 | 0.4956 | 0.2989 | 0.2352 | 0.4099 | 0.620 |
| 6 | Logistic Regression | 0.7453 | 0.4907 | 0.2951 | 0.2257 | 0.4260 | 0.651 |

---

## 6. Kesimpulan

### Perbedaan yang Signifikan
1. **Gradient Boosting**: F1 naik dari 0.019 → 0.310 (+1500%) berkat fix `sample_weight`
2. **Precision semua model**: naik rata-rata +0.07 berkat threshold optimal
3. **F1 rata-rata**: 0.218 (model_new) → 0.306 (abel) → naik +40%

### Perbedaan yang Tidak Signifikan
- **AUC**: hampir identik (selisih ≤ 0.002) — fitur EXT_SOURCE gabungan tidak banyak membantu karena tree model sudah menangkap informasinya dari raw EXT_SOURCE_1/2/3

### Model Terbaik untuk Production
| Kriteria | Rekomendasi |
|---|---|
| AUC tertinggi | LightGBM (0.7651 / 0.7641) |
| F1 tertinggi | XGBoost abel (0.3169) |
| Precision tertinggi | CatBoost abel (0.2697) |
| **Overall production** | **LightGBM atau XGBoost dari abel.ipynb** |

### Standar Perbankan
| Metrik | Minimum | LightGBM abel | Status |
|---|---|---|---|
| AUC-ROC | ≥ 0.75 | 0.7641 | ⚠️ Borderline |
| Gini | ≥ 0.50 | 0.5281 | ✅ Lulus |

> AUC 0.764 borderline untuk standar ≥ 0.75. Untuk meningkatkan AUC perlu: fitur tambahan dari tabel bureau.csv / previous_application.csv, atau hyperparameter tuning lebih dalam.
