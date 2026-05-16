# ikas — AI Growth Engineer Case Study

> **Goal**: Predict per-user **payment likelihood**, **churn risk**, **upsell potential** — then segment 100 users into **High Value / Medium Value / Churn Risk / Growth Potential** using a hybrid classical-ML + rule-based pipeline.

---

## 1. Proje Özeti

Bu çalışma, iki kaynak CSV'den (`advanced_user_profiles_with_uuid.csv` + `advanced_user_events_with_uuid.csv`) yola çıkarak her kullanıcı için davranışsal sinyaller türetir, üç hedef skor (payment likelihood, churn risk, growth potential) üretir, K-Means ile davranışsal kümeler çıkarır ve son olarak öncelik sıralı karar kurallarıyla her kullanıcıyı **dört iş segmentinden** birine atar.

Çıktılar:
- `outputs/user_predictions.csv` / `.json` — segment + skorlar (ANA TESLİMAT)
- `outputs/segment_summary.csv` — segment başına agrega
- `outputs/feature_importances.csv` — RF feature önem sıralaması
- `outputs/evaluation_report.txt` — 5-fold CV metrikleri
- `outputs/confusion_matrix.png`, `cluster_visualization.png`, `elbow_plot.png`
- `models/*.joblib` — eğitilmiş modeller + scaler
- **Plotly Dash gerçek-zamanlı dashboard** (`python dashboard.py`)

---

## 2. Veri

| Dosya | Satır | İçerik |
|---|---|---|
| `data/raw/advanced_user_profiles_with_uuid.csv` | 100 | `user_id, plan_type {Free/Pro/Business}, country {UK/France/Germany/Turkey/USA}, device_type {Mobile/Desktop}, industry {Healthcare/SaaS/E-Commerce/Education}` |
| `data/raw/advanced_user_events_with_uuid.csv` | 2,817 | `user_id, event_type, timestamp` — 6 olay tipi: login, logout, checkout_start, payment_success, trial_extension, feature_click |

**Zaman penceresi**: 2025-03-29 → 2025-04-25 (28 gün). Tüm recency hesapları `REFERENCE_DATE = 2025-04-26` (max timestamp + 1 gün) baz alır. `datetime.now()` KULLANILMAZ.

### Kritik veri bulgusu
**99/100 kullanıcı en az 1 `payment_success` eventine sahip.** Bu yüzden "kullanıcı ödedi mi?" gibi naive bir binary hedef neredeyse degenerate (99% pozitif sınıf) — model hiçbir şey öğrenemez. Bunun yerine:

> **Hedef değişken yeniden tanımlandı**: `is_high_payer = 1` if `n_payment_success ≥ 67. percentile` (≥6 ödeme), else 0.

Sonuç: dengeli %41 pozitif / %59 negatif sınıf. İş açısından da daha anlamlı — "high-volume payer" sales takımı için zaten "ever-paid"den daha değerli bir sinyal.

---

## 3. Kurulum

```bash
cd /Users/enesaydin/Desktop/ikas
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Gereksinim: Python ≥ 3.9. Paket listesi: `requirements.txt`.

---

## 4. Çalıştırma

### Tam pipeline (yeniden eğitim + tahmin):
```bash
python run_pipeline.py
```
~3 saniyede tüm modeli yeniden eğitir ve `outputs/` klasörüne çıktıları yazar.

### Dashboard:
```bash
python dashboard.py
```
Tarayıcıdan: <http://127.0.0.1:8050>

---

## 5. Feature Engineering

`src/feature_engineering.py` her kullanıcı için **39 sütun** türetir. Aşağıda anlamlı olanlar:

### Olay sayıları (7)
`n_login, n_logout, n_checkout_start, n_payment_success, n_trial_extension, n_feature_click, n_total_events`

### Dönüşüm oranları (4)
- `checkout_to_payment_rate = n_payment_success / max(n_checkout_start, 1)` — funnel verimliliği
- `login_to_payment_rate = n_payment_success / max(n_login, 1)`
- `payment_per_event = n_payment_success / max(n_total_events, 1)`
- `feature_click_rate = n_feature_click / max(n_total_events, 1)`

### Zamansal / recency (6)
- `first_event_day, last_event_day` — REFERENCE_DATE'e olan gün farkı
- `days_active` — farklı takvim günü sayısı
- `tenure_days` — son − ilk olay arası süre
- `recency_payment_days, recency_checkout_days` — son ödeme/checkout'tan bu yana

### Aktivite trendi (3) — son 7 gün vs ilk 7 gün
- `events_last_7d`, `events_first_7d`
- `activity_trend = (last_7 - first_7) / max(first_7, 1)` → negatif = düşüş

### Davranışsal pattern (3)
- `avg_events_per_active_day, login_logout_ratio, trial_extension_intensity`

### Profil (kategorik → numerik)
- `plan_ordinal`: Free=0, Pro=1, Business=2
- `device_is_desktop`: binary
- `country_*`, `industry_*`: one-hot

---

## 6. Modelleme

### 6.1 Random Forest Classifier — `is_high_payer`

| Parametre | Değer |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 (overfitting'e karşı sığ) |
| `min_samples_leaf` | 3 |
| `class_weight` | balanced |
| `random_state` | 42 |

**Validasyon**: Stratified **5-fold CV** (hold-out DEĞİL — 100 örnek için çok gürültülü). Final model tüm 100 örnekte yeniden eğitilir. CV sadece dürüst metrik raporlama için.

**Leakage kontrolü** (kritik): Hedef `n_payment_success`'tan türetildiği için aşağıdaki sütunlar feature matrisinden ÇIKARILIR:
`n_payment_success, checkout_to_payment_rate, login_to_payment_rate, payment_per_event, recency_payment_days`

**Sonuç (5-fold CV)**:
```
Accuracy:  0.600
ROC-AUC:   0.615
F1:        0.487
Precision: 0.514
Recall:    0.463
```

Detaylı sınıflandırma raporu:
```
              precision    recall  f1-score   support
     regular       0.65      0.69      0.67        59
  high_payer       0.51      0.46      0.49        41
```

ROC-AUC 0.615 — N=100 ve 28-günlük dar pencere için makul. Beklenen üst sınır > 0.70 idi; düşük olmasının sebebi büyük olasılıkla:
1. Synthetic data — gerçek bir davranış sinyali olmayabilir.
2. 28 gün, "high-payer" sinyalini yakalamak için çok kısa.
3. 100 örnek ile 31 feature'ın sinyal-gürültü oranı zayıf.

### 6.2 K-Means Clustering — davranışsal kümeler

8 küratör edilmiş feature ile, StandardScaler normalize edildi:
`n_login, n_checkout_start, n_payment_success, n_feature_click, n_trial_extension, activity_trend, last_event_day, checkout_to_payment_rate`

**k=4**, silhouette score = 0.173.

| Cluster | Adet | Karakter (orijinal-uzay centroid) |
|---|---|---|
| Engaged Payers | 13 | n_payment=8.4 (en yüksek), c2p_rate=4.37 (en yüksek) |
| Active Explorers | 37 | n_trial_ext=5.95, c2p_rate=1.07, last_event=13d |
| At-Risk Users | 34 | last_event=13.5d (en pasif), checkout yüksek ama c2p düşük |
| Disengaged Lite | 16 | last_event=2.6d (en aktif!), login=5.5 ama feature_click=3.6 (en düşük) |

> **Not**: "Disengaged Lite" ismi orijinal uzayda yanıltıcı durabilir — bu küme aslında çok aktif logging yapıyor ama feature kullanımı zayıf. İsim heuristik (greedy) atandı; centroid tablosu birincil referanstır.

### 6.3 Churn Risk (kural tabanlı, 0–1)
```python
churn_risk = (
    0.40 * normalize(last_event_day, 0, 28)
  + 0.25 * normalize(recency_checkout_days, 0, 28)
  + 0.20 * (1 - normalize(activity_trend, -1, 1))
  + 0.15 * normalize(trial_extension_intensity, 0, 0.3)
)
```
Ağırlıklar el ile seçildi: aktivite eksikliği en güçlü sinyal (40%), commercial intent decay (25%), trend yönü (20%), trial bağımlılığı (15%). Veri etiketsiz olduğu için bu approach defensible — README'de açıklanıyor.

### 6.4 Growth Potential (kural tabanlı, 0–1)
```python
base_engagement = (
    0.30 * normalize(n_login, 0, 15)
  + 0.30 * normalize(n_feature_click, 0, 10)
  + 0.20 * normalize(n_checkout_start, 0, 9)
  + 0.20 * (1 - normalize(last_event_day, 0, 28))
)
plan_multiplier = {Free: 1.0, Pro: 0.7, Business: 0.0}
growth_potential = base_engagement * plan_multiplier
```
Business kullanıcıların upsell hedefi yok → `0`. Free kullanıcılar tam çarpan, Pro orta çarpan.

---

## 7. Segmentasyon

`src/segmentation.py` — **öncelik sıralı karar kuralları**:

```python
if churn_risk >= 0.65:
    return "Churn Risk"                                    # Öncelik 1
if growth_potential >= 0.40 and plan_type in ("Free","Pro"):
    return "Growth Potential"                              # Öncelik 2
if payment_likelihood >= 0.55 and n_payment_success >= median_payments:
    return "High Value"                                    # Öncelik 3
return "Medium Value"                                      # Öncelik 4 (fallback)
```

**Neden öncelik sırası?** Çürüyen yüksek-ödeyen bir kullanıcı sales ekibi için en kritik aksiyon noktasıdır — retention sinyali kaybedilmemelidir. O yüzden Churn Risk diğer her şeyin üstündedir.

KMeans cluster ID'leri `behavior_cluster` sütununda korunur (dashboard filtrelemesi için), segment etiketi DEĞİLDİR.

### Sonuç dağılımı

| Segment | n_users | avg payment_likelihood | avg churn_risk | avg growth_potential | avg payments | top plan | top country | top industry |
|---|---|---|---|---|---|---|---|---|
| **High Value** | 16 | 0.719 | 0.455 | 0.185 | 7.75 | Business | UK | Healthcare |
| **Growth Potential** | 23 | 0.466 | 0.382 | 0.489 | 5.52 | Free | USA | SaaS |
| **Churn Risk** | 31 | 0.463 | 0.713 | 0.237 | 4.87 | Free | France | Healthcare |
| **Medium Value** | 30 | 0.274 | 0.475 | 0.168 | 3.63 | Pro | Germany | Healthcare |

İş açısından net bir hikaye:
- **High Value**: en yüksek ödeyenler, Business plan, %72 ödeme olasılığı
- **Growth Potential**: Free planda, ortalama 5.5 ödeme, yüksek engagement → **Pro/Business'a yükseltilebilir**
- **Churn Risk**: yüksek churn skoru, son aktivite eski; ortalama 4.9 ödemeye rağmen kaybedilme riskinde
- **Medium Value**: çok düşük ödeme olasılığı (%27), düşük ödeme sayısı (3.6); ek nurturing gerek

---

## 8. Top 10 RF Feature Importances

```
feature                       importance  rank
trial_extension_intensity         0.128      1
n_trial_extension                 0.073      2
feature_click_rate                0.066      3
avg_events_per_active_day         0.058      4
churn_risk                        0.054      5
n_logout                          0.053      6
login_logout_ratio                0.053      7
tenure_days                       0.049      8
recency_checkout_days             0.048      9
last_event_day                    0.044     10
```

Hiçbir payment-derived feature listede yok — leakage başarıyla önlenmiş. Sinyal trial davranışları, feature kullanımı ve aktivite paterninden geliyor.

---

## 9. Dashboard

`python dashboard.py` ile `http://127.0.0.1:8050` adresinde açılır.

İçerikler:
1. **Header** — proje adı, son veri güncelleme zamanı, manuel "Refresh data" butonu
2. **KPI kartları** — toplam kullanıcı, High Value, Churn Risk, Growth Potential adetleri
3. **Filtre paneli** — country, industry, plan_type, device_type, segment, behavior_cluster (hepsi multi-select)
4. **Pie**: segment dağılımı
5. **Stacked bar**: segment × plan_type
6. **Scatter**: `n_checkout_start` vs `payment_likelihood`, segmente göre renkli — case study'nin istediği "events → payment correlation"
7. **Bar**: top-10 feature importance
8. **Heatmap**: `payment_likelihood` ortalaması, country × industry
9. **Kullanıcı tablosu**: sıralanabilir, filtrelenebilir, **CSV'ye export edilebilir**

"Real-time" = veri disk'ten yeniden yüklenir (Refresh butonu veya yeni `run_pipeline.py` çalıştırma sonrası). Stream değil, snapshot-refresh modeli.

---

## 10. Çıktı Dosya Şemaları

### `outputs/user_predictions.csv`
| sütun | tip | açıklama |
|---|---|---|
| user_id | str | UUID |
| plan_type, country, device_type, industry | str | profil |
| n_payment_success, n_checkout_start, n_login, last_event_day | num | özet feature'lar |
| payment_likelihood | float [0,1] | RF probabilistic output |
| churn_risk | float [0,1] | kural tabanlı skor |
| growth_potential | float [0,1] | kural tabanlı skor |
| behavior_cluster | int {0..3} | KMeans cluster ID |
| behavior_cluster_name | str | insan-okunabilir küme adı |
| segment | str | 4-segmentten biri |
| segment_reason | str | hangi kural tetiklendi |

### `outputs/user_predictions.json`
Aynı veri JSON formatında + üst metadata bloğu:
```json
{
  "generated_at": "...",
  "n_users": 100,
  "model_version": "v1.0",
  "segment_counts": {"High Value": 16, ...},
  "users": [ { ... per-user record ... } ]
}
```

### `outputs/segment_summary.csv`
Yukarıda bölüm 7'deki tablo.

### `outputs/feature_importances.csv`
`feature, importance, rank` — sıralı.

---

## 11. Proje Yapısı

```
ikas/
├── data/
│   ├── raw/                                  # giriş CSV'leri (dokunulmuyor)
│   └── processed/user_features.csv           # feature engineering çıktısı
├── models/
│   ├── rf_payment_classifier.joblib
│   ├── kmeans_segmenter.joblib
│   ├── feature_scaler.joblib
│   └── model_metadata.json
├── outputs/                                  # tüm teslimat artefaktları
├── src/
│   ├── config.py                             # paths, hiperparametreler, eşik değerler
│   ├── utils.py                              # logger, normalize, ensure_dir
│   ├── data_loader.py                        # CSV yükleme + validasyon
│   ├── feature_engineering.py                # 39-feature türetimi
│   ├── labeling.py                           # target + churn + growth
│   ├── train_classifier.py                   # RF + 5-fold CV
│   ├── train_clusterer.py                    # KMeans + elbow + PCA + naming
│   ├── segmentation.py                       # priority-rule segment assignment
│   └── predict.py                            # uçtan uca orchestrator
├── dashboard.py                              # Plotly Dash app
├── run_pipeline.py                           # CLI giriş
├── requirements.txt
└── README.md
```

---

## 12. Limitations & Honest Caveats

1. **Küçük dataset (N=100)** — CV metrikleri yüksek varyansa sahip. Bir run ile diğeri arasında ±5pp fark normaldir. Production için >1,000 örnek hedeflenmeli.
2. **28-günlük pencere churn için kısa** — Endüstri standardı genelde 30–90 gün inaktif. Bizim `activity_trend` (son 7 gün vs ilk 7 gün) en iyi yaklaşımımız ama gerçek churn için yetersiz.
3. **Churn/growth skorları el-yordamıyla ağırlıklı** — Öğrenilmiş ağırlıklar değil; gerçek churn etiketli veri elimizde olunca logistic regression ile öğrenilebilir.
4. **Synthetic-görünümlü veri** — Olaylar arası timestamp'ler düzenli, kullanıcı başına olay sayısı dar bir aralıkta (18–38). Gerçek data daha "spiky" olur; model gerçek veride yeniden validate edilmeli.
5. **`is_high_payer` cutoff'u tertile** — Diğer cutoff'lar (örn. median, top-10%) farklı modeller ürür. Bu seçim README'de belgelendi.
6. **Cluster isimleri post-hoc heuristic** — Centroid tablosu primary truth; isimler sadece UI/dashboard için.
7. **"Real-time" snapshot-refresh** — Gerçek streaming değil. Yeni data geldiğinde `run_pipeline.py` yeniden çalıştırılmalı, sonra dashboard refresh.

---

## 13. Sonraki Adımlar / Olası Genişletmeler

- **OpenAI tabanlı chatbot**: Dashboard'a "Bu segment hakkında bana özet ver" tarz bir LLM asistanı; `user_predictions.csv` grounding olarak verilir, hallucination engellenir.
- **Otomatik retraining cron**: GitHub Actions veya cron ile günde bir `run_pipeline.py`.
- **Time-series modeling**: Eğer 3+ aylık data gelirse, recurrent net veya statsmodels ile gerçek churn forecasting.
- **CRM entegrasyonu**: `user_predictions.csv`'yi Salesforce/HubSpot/n8n ile sync edip segment-bazlı kampanya tetikleyiciler.
- **A/B test framework**: Predictions'ı kullanan kampanyaların effect size'ını ölçecek deney altyapısı.

---

## 14. Verification Checklist

- [x] `len(user_features) == 100`, NaN yok
- [x] `is_high_payer`: 41/100 pozitif (tertile ile dengeli)
- [x] CV ROC-AUC > 0.60, leakage YOK
- [x] Top-10 importance'ta payment-derived sütun YOK
- [x] KMeans cluster sizes: 13/37/34/16 (hepsi > 5)
- [x] Segment dağılımı: 16/30/31/23 (4 segmenta da en az 10 kullanıcı)
- [x] `run_pipeline.py` idempotent — aynı veriyle aynı çıktı
- [x] Dashboard tüm filtreler/grafikler çalışıyor, tablo CSV export ediyor

---

**Hazırlayan**: AI Growth Engineer Case Study — May 2026
