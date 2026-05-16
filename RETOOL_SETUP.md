# Retool Integration Guide

Bu kılavuz, ikas Growth API'sini Retool'a bağlayıp dashboard kurmanı adım adım anlatır. Toplam süre ~30-45 dk.

---

## Genel Mimari

```
   [Local Mac]                    [Internet]                  [Retool Cloud]
   ┌──────────────┐               ┌──────────┐                ┌─────────────┐
   │  api.py      │ ── tunnel ──► │  ngrok   │ ── HTTPS ────► │  Retool app │
   │  (FastAPI)   │               │          │                │   (UI)      │
   │  :8000       │               │  public  │                │             │
   └──────────────┘               │   URL    │                └─────────────┘
                                  └──────────┘
```

Senin Mac'inde **api.py** çalışır (port 8000). **ngrok** bunu internete public bir HTTPS URL ile expose eder. Retool da bu URL'e HTTP istekleri atar.

---

## Bölüm 1 — Senin Yapacakların (Local'de)

### Adım 1.1: API'yi Başlat

Terminal 1'de:
```bash
cd /Users/enesaydin/Desktop/ikas
source venv/bin/activate
python api.py
```

Şunu göreceksin:
```
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test et**: Tarayıcıda <http://127.0.0.1:8000/docs> aç — Swagger UI tüm endpoint'leri gösterecek.

### Adım 1.2: ngrok Kur

ngrok, local server'ını internete açar (ücretsiz tier yeterli).

```bash
# Mac'te brew ile (yoksa: https://ngrok.com/download)
brew install ngrok
```

Sonra ngrok hesabı aç (https://dashboard.ngrok.com/signup, ücretsiz, 1 dk). Authtoken'ını al:
```bash
ngrok config add-authtoken <SENIN_TOKEN>
```

### Adım 1.3: API'yi Tunnel ile Aç

Terminal 2'de (api.py çalışırken):
```bash
ngrok http 8000
```

Şunu göreceksin:
```
Forwarding   https://abc123-xx-xx.ngrok-free.app  ->  http://localhost:8000
```

> **Bu URL'i kopyala** — Retool'a vereceksin. URL her ngrok başlatışında değişir (ücretsiz tier). Sabit URL için $8/ay ngrok plan veya alternatif: Cloudflare Tunnel.

**Test et**: Browser'da `https://abc123-xx-xx.ngrok-free.app/health` aç — `{"status":"ok",...}` görmeli.

> **NOT**: ngrok bir uyarı sayfası gösterirse, "Visit Site" butonuna bas. Bu sadece ilk istekte oluyor, Retool'un sonraki istekleri direkt geçer.

---

## Bölüm 2 — Retool'da Yapılacaklar

### Adım 2.1: Yeni App Oluştur

1. Retool'a login (<https://retool.com>)
2. Sol menüden **Apps** → **Create new** → **App** seç
3. İsim ver: "ikas Growth Dashboard"

### Adım 2.2: REST API Resource Ekle

1. Sol panel'de **Resources** sekmesine git (alt taraf)
2. **+ Create new resource** → **REST API** seç
3. Şu alanları doldur:
   - **Name**: `ikas-growth-api`
   - **Base URL**: `https://abc123-xx-xx.ngrok-free.app` (ngrok URL'in, **sonunda / olmasın**)
   - **Headers**: Boş bırak
   - **Authentication**: None
4. **Test connection** butonuna bas → "Connection successful" görmeli
5. **Save**

### Adım 2.3: İlk Query'yi Oluştur

Sol altta **Queries** paneli, **+ New** → **Resource query**:

1. **Resource**: `ikas-growth-api`
2. **Action type**: GET
3. **URL**: `/dashboard-data`
4. Sağ üstte **Run** butonuna bas — JSON cevabı görmeli (KPI'lar + 100 user)
5. Query'yi adlandır: `getDashboardData`
6. **Run query on page load**: ON yap
7. **Save**

> ✓ Bu tek query bizim "her şey" endpoint'imiz — KPI'lar, predictions, feature importances hepsi içinde.

### Adım 2.4: UI'yi Kur — Component'ler

Sol panelde **Components** sekmesi. Sürükle-bırak ekle:

#### A) KPI Kartları (4 adet, üst sıra)
1. **Statistic** component'i sürükle ekle
2. Sağ panelden **Value**: `{{ getDashboardData.data.kpis.total_users }}`
3. **Label**: "Total Users"
4. Aynı şekilde 3 tane daha:
   - High Value → `{{ getDashboardData.data.kpis.high_value }}` (color: green)
   - Churn Risk → `{{ getDashboardData.data.kpis.churn_risk }}` (color: red)
   - Growth Potential → `{{ getDashboardData.data.kpis.growth_potential }}` (color: orange)

#### B) Segment Distribution — Pie Chart
1. **Chart** component'i sürükle (Plotly chart)
2. **Chart type**: Pie
3. **Data source**: `{{ getDashboardData.data.predictions }}`
4. **Labels**: `segment`
5. **Aggregation**: Count
6. Color palette: özel renkler atayabilirsin (High Value=green, Churn=red, vs.)

#### C) Predictions Tablosu
1. **Table** component'i ekle
2. **Data source**: `{{ getDashboardData.data.predictions }}`
3. Görünecek sütunlar (sağ panel → Columns):
   - `user_id`, `plan_type`, `country`, `industry`, `segment`, `payment_likelihood`, `churn_risk`, `growth_potential`, `n_payment_success`, `last_event_day`
4. **Searchable**: ON
5. **Sortable**: ON (her sütun)
6. **Pagination**: ON, page size 15
7. **Export to CSV**: General settings → Show download button ON
8. **Conditional styling**: Segment sütunu için
   - High Value → background: light green
   - Churn Risk → background: light red
   - Growth Potential → background: light orange
   - Medium Value → background: light blue

#### D) Filtreler — Multiselect Dropdowns

Önce dropdown'ları beslemek için ikinci bir query gerek:
1. **Queries** → **+ New** → **Resource query**
2. URL: `/filters` → Run on page load ON → adlandır: `getFilterOptions`

Sonra 5 adet **Multiselect** component ekle:
1. **Multiselect** sürükle, Label: "Country"
   - **Data source**: `{{ getFilterOptions.data.country }}`
   - **Allow multiple**: ON
   - Component name: `filterCountry`
2. Aynı şekilde: `filterIndustry`, `filterPlan`, `filterDevice`, `filterSegment`

Şimdi tablo'yu filtrele:
- Table'ın **Data source**'unu güncelle:
```javascript
{{ getDashboardData.data.predictions.filter(u =>
    (filterCountry.value.length === 0 || filterCountry.value.includes(u.country)) &&
    (filterIndustry.value.length === 0 || filterIndustry.value.includes(u.industry)) &&
    (filterPlan.value.length === 0 || filterPlan.value.includes(u.plan_type)) &&
    (filterDevice.value.length === 0 || filterDevice.value.includes(u.device_type)) &&
    (filterSegment.value.length === 0 || filterSegment.value.includes(u.segment))
) }}
```

> Bu transformation'ı bir kez yazıp tüm component'lere referans olarak verebilirsin: **Transformer** ekle (Queries panel altında **+ New** → **JS transformer**), adı `filteredUsers`, kodu yukarıdaki gibi. Sonra Table source = `{{ filteredUsers.value }}`, Pie chart source = `{{ filteredUsers.value }}` vb.

#### E) Feature Importance — Horizontal Bar
1. **Chart** ekle, type: Bar
2. **Data source**: `{{ getDashboardData.data.feature_importances }}`
3. **Orientation**: Horizontal
4. **X axis**: `importance`
5. **Y axis**: `feature`

#### F) Scatter — Checkouts vs Payment Likelihood
1. **Chart** ekle, type: Scatter
2. **Data source**: `{{ filteredUsers.value }}` (varsa transformer)
3. **X**: `n_checkout_start`
4. **Y**: `payment_likelihood`
5. **Color by**: `segment`
6. **Size**: `n_payment_success`

#### G) Refresh Butonu
1. **Button** ekle, Label: "Refresh data"
2. **Event handlers** → **+ Add** → **Trigger query** → `getDashboardData`
3. (Opsiyonel) İkinci buton "Retrain model" — POST `/refresh` endpoint'ini çağırır:
   - **Queries** → **+ New** → URL: `/refresh`, Action: POST
   - Button'un onClick'ine bağla

### Adım 2.5: Layout Düzenle

Üstten alta:
1. Header: H2 text "ikas Growth Dashboard" + Refresh butonu (right-aligned)
2. KPI sırası (4 Statistic yan yana, full-width grid)
3. Filtre sırası (5 dropdown yan yana)
4. Grafik sırası 1: Pie (sol) + Bar/feature-importance (sağ)
5. Grafik sırası 2: Scatter (full-width)
6. Table (full-width, bottom)

### Adım 2.6: Test ve Yayınla

1. Sağ üstten **Preview** modunu aç
2. Filtreleri dene, tablonun filtrelendiğini gör
3. CSV export'un çalıştığını test et
4. Hata yoksa **Publish** ile yayınla
5. Retool sana bir share URL verir — ekibinle paylaşabilirsin

---

## Bölüm 3 — Production'a Geçiş (İleride)

Şu anda çalışan şey demo seviyesi. Gerçek production için:

| Şu an | Production |
|---|---|
| ngrok (URL değişir, sayfa uyarısı, free=1 connection) | Sabit domain (ngrok paid / Cloudflare Tunnel / nginx reverse proxy) |
| FastAPI dev server | Gunicorn + uvicorn workers (`gunicorn api:app -k uvicorn.workers.UvicornWorker -w 4`) |
| Local Mac | Docker container → Render/Railway/Fly.io ($5-10/ay) |
| Auth yok | API key header (`X-API-Key`) → `Depends(verify_api_key)` |
| CORS `*` | Sadece Retool subdomain'i |
| HTTPS ngrok ile | Let's Encrypt / Cloudflare otomatik |
| CSV okuma | Postgres (Supabase free tier yeterli) |

API'yi gerçek dünyada deploy etmek istersen şu komut yeter:
```bash
# Procfile (Render/Railway için)
web: gunicorn api:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

---

## Sorun Çözme

**"Connection failed" Retool'da**
- ngrok URL'i sonunda `/` olmamalı
- ngrok terminal'i hala açık mı?
- `curl <ngrok_url>/health` ile dışarıdan da çalışıyor mu?

**ngrok "Visit Site" uyarısı sürekli çıkıyor**
- Bu sadece browser'da, API çağrılarında çıkmaz
- Garanti için Retool'un Resource Headers'ına ekle: `ngrok-skip-browser-warning: true`

**"Network error" Retool'da**
- API çalışıyor mu? (`curl http://127.0.0.1:8000/health`)
- Browser'dan ngrok URL'i açılıyor mu?
- CORS header'ları doğru mu? (`curl -I -X OPTIONS <ngrok_url>/predictions`)

**Tablo boş**
- `getDashboardData` query'sini sağ panel'den manuel **Run** et
- Cevapta `predictions` array'i dolu mu kontrol et

---

## Hızlı Referans — API Endpoint Listesi

| Endpoint | Açıklama | Retool için kullanım |
|---|---|---|
| `GET /health` | Sağlık kontrolü | Bağlantı testi |
| `GET /dashboard-data` | Tüm dashboard verisi tek seferde | Ana query (recommended) |
| `GET /predictions` | Filtrelenebilir kullanıcı listesi | Table source |
| `GET /predictions/{user_id}` | Tek kullanıcı detayı | User detail modal |
| `GET /segments/counts` | Segment başına sayım | KPI cards |
| `GET /segments/summary` | Segment agrega tablosu | Summary chart |
| `GET /feature-importances` | RF top features | Importance bar chart |
| `GET /filters` | Dropdown seçenekleri | Filter dropdowns |
| `GET /metadata` | Model metadata (CV scores, vb.) | Info panel |
| `POST /refresh` | Pipeline'ı yeniden çalıştır (~3 sn) | "Retrain" butonu |

Swagger UI: `<ngrok_url>/docs` — interaktif test için

---

## Bana Sor

Yapamadığın bir adım olursa screenshot at, beraber çözeriz. Özellikle:
- Component bağlantıları (`{{ }}` syntax)
- Transformer JS
- Conditional styling
