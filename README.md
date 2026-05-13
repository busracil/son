# 🚛 MAN TÜRKİYE STOK TAHMİN PROJESİ

Parça bazlı en iyi model seçimi ile stok talebi tahmini.

## 📊 Veri Yapısı

| Özellik | Değer |
|---|---|
| Toplam satır | 76,860 |
| Benzersiz parça | 2,135 |
| Dönem | 2022-01 / 2024-12 (36 ay) |
| Sıfır talep oranı | %38.7 |
| Aralıklı talep | %52.4 |

## 🎯 Model Stratejisi

Her parça için `is_intermittent` değerine göre model havuzu belirlenir:

| Talep Tipi | Model Havuzu |
|---|---|
| Aralıklı (`is_intermittent=1`) | Croston, SBA, TSB |
| Normal (`is_intermittent=0`) | XGBoost, LightGBM, CatBoost, RandomForest |

**Akış:** Tüm modeller eğitilir → Test WAPE hesaplanır → En düşük WAPE'li model seçilir.

## ⚙️ Hiperparametreler

```
XGBoost     : max_depth=5, learning_rate=0.05, n_estimators=150, subsample=0.8
LightGBM    : max_depth=5, learning_rate=0.05, n_estimators=150, subsample=0.8
CatBoost    : depth=5, learning_rate=0.05, iterations=150
RandomForest: max_depth=8, n_estimators=100
Croston/SBA/TSB: alpha=0.1
```

**Sample Weights:** 2024→5.0x, 2023→2.0x, 2022→1.0x

**Hedef değişken:** `Talep_log` (log1p) → tahmin sonrası `expm1()` ile geri dönüşüm

## 📁 Dosyalar

| Dosya | Açıklama |
|---|---|
| `man_stock_forecast.ipynb` | Ana Colab notebook (12 hücre) |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Python bağımlılıkları |
| `final_predictions.csv` | Tahmin sonuçları (notebook tarafından üretilir) |
| `model_selection_summary.csv` | Parça bazlı model seçim özeti |
| `inventory_parameters.csv` | (s,Q) stok parametreleri |

## 🚀 Kullanım

### Colab Notebook

1. `man_stock_forecast.ipynb` dosyasını Google Colab'a yükleyin
2. Hücre 2'de `MAN_ML_Dataset_v4.xlsx` dosyasını yükleyin
3. Tüm hücreleri sırayla çalıştırın
4. Hücre 12'de Streamlit dashboard'u başlatın

### Streamlit Dashboard (Lokal)

```bash
pip install -r requirements.txt
streamlit run app.py
```

CSV dosyaları (`final_predictions.csv`, `model_selection_summary.csv`, `inventory_parameters.csv`) aynı dizinde olmalıdır. Yoksa dashboard demo modunda açılır.

## 📈 Beklenen Çıktı

```
MODEL KULLANIM İSTATİSTİKLERİ:
LightGBM    : ~410 parça (%19.2) - Ort. WAPE: %20.8  ⭐ EN ÇOK SEÇİLEN
XGBoost     : ~320 parça (%15.0) - Ort. WAPE: %21.3
Croston     : ~650 parça (%30.4) - Ort. WAPE: %28.5
...

GENEL PERFORMANS:
WAPE: %22.15  ✅ (Hedef: <%25)
RMSE: 198.45
MAE : 107.23
```

## 🗂️ Dashboard Özellikleri

- **4 Metrik Kart:** WAPE, RMSE, MAE, Doğruluk
- **Pie Chart:** Model kullanım dağılımı
- **Bar Chart:** Model bazlı ortalama WAPE (hedef çizgisi ile)
- **Zaman Serisi:** Parça bazlı Gerçek vs Tahmin
- **Segment WAPE:** Segment bazlı performans
- **Model Karşılaştırma Tablosu:** Seçilen parça için tüm modellerin WAPE'leri
- **En Kötü 20 Parça:** İyileştirme odaklı analiz
- **Stok Parametreleri:** EOQ, güvenlik stoğu, yeniden sipariş noktası
