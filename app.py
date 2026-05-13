"""
MAN TÜRKİYE STOK TAHMİN DASHBOARD
Model karşılaştırmalı, parça bazlı stok tahmin analizi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAN Türkiye Stok Tahmin",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Renk Paleti ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "XGBoost"     : "#2196F3",
    "LightGBM"    : "#4CAF50",
    "CatBoost"    : "#FF9800",
    "RandomForest": "#9C27B0",
    "Croston"     : "#F44336",
    "SBA"         : "#E91E63",
    "TSB"         : "#795548",
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px; border-radius: 12px;
        color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 13px; opacity: 0.85; margin-bottom: 5px; }
    .metric-value { font-size: 32px; font-weight: 700; }
    .metric-sub   { font-size: 12px; opacity: 0.7; margin-top: 4px; }
    .section-title {
        font-size: 18px; font-weight: 600; color: #1e3c72;
        border-left: 4px solid #2a5298; padding-left: 10px;
        margin: 20px 0 10px 0;
    }
    .model-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Metrik Fonksiyonları ─────────────────────────────────────────────────────
def wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    return (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def accuracy_pct(wape_val):
    return max(0, 100 - wape_val)


# ─── Veri Yükleme ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    preds_path = os.path.join(base, "final_predictions.csv")
    model_path = os.path.join(base, "model_selection_summary.csv")
    inv_path   = os.path.join(base, "inventory_parameters.csv")

    missing = [p for p in [preds_path, model_path] if not os.path.exists(p)]
    if missing:
        return None, None, None

    preds  = pd.read_csv(preds_path, parse_dates=["Tarih"])
    models = pd.read_csv(model_path)
    inv    = pd.read_csv(inv_path) if os.path.exists(inv_path) else pd.DataFrame()

    return preds, models, inv


# ─── Ana Uygulama ─────────────────────────────────────────────────────────────
def main():
    # Başlık
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3c72, #2a5298);
                padding: 25px; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='color:white; margin:0; font-size:28px;'>
            🚛 MAN TÜRKİYE STOK TAHMİN SİSTEMİ
        </h1>
        <p style='color:rgba(255,255,255,0.8); margin:5px 0 0 0; font-size:14px;'>
            Parça bazlı model karşılaştırma & en iyi model seçimi
        </p>
    </div>
    """, unsafe_allow_html=True)

    preds, models, inv = load_data()

    if preds is None:
        st.warning("⚠️ CSV dosyaları bulunamadı. Lütfen Colab notebook'unu çalıştırıp CSV'leri buraya kopyalayın.")
        st.info("Beklenen dosyalar: `final_predictions.csv`, `model_selection_summary.csv`, `inventory_parameters.csv`")

        # Demo modu
        st.markdown("---")
        st.subheader("📊 Demo Modu (Örnek Veriler)")
        preds, models, inv = generate_demo_data()

    # ─── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Filtreler")

        # Parça seçici
        all_parts = sorted(preds["Parça_Kodu"].unique().tolist())
        selected_part = st.selectbox(
            "Parça Seç",
            all_parts,
            index=0,
            help="Detay görünümü için parça seçin"
        )

        st.markdown("---")

        # Segment filtresi
        if "Segment" in preds.columns:
            segments = ["Tümü"] + sorted(preds["Segment"].dropna().unique().tolist())
            sel_segment = st.multiselect("Segment", segments[1:], default=None,
                                          placeholder="Tüm segmentler")
        else:
            sel_segment = []

        # Model filtresi
        all_models = sorted(models["Selected_Model"].unique().tolist())
        sel_models = st.multiselect("Model Filtresi", all_models, default=None,
                                     placeholder="Tüm modeller")

        st.markdown("---")
        st.markdown("### 📥 Veri İndir")
        st.download_button("⬇ Tahminler (CSV)",
                           preds.to_csv(index=False).encode(),
                           "final_predictions.csv", "text/csv")
        st.download_button("⬇ Model Özeti (CSV)",
                           models.to_csv(index=False).encode(),
                           "model_selection_summary.csv", "text/csv")
        if not inv.empty:
            st.download_button("⬇ Stok Parametreleri (CSV)",
                               inv.to_csv(index=False).encode(),
                               "inventory_parameters.csv", "text/csv")

    # ─── Filtre Uygula ────────────────────────────────────────────────────────
    filtered_preds  = preds.copy()
    filtered_models = models.copy()

    if sel_segment:
        filtered_preds  = filtered_preds[filtered_preds["Segment"].isin(sel_segment)]
        filtered_models = filtered_models[filtered_models["Parça_Kodu"].isin(
            filtered_preds["Parça_Kodu"])]

    if sel_models:
        filtered_models = filtered_models[filtered_models["Selected_Model"].isin(sel_models)]
        filtered_preds  = filtered_preds[filtered_preds["Parça_Kodu"].isin(
            filtered_models["Parça_Kodu"])]

    # ─── Metrik Kartlar ───────────────────────────────────────────────────────
    overall_wape = wape(filtered_preds["Talep"], filtered_preds["Tahmin"])
    overall_rmse = rmse(filtered_preds["Talep"], filtered_preds["Tahmin"])
    overall_mae  = mae(filtered_preds["Talep"],  filtered_preds["Tahmin"])
    acc          = accuracy_pct(overall_wape)
    target_color = "#00e676" if overall_wape < 25 else "#ff5252"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>WAPE</div>
            <div class='metric-value' style='color:{target_color};'>%{overall_wape:.1f}</div>
            <div class='metric-sub'>Hedef: &lt;%25</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>RMSE</div>
            <div class='metric-value'>{overall_rmse:.1f}</div>
            <div class='metric-sub'>Root Mean Squared Error</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MAE</div>
            <div class='metric-value'>{overall_mae:.1f}</div>
            <div class='metric-sub'>Mean Absolute Error</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Doğruluk</div>
            <div class='metric-value' style='color:#00e676;'>%{acc:.1f}</div>
            <div class='metric-sub'>{len(filtered_models):,} parça analiz edildi</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Grafik 1 & 2: Model Dağılım ──────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Model Kullanım Analizi</div>",
                unsafe_allow_html=True)

    model_agg = (
        filtered_models
        .groupby("Selected_Model")
        .agg(Seçilme=("Selected_Model", "count"),
             Ort_WAPE=("WAPE", "mean"),
             Min_WAPE=("WAPE", "min"),
             Max_WAPE=("WAPE", "max"))
        .reset_index()
        .sort_values("Seçilme", ascending=False)
    )
    model_agg["Oran"] = model_agg["Seçilme"] / model_agg["Seçilme"].sum() * 100
    model_agg["Renk"] = model_agg["Selected_Model"].map(MODEL_COLORS).fillna("#607D8B")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_pie = px.pie(
            model_agg, values="Seçilme", names="Selected_Model",
            title="Model Kullanım Dağılımı",
            color="Selected_Model",
            color_discrete_map=MODEL_COLORS,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=12)
        fig_pie.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        fig_bar = px.bar(
            model_agg.sort_values("Ort_WAPE"),
            x="Selected_Model", y="Ort_WAPE",
            title="Model Bazlı Ortalama WAPE",
            color="Selected_Model",
            color_discrete_map=MODEL_COLORS,
            text=model_agg.sort_values("Ort_WAPE")["Ort_WAPE"].round(1).astype(str) + "%",
            error_y=model_agg.sort_values("Ort_WAPE")["Max_WAPE"] - model_agg.sort_values("Ort_WAPE")["Ort_WAPE"],
            error_y_minus=model_agg.sort_values("Ort_WAPE")["Ort_WAPE"] - model_agg.sort_values("Ort_WAPE")["Min_WAPE"]
        )
        fig_bar.add_hline(y=25, line_dash="dash", line_color="red",
                          annotation_text="Hedef: %25", annotation_position="top right")
        fig_bar.update_layout(height=380, showlegend=False,
                              yaxis_title="Ortalama WAPE (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Tablo 1: Model Seçim Özeti ───────────────────────────────────────────
    st.markdown("<div class='section-title'>📋 Model Seçim Özeti Tablosu</div>",
                unsafe_allow_html=True)

    display_model_agg = model_agg.copy()
    display_model_agg["Oran"]    = display_model_agg["Oran"].round(1).astype(str) + "%"
    display_model_agg["Ort_WAPE"] = display_model_agg["Ort_WAPE"].round(1).astype(str) + "%"
    display_model_agg["Min_WAPE"] = display_model_agg["Min_WAPE"].round(1).astype(str) + "%"
    display_model_agg["Max_WAPE"] = display_model_agg["Max_WAPE"].round(1).astype(str) + "%"
    display_model_agg = display_model_agg.rename(columns={
        "Selected_Model": "Model",
        "Seçilme": "Seçilme Sayısı",
        "Oran": "Seçilme Oranı",
        "Ort_WAPE": "Ort. WAPE",
        "Min_WAPE": "Min WAPE",
        "Max_WAPE": "Max WAPE"
    })[["Model", "Seçilme Sayısı", "Seçilme Oranı", "Ort. WAPE", "Min WAPE", "Max WAPE"]]

    st.dataframe(display_model_agg, use_container_width=True, hide_index=True)

    # ─── Grafik 3: Parça Detay - Gerçek vs Tahmin ─────────────────────────────
    st.markdown(f"<div class='section-title'>🔍 Parça Detay: {selected_part}</div>",
                unsafe_allow_html=True)

    part_data = filtered_preds[filtered_preds["Parça_Kodu"] == selected_part].sort_values("Tarih")

    if len(part_data) > 0:
        part_model_info = filtered_models[filtered_models["Parça_Kodu"] == selected_part]
        sel_model_name  = part_model_info["Selected_Model"].iloc[0] if len(part_model_info) > 0 else "?"
        part_wape_val   = part_model_info["WAPE"].iloc[0] if len(part_model_info) > 0 else np.nan

        col_i1, col_i2, col_i3 = st.columns(3)
        model_color = MODEL_COLORS.get(sel_model_name, "#607D8B")
        with col_i1:
            st.markdown(f"""
            <div style='background:{model_color};padding:15px;border-radius:10px;color:white;text-align:center;'>
                <b>Seçilen Model</b><br><span style='font-size:22px;font-weight:700;'>{sel_model_name}</span>
            </div>""", unsafe_allow_html=True)
        with col_i2:
            target_ok = "✅" if (not np.isnan(part_wape_val) and part_wape_val < 25) else "❌"
            st.markdown(f"""
            <div style='background:#37474f;padding:15px;border-radius:10px;color:white;text-align:center;'>
                <b>Parça WAPE</b><br><span style='font-size:22px;font-weight:700;'>%{part_wape_val:.1f} {target_ok}</span>
            </div>""", unsafe_allow_html=True)
        with col_i3:
            st.markdown(f"""
            <div style='background:#37474f;padding:15px;border-radius:10px;color:white;text-align:center;'>
                <b>Test Dönemi</b><br><span style='font-size:18px;font-weight:700;'>{len(part_data)} ay</span>
            </div>""", unsafe_allow_html=True)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=part_data["Tarih"], y=part_data["Talep"],
            name="Gerçek Talep", mode="lines+markers",
            line=dict(color="#1976D2", width=2),
            marker=dict(size=6)
        ))
        fig_ts.add_trace(go.Scatter(
            x=part_data["Tarih"], y=part_data["Tahmin"],
            name=f"Tahmin ({sel_model_name})", mode="lines+markers",
            line=dict(color=model_color, width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond")
        ))
        fig_ts.update_layout(
            title=f"{selected_part} — Gerçek vs Tahmin",
            xaxis_title="Tarih", yaxis_title="Talep",
            height=380, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Tablo 3: Seçilen Parça İçin Tüm Model WAPE'leri
        if len(part_model_info) > 0:
            st.markdown("<div class='section-title'>🏆 Parça İçin Model Karşılaştırması</div>",
                        unsafe_allow_html=True)

            wape_cols = [c for c in part_model_info.columns if c.endswith("_WAPE")]
            if wape_cols:
                comp_rows = []
                for wc in wape_cols:
                    model_nm = wc.replace("_WAPE", "")
                    val = part_model_info[wc].iloc[0]
                    if not np.isnan(val):
                        comp_rows.append({
                            "Model"    : model_nm,
                            "WAPE"     : f"%{val:.1f}",
                            "WAPE_raw" : val,
                            "Seçildi"  : "✅ Seçildi" if model_nm == sel_model_name else "❌"
                        })

                if comp_rows:
                    comp_df = pd.DataFrame(comp_rows).sort_values("WAPE_raw")
                    st.dataframe(
                        comp_df[["Model", "WAPE", "Seçildi"]],
                        use_container_width=True, hide_index=True
                    )
    else:
        st.info("Bu parça için test verisi bulunamadı.")

    # ─── Grafik 4: Segment Bazlı WAPE ─────────────────────────────────────────
    if "Segment" in filtered_preds.columns:
        st.markdown("<div class='section-title'>📦 Segment Bazlı Performans</div>",
                    unsafe_allow_html=True)

        seg_stats = []
        for seg, grp in filtered_preds.groupby("Segment"):
            seg_wape = wape(grp["Talep"], grp["Tahmin"])
            seg_rmse = rmse(grp["Talep"], grp["Tahmin"])
            seg_stats.append({
                "Segment": seg,
                "WAPE_%": round(seg_wape, 1),
                "RMSE"  : round(seg_rmse, 1),
                "Parça" : grp["Parça_Kodu"].nunique(),
                "Satır" : len(grp)
            })

        seg_df = pd.DataFrame(seg_stats).sort_values("WAPE_%")

        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            fig_seg = px.bar(
                seg_df, x="Segment", y="WAPE_%",
                title="Segment Bazlı WAPE",
                color="WAPE_%", color_continuous_scale="RdYlGn_r",
                text=seg_df["WAPE_%"].astype(str) + "%"
            )
            fig_seg.add_hline(y=25, line_dash="dash", line_color="red",
                              annotation_text="Hedef %25")
            fig_seg.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_seg, use_container_width=True)

        with col_s2:
            # Tablo 2
            st.markdown("**Segment Performans Tablosu**")
            st.dataframe(seg_df, use_container_width=True, hide_index=True)

    # ─── Stok Parametreleri ───────────────────────────────────────────────────
    if not inv.empty:
        part_inv = inv[inv["Parça_Kodu"] == selected_part]
        if len(part_inv) > 0:
            st.markdown(f"<div class='section-title'>⚙️ Stok Parametreleri: {selected_part}</div>",
                        unsafe_allow_html=True)
            row = part_inv.iloc[0]

            cols = st.columns(4)
            params = [
                ("EOQ", f"{row.get('EOQ', '-'):.0f} adet"),
                ("Güvenlik Stoğu", f"{row.get('Güvenlik_Stoku', '-'):.0f} adet"),
                ("Yeniden Sipariş Noktası", f"{row.get('Yeniden_Sipariş_N', '-'):.0f} adet"),
                ("Lead Time", f"{row.get('Lead_Time_Gun', '-'):.0f} gün"),
            ]
            for col, (lbl, val) in zip(cols, params):
                with col:
                    st.metric(lbl, val)

    # ─── Tablo 4: En Kötü 20 Parça ────────────────────────────────────────────
    st.markdown("<div class='section-title'>⚠️ En Kötü 20 Parça (WAPE'ye Göre)</div>",
                unsafe_allow_html=True)

    worst_20 = (
        filtered_models
        .nlargest(20, "WAPE")
        [["Parça_Kodu", "Selected_Model", "WAPE", "is_intermittent"]]
        .copy()
    )
    worst_20["WAPE"] = worst_20["WAPE"].round(1).astype(str) + "%"
    worst_20["Tip"]  = worst_20["is_intermittent"].map({1: "Aralıklı", 0: "Normal"})
    worst_20 = worst_20.rename(columns={
        "Parça_Kodu"     : "Parça Kodu",
        "Selected_Model" : "Seçilen Model",
        "WAPE"           : "WAPE",
        "Tip"            : "Talep Tipi"
    })[["Parça Kodu", "Seçilen Model", "WAPE", "Talep Tipi"]]

    st.dataframe(worst_20, use_container_width=True, hide_index=True)

    # ─── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#78909c;font-size:12px;'>"
        "🚛 MAN Türkiye Stok Tahmin Sistemi — "
        "XGBoost | LightGBM | CatBoost | RandomForest | Croston | SBA | TSB"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Demo Veri Üreteci (CSV yoksa) ────────────────────────────────────────────
def generate_demo_data():
    np.random.seed(42)
    parts   = [f"P{str(i).zfill(4)}" for i in range(1, 201)]
    months  = pd.date_range("2024-01-01", periods=6, freq="MS")
    models_ = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Croston", "SBA", "TSB"]
    segs    = ["A-X", "A-Y", "B-X", "B-Y", "C-X", "C-Z"]

    rows = []
    for p in parts:
        seg = np.random.choice(segs)
        is_int = 1 if seg.endswith(("Y", "Z")) else 0
        for m in months:
            demand = max(0, int(np.random.poisson(10 if not is_int else 2)))
            noise  = np.random.uniform(0.8, 1.2)
            rows.append({
                "Parça_Kodu": p, "Tarih": m,
                "Talep": demand,
                "Tahmin": max(0, demand * noise),
                "Segment": seg,
                "is_intermittent": is_int
            })

    preds = pd.DataFrame(rows)

    model_rows = []
    for p in parts:
        seg    = preds[preds["Parça_Kodu"] == p]["Segment"].iloc[0]
        is_int = preds[preds["Parça_Kodu"] == p]["is_intermittent"].iloc[0]
        pool   = ["Croston", "SBA", "TSB"] if is_int else ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
        best   = np.random.choice(pool)
        bwape  = np.random.uniform(5, 40)
        row = {
            "Parça_Kodu": p, "Selected_Model": best, "WAPE": bwape,
            "is_intermittent": is_int,
            "XGBoost_WAPE": np.nan, "LightGBM_WAPE": np.nan,
            "CatBoost_WAPE": np.nan, "RandomForest_WAPE": np.nan,
            "Croston_WAPE": np.nan, "SBA_WAPE": np.nan, "TSB_WAPE": np.nan
        }
        for m in pool:
            row[f"{m}_WAPE"] = np.random.uniform(5, 50)
        row[f"{best}_WAPE"] = bwape
        model_rows.append(row)

    models = pd.DataFrame(model_rows)

    inv_rows = [{
        "Parça_Kodu": p,
        "EOQ": round(np.random.uniform(10, 200), 0),
        "Güvenlik_Stoku": round(np.random.uniform(2, 30), 0),
        "Yeniden_Sipariş_N": round(np.random.uniform(5, 50), 0),
        "Lead_Time_Gun": np.random.choice([14, 21, 30, 45, 60])
    } for p in parts]
    inv = pd.DataFrame(inv_rows)

    preds["Segment"] = preds["Parça_Kodu"].map(
        models.set_index("Parça_Kodu")["is_intermittent"].map(
            {0: "A-X", 1: "C-Z"}
        )
    )

    return preds, models, inv


if __name__ == "__main__":
    main()
