"""
MAN TÜRKİYE STOK YÖNETİM DASHBOARD
Amaç: Her parça için R (Yeniden Sipariş Noktası) ve Q (Sipariş Miktarı) optimizasyonu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="MAN Türkiye | Stok Optimizasyonu",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_COLORS = {
    "XGBoost"     : "#2196F3",
    "LightGBM"    : "#4CAF50",
    "CatBoost"    : "#FF9800",
    "RandomForest": "#9C27B0",
    "Croston"     : "#F44336",
    "SBA"         : "#E91E63",
    "TSB"         : "#795548",
}

st.markdown("""
<style>
    .hero-card {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 28px 20px; border-radius: 16px; color: white;
        text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .hero-label { font-size: 13px; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase; }
    .hero-value { font-size: 48px; font-weight: 800; margin: 8px 0; }
    .hero-unit  { font-size: 14px; opacity: 0.75; }
    .hero-sub   { font-size: 12px; opacity: 0.65; margin-top: 6px; }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 18px; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 12px; opacity: 0.8; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .metric-sub   { font-size: 11px; opacity: 0.65; margin-top: 3px; }
    .section-title {
        font-size: 17px; font-weight: 700; color: #1e3c72;
        border-left: 4px solid #2a5298; padding-left: 10px;
        margin: 22px 0 12px 0;
    }
    .rule-box {
        background: #f0f7ff; border: 2px solid #2a5298;
        border-radius: 12px; padding: 18px; margin: 10px 0;
        font-size: 15px;
    }
    .warn-box {
        background: #fff8e1; border: 2px solid #f9a825;
        border-radius: 10px; padding: 14px; margin: 8px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ── Metrik fonksiyonları ──────────────────────────────────────────────────────
def wape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    d = np.sum(np.abs(yt))
    return (np.sum(np.abs(yt - yp)) / d * 100) if d > 0 else np.nan

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae_fn(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


# ── Veri yükleme ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "final_predictions.csv")
    m = os.path.join(base, "model_selection_summary.csv")
    i = os.path.join(base, "inventory_parameters.csv")
    if not os.path.exists(p) or not os.path.exists(m):
        return None, None, None
    preds  = pd.read_csv(p, parse_dates=["Tarih"])
    models = pd.read_csv(m)
    inv    = pd.read_csv(i) if os.path.exists(i) else pd.DataFrame()
    return preds, models, inv


# ── Demo verisi ───────────────────────────────────────────────────────────────
def generate_demo_data():
    np.random.seed(42)
    parts  = [f"P{str(i).zfill(4)}" for i in range(1, 201)]
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    segs   = ["A-X", "A-Y", "B-X", "B-Y", "C-X", "C-Z"]
    rows = []
    for p in parts:
        seg    = np.random.choice(segs)
        is_int = 1 if seg.endswith(("Y", "Z")) else 0
        for m in months:
            d = max(0, int(np.random.poisson(15 if not is_int else 2)))
            rows.append({"Parça_Kodu": p, "Tarih": m, "Talep": d,
                         "Tahmin": max(0, d * np.random.uniform(0.75, 1.25)),
                         "Segment": seg, "is_intermittent": is_int})
    preds = pd.DataFrame(rows)

    model_rows = []
    for p in parts:
        is_int = preds[preds["Parça_Kodu"] == p]["is_intermittent"].iloc[0]
        pool   = ["Croston", "SBA", "TSB"] if is_int else ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
        best   = np.random.choice(pool)
        bw     = np.random.uniform(8, 38)
        row    = {"Parça_Kodu": p, "Selected_Model": best, "WAPE": bw, "is_intermittent": is_int,
                  "XGBoost_WAPE": np.nan, "LightGBM_WAPE": np.nan,
                  "CatBoost_WAPE": np.nan, "RandomForest_WAPE": np.nan,
                  "Croston_WAPE": np.nan, "SBA_WAPE": np.nan, "TSB_WAPE": np.nan}
        for mn in pool:
            row[f"{mn}_WAPE"] = np.random.uniform(8, 55)
        row[f"{best}_WAPE"] = bw
        model_rows.append(row)
    models = pd.DataFrame(model_rows)

    inv_rows = []
    for p in parts:
        md = preds[preds["Parça_Kodu"] == p]["Tahmin"].mean()
        sd = preds[preds["Parça_Kodu"] == p]["Tahmin"].std()
        lt = float(np.random.choice([14, 21, 30, 45, 60]))
        oc = float(np.random.uniform(200, 1000))
        hc = float(np.random.uniform(10, 80))
        eoq = np.sqrt((2 * md * 12 * oc) / (hc * 12)) if md > 0 else 0
        ss  = 1.645 * sd * np.sqrt(lt / 30) if sd > 0 else 0
        rop = md * (lt / 30) + ss
        seg = preds[preds["Parça_Kodu"] == p]["Segment"].iloc[0]
        inv_rows.append({
            "Parça_Kodu": p, "Segment": seg,
            "Ort_Talep_Ay": round(md, 1), "Std_Talep_Ay": round(sd, 1),
            "EOQ": round(eoq, 0), "Güvenlik_Stoku": round(ss, 0),
            "Yeniden_Sipariş_N": round(rop, 0),
            "Lead_Time_Gun": lt, "Birim_Maliyet_TL": round(np.random.uniform(500, 5000), 0),
            "Siparis_Maliyeti": oc, "Elde_Tutma_Maliyet": hc,
            "Model_WAPE": models[models["Parça_Kodu"] == p]["WAPE"].iloc[0]
        })
    inv = pd.DataFrame(inv_rows)
    return preds, models, inv


# ── Ana uygulama ──────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3c72,#2a5298);
                padding:22px 28px;border-radius:14px;margin-bottom:18px;'>
        <h1 style='color:white;margin:0;font-size:26px;'>
            🚛 MAN TÜRKİYE — STOK PARAMETRESİ OPTİMİZASYONU
        </h1>
        <p style='color:rgba(255,255,255,0.75);margin:6px 0 0;font-size:13px;'>
            Her parça için: <b>Ne zaman sipariş ver (R)?</b> &nbsp;|&nbsp;
            <b>Ne kadar sipariş ver (Q)?</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    preds, models, inv = load_data()
    demo_mode = preds is None
    if demo_mode:
        st.info("📊 CSV dosyaları bulunamadı — **Demo Modu** çalışıyor. Notebook'u çalıştırıp CSV'leri ekleyin.")
        preds, models, inv = generate_demo_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Parça Seç")
        if not inv.empty and "Model_WAPE" in inv.columns:
            sort_by = st.radio("Sıralama", ["Parça Kodu", "En Kötü WAPE", "En İyi WAPE"], horizontal=True)
            if sort_by == "En Kötü WAPE":
                sorted_parts = inv.sort_values("Model_WAPE", ascending=False)["Parça_Kodu"].tolist()
            elif sort_by == "En İyi WAPE":
                sorted_parts = inv.sort_values("Model_WAPE", ascending=True)["Parça_Kodu"].tolist()
            else:
                sorted_parts = sorted(inv["Parça_Kodu"].tolist())
        else:
            sorted_parts = sorted(preds["Parça_Kodu"].unique().tolist())

        selected_part = st.selectbox("Parça Kodu", sorted_parts, index=0)

        st.markdown("---")
        st.markdown("### 🏷️ Filtrele")
        if "Segment" in preds.columns:
            segs = sorted(preds["Segment"].dropna().unique().tolist())
            sel_seg = st.multiselect("Segment", segs, placeholder="Tümü")
        else:
            sel_seg = []
        all_models_list = sorted(models["Selected_Model"].unique().tolist())
        sel_models = st.multiselect("Model", all_models_list, placeholder="Tüm modeller")

        st.markdown("---")
        st.markdown("### 📥 İndir")
        if not inv.empty:
            st.download_button("⬇ Stok Parametreleri", inv.to_csv(index=False).encode(),
                               "inventory_parameters.csv", "text/csv")
        st.download_button("⬇ Tahminler", preds.to_csv(index=False).encode(),
                           "final_predictions.csv", "text/csv")
        st.download_button("⬇ Model Özeti", models.to_csv(index=False).encode(),
                           "model_selection_summary.csv", "text/csv")

    # ── Filtre ────────────────────────────────────────────────────────────────
    fp = preds.copy()
    fm = models.copy()
    if sel_seg:
        fp = fp[fp["Segment"].isin(sel_seg)]
        fm = fm[fm["Parça_Kodu"].isin(fp["Parça_Kodu"])]
    if sel_models:
        fm = fm[fm["Selected_Model"].isin(sel_models)]
        fp = fp[fp["Parça_Kodu"].isin(fm["Parça_Kodu"])]

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLÜM 1 — PARÇA STOK PARAMETRESİ (R, Q)
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown(f"<div class='section-title'>⚙️ Stok Kararı: {selected_part}</div>",
                unsafe_allow_html=True)

    part_inv = inv[inv["Parça_Kodu"] == selected_part] if not inv.empty else pd.DataFrame()

    if len(part_inv) > 0:
        row = part_inv.iloc[0]
        R   = row.get("Yeniden_Sipariş_N", 0)
        Q   = row.get("EOQ", 0)
        SS  = row.get("Güvenlik_Stoku", 0)
        LT  = row.get("Lead_Time_Gun", 0)
        avg = row.get("Ort_Talep_Ay", 0)
        std = row.get("Std_Talep_Ay", 0)
        bm  = row.get("Birim_Maliyet_TL", 0)
        oc  = row.get("Siparis_Maliyeti", 0)
        hc  = row.get("Elde_Tutma_Maliyet", 0)

        # Hero kartlar: R ve Q
        col_r, col_q = st.columns(2)
        with col_r:
            st.markdown(f"""
            <div class='hero-card' style='background:linear-gradient(135deg,#1b5e20,#388e3c);'>
                <div class='hero-label'>📦 R — Yeniden Sipariş Noktası</div>
                <div class='hero-value'>{R:.0f}</div>
                <div class='hero-unit'>adet</div>
                <div class='hero-sub'>Stok bu seviyeye düştüğünde sipariş ver</div>
            </div>""", unsafe_allow_html=True)
        with col_q:
            st.markdown(f"""
            <div class='hero-card' style='background:linear-gradient(135deg,#0d47a1,#1565c0);'>
                <div class='hero-label'>🛒 Q — Optimal Sipariş Miktarı (EOQ)</div>
                <div class='hero-value'>{Q:.0f}</div>
                <div class='hero-unit'>adet</div>
                <div class='hero-sub'>Her siparişte bu kadar ısmarlayın</div>
            </div>""", unsafe_allow_html=True)

        # Karar kutusu
        st.markdown(f"""
        <div class='rule-box'>
            📋 <b>Sipariş Kuralı:</b> Stok <b>{R:.0f} adete</b> düştüğünde,
            <b>{Q:.0f} adet</b> sipariş ver.
            Sipariş <b>{LT:.0f} gün</b> sonra gelir.
            Güvenlik stoğu: <b>{SS:.0f} adet</b> (servis seviyesi: %95).
        </div>""", unsafe_allow_html=True)

        # Detay metrikler
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Ort. Aylık Talep</div><div class='metric-value'>{avg:.1f}</div><div class='metric-sub'>adet/ay</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Talep Std. Sapması</div><div class='metric-value'>{std:.1f}</div><div class='metric-sub'>adet/ay</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Lead Time</div><div class='metric-value'>{LT:.0f}</div><div class='metric-sub'>gün</div></div>", unsafe_allow_html=True)
        with c4:
            annual_cost = Q * bm if Q > 0 else 0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Yıllık Sipariş Maliyeti</div><div class='metric-value'>{annual_cost:,.0f}</div><div class='metric-sub'>TL</div></div>", unsafe_allow_html=True)

    else:
        st.warning("Bu parça için stok parametresi bulunamadı.")

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLÜM 2 — TAHMİN MODELİ PERFORMANSI (R ve Q'yu etkiler)
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown(f"<div class='section-title'>🤖 Tahmin Modeli: {selected_part}</div>",
                unsafe_allow_html=True)

    part_preds  = fp[fp["Parça_Kodu"] == selected_part].sort_values("Tarih")
    part_minfo  = fm[fm["Parça_Kodu"] == selected_part]

    if len(part_preds) > 0 and len(part_minfo) > 0:
        sel_model   = part_minfo["Selected_Model"].iloc[0]
        part_wape_v = part_minfo["WAPE"].iloc[0]
        part_rmse_v = rmse(part_preds["Talep"], part_preds["Tahmin"])
        part_mae_v  = mae_fn(part_preds["Talep"], part_preds["Tahmin"])
        is_int      = int(part_minfo["is_intermittent"].iloc[0]) if "is_intermittent" in part_minfo.columns else 0
        model_color = MODEL_COLORS.get(sel_model, "#607D8B")
        ok          = "✅" if part_wape_v < 25 else "⚠️"

        # Model bilgisi + metrikler
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class='metric-card' style='background:linear-gradient(135deg,{model_color},{model_color}cc);'>
                <div class='metric-label'>Seçilen Model</div>
                <div class='metric-value' style='font-size:20px;'>{sel_model}</div>
                <div class='metric-sub'>{'Aralıklı Talep → Croston grubu' if is_int else 'Normal Talep → ML grubu'}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            tc = "#00e676" if part_wape_v < 25 else "#ffab40"
            st.markdown(f"<div class='metric-card'><div class='metric-label'>WAPE {ok}</div><div class='metric-value' style='color:{tc};'>%{part_wape_v:.1f}</div><div class='metric-sub'>Hedef: &lt;%25</div></div>", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE</div><div class='metric-value'>{part_rmse_v:.1f}</div><div class='metric-sub'>adet</div></div>", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MAE</div><div class='metric-value'>{part_mae_v:.1f}</div><div class='metric-sub'>adet</div></div>", unsafe_allow_html=True)

        # Gerçek vs Tahmin grafiği
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=part_preds["Tarih"], y=part_preds["Talep"],
            name="Gerçek Talep", marker_color="rgba(25,118,210,0.6)"
        ))
        fig_ts.add_trace(go.Scatter(
            x=part_preds["Tarih"], y=part_preds["Tahmin"],
            name=f"Tahmin ({sel_model})", mode="lines+markers",
            line=dict(color=model_color, width=2.5),
            marker=dict(size=8, symbol="diamond")
        ))
        if len(part_inv) > 0:
            avg_val = part_inv.iloc[0].get("Ort_Talep_Ay", 0)
            fig_ts.add_hline(y=avg_val, line_dash="dot", line_color="#888",
                             annotation_text=f"Ort. Talep: {avg_val:.1f}")
        fig_ts.update_layout(
            title=f"{selected_part} — Test Dönemi: Gerçek Talep vs Model Tahmini",
            xaxis_title="Tarih", yaxis_title="Talep (adet)",
            height=360, hovermode="x unified", barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Tüm modellerin karşılaştırması
        wape_cols = [c for c in part_minfo.columns if c.endswith("_WAPE")]
        if wape_cols:
            st.markdown("**📊 Bu Parça İçin Tüm Modellerin Test Performansı:**")
            comp_rows = []
            for wc in wape_cols:
                mn  = wc.replace("_WAPE", "")
                val = part_minfo[wc].iloc[0]
                if not (isinstance(val, float) and np.isnan(val)):
                    pool_type = "Aralıklı" if mn in ["Croston","SBA","TSB"] else "ML"
                    comp_rows.append({
                        "Model"    : mn,
                        "Tip"      : pool_type,
                        "WAPE"     : val,
                        "WAPE_str" : f"%{val:.1f}",
                        "Seçildi"  : "✅ KAZANAN" if mn == sel_model else ""
                    })
            if comp_rows:
                comp_df = pd.DataFrame(comp_rows).sort_values("WAPE")
                # Bar chart
                colors = [MODEL_COLORS.get(r["Model"], "#607D8B") for _, r in comp_df.iterrows()]
                fig_comp = go.Figure(go.Bar(
                    x=comp_df["Model"], y=comp_df["WAPE"],
                    marker_color=colors,
                    text=comp_df["WAPE_str"],
                    textposition="outside"
                ))
                fig_comp.add_hline(y=25, line_dash="dash", line_color="red",
                                   annotation_text="Hedef %25")
                # Kazananı vurgula
                winner_idx = comp_df["Model"].tolist().index(sel_model) if sel_model in comp_df["Model"].tolist() else -1
                if winner_idx >= 0:
                    fig_comp.add_annotation(
                        x=sel_model, y=comp_df[comp_df["Model"]==sel_model]["WAPE"].iloc[0],
                        text="⭐ Seçildi", showarrow=True, arrowhead=2,
                        font=dict(size=12, color="white"),
                        bgcolor=model_color, bordercolor=model_color
                    )
                fig_comp.update_layout(
                    title="Model Karşılaştırması — WAPE (düşük = iyi)",
                    yaxis_title="WAPE (%)", height=320, showlegend=False
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Tablo
                disp = comp_df[["Model", "Tip", "WAPE_str", "Seçildi"]].rename(
                    columns={"WAPE_str": "WAPE", "Seçildi": "Durum"})
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLÜM 3 — GENEL MODEL PERFORMANSI
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📊 Genel Model Performansı</div>",
                unsafe_allow_html=True)

    overall_wape = wape(fp["Talep"], fp["Tahmin"])
    overall_rmse = rmse(fp["Talep"], fp["Tahmin"])
    overall_mae  = mae_fn(fp["Talep"], fp["Tahmin"])
    tc           = "#00e676" if overall_wape < 25 else "#ff5252"

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Genel WAPE</div><div class='metric-value' style='color:{tc};'>%{overall_wape:.1f}</div><div class='metric-sub'>Hedef: &lt;%25</div></div>", unsafe_allow_html=True)
    with g2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Genel RMSE</div><div class='metric-value'>{overall_rmse:.1f}</div><div class='metric-sub'>tüm parçalar</div></div>", unsafe_allow_html=True)
    with g3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Genel MAE</div><div class='metric-value'>{overall_mae:.1f}</div><div class='metric-sub'>tüm parçalar</div></div>", unsafe_allow_html=True)
    with g4:
        n_ok = (fm["WAPE"] < 25).sum()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Hedef Altı Parça</div><div class='metric-value'>{n_ok}</div><div class='metric-sub'>/ {len(fm)} parça WAPE&lt;%25</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model seçim dağılımı + WAPE karşılaştırması
    model_agg = (
        fm.groupby("Selected_Model")
        .agg(Secilme=("Selected_Model", "count"),
             Ort_WAPE=("WAPE", "mean"),
             Min_WAPE=("WAPE", "min"),
             Max_WAPE=("WAPE", "max"))
        .reset_index()
    )
    model_agg["Oran"] = model_agg["Secilme"] / model_agg["Secilme"].sum() * 100

    cg1, cg2 = st.columns(2)
    with cg1:
        fig_pie = px.pie(model_agg, values="Secilme", names="Selected_Model",
                         title="Hangi Model Kaç Parçada Kazandı?",
                         color="Selected_Model", color_discrete_map=MODEL_COLORS, hole=0.45)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
        fig_pie.update_layout(height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    with cg2:
        bar_df = model_agg.sort_values("Ort_WAPE")
        fig_bar = px.bar(bar_df, x="Selected_Model", y="Ort_WAPE",
                         title="Kazanan Modellerin Ortalama WAPE'si",
                         color="Selected_Model", color_discrete_map=MODEL_COLORS,
                         text=bar_df["Ort_WAPE"].round(1).astype(str) + "%",
                         error_y=bar_df["Max_WAPE"] - bar_df["Ort_WAPE"],
                         error_y_minus=bar_df["Ort_WAPE"] - bar_df["Min_WAPE"])
        fig_bar.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Hedef %25")
        fig_bar.update_layout(height=360, showlegend=False, yaxis_title="Ort. WAPE (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Model özet tablosu
    disp_agg = model_agg.copy()
    disp_agg["Oran"]    = disp_agg["Oran"].round(1).astype(str) + "%"
    disp_agg["Ort_WAPE"] = disp_agg["Ort_WAPE"].round(1).astype(str) + "%"
    disp_agg["Min_WAPE"] = disp_agg["Min_WAPE"].round(1).astype(str) + "%"
    disp_agg["Max_WAPE"] = disp_agg["Max_WAPE"].round(1).astype(str) + "%"
    disp_agg = disp_agg.rename(columns={"Selected_Model":"Model","Secilme":"Kazanma Sayısı",
                                         "Oran":"Oran","Ort_WAPE":"Ort. WAPE",
                                         "Min_WAPE":"Min WAPE","Max_WAPE":"Max WAPE"})
    st.dataframe(disp_agg[["Model","Kazanma Sayısı","Oran","Ort. WAPE","Min WAPE","Max WAPE"]],
                 use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLÜM 4 — TÜM PARÇALAR STOK PARAMETRESİ TABLOSU
    # ════════════════════════════════════════════════════════════════════════════
    if not inv.empty:
        st.markdown("<div class='section-title'>📋 Tüm Parçalar — Stok Parametreleri</div>",
                    unsafe_allow_html=True)

        # WAPE filtresi
        wape_filter = st.slider("WAPE filtresi (üst sınır)", 0, 100, 100, 5,
                                help="Seçilen değerin altındaki WAPE'li parçaları göster")

        display_inv = inv.copy()
        if "Model_WAPE" in display_inv.columns:
            display_inv = display_inv[display_inv["Model_WAPE"] <= wape_filter]

        # Modeli de ekle
        if "Selected_Model" not in display_inv.columns:
            model_map = fm.set_index("Parça_Kodu")["Selected_Model"].to_dict()
            display_inv["Seçilen_Model"] = display_inv["Parça_Kodu"].map(model_map)

        show_cols = ["Parça_Kodu"]
        for c in ["Segment", "Seçilen_Model", "Model_WAPE"]:
            if c in display_inv.columns:
                show_cols.append(c)
        for c in ["Ort_Talep_Ay", "Std_Talep_Ay", "Lead_Time_Gun",
                  "Güvenlik_Stoku", "Yeniden_Sipariş_N", "EOQ",
                  "Birim_Maliyet_TL"]:
            if c in display_inv.columns:
                show_cols.append(c)

        rename_map = {
            "Parça_Kodu"       : "Parça",
            "Seçilen_Model"    : "Model",
            "Model_WAPE"       : "WAPE (%)",
            "Ort_Talep_Ay"     : "Ort. Talep/Ay",
            "Std_Talep_Ay"     : "Std",
            "Lead_Time_Gun"    : "Lead Time (gün)",
            "Güvenlik_Stoku"   : "Güvenlik Stoğu",
            "Yeniden_Sipariş_N": "R (Sipariş Noktası)",
            "EOQ"              : "Q (Sipariş Miktarı)",
            "Birim_Maliyet_TL" : "Birim Maliyet (TL)"
        }
        disp_table = display_inv[show_cols].rename(columns=rename_map)
        if "WAPE (%)" in disp_table.columns:
            disp_table["WAPE (%)"] = disp_table["WAPE (%)"].round(1)
        st.dataframe(disp_table.sort_values("R (Sipariş Noktası)", ascending=False)
                     if "R (Sipariş Noktası)" in disp_table.columns else disp_table,
                     use_container_width=True, hide_index=True, height=400)

        st.caption(f"Toplam {len(disp_table):,} parça gösteriliyor.")

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLÜM 5 — SEGMENT & EN KÖTÜ PARÇALAR
    # ════════════════════════════════════════════════════════════════════════════
    col_s, col_w = st.columns(2)

    with col_s:
        if "Segment" in fp.columns:
            st.markdown("<div class='section-title'>📦 Segment Bazlı WAPE</div>",
                        unsafe_allow_html=True)
            seg_stats = []
            for seg, grp in fp.groupby("Segment"):
                seg_stats.append({"Segment": seg,
                                   "WAPE_%": round(wape(grp["Talep"], grp["Tahmin"]), 1),
                                   "Parça": grp["Parça_Kodu"].nunique()})
            seg_df = pd.DataFrame(seg_stats).sort_values("WAPE_%")
            fig_seg = px.bar(seg_df, x="Segment", y="WAPE_%",
                             color="WAPE_%", color_continuous_scale="RdYlGn_r",
                             text=seg_df["WAPE_%"].astype(str) + "%",
                             title="Segment Bazlı WAPE")
            fig_seg.add_hline(y=25, line_dash="dash", line_color="red")
            fig_seg.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_seg, use_container_width=True)

    with col_w:
        st.markdown("<div class='section-title'>⚠️ En Kötü 20 Parça</div>",
                    unsafe_allow_html=True)
        worst = fm.nlargest(20, "WAPE")[["Parça_Kodu", "Selected_Model", "WAPE"]].copy()
        worst["WAPE"] = worst["WAPE"].round(1).astype(str) + "%"
        if not inv.empty:
            rq = inv[["Parça_Kodu", "Yeniden_Sipariş_N", "EOQ"]].rename(
                columns={"Yeniden_Sipariş_N": "R", "EOQ": "Q"})
            worst = worst.merge(rq, on="Parça_Kodu", how="left")
            worst["R"] = worst["R"].fillna(0).astype(int)
            worst["Q"] = worst["Q"].fillna(0).astype(int)
        worst = worst.rename(columns={"Parça_Kodu": "Parça", "Selected_Model": "Model"})
        st.dataframe(worst, use_container_width=True, hide_index=True, height=320)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#78909c;font-size:11px;'>"
        "🚛 MAN Türkiye Malzeme Yönetimi — (R,Q) Stok Politikası Optimizasyonu — "
        "XGBoost | LightGBM | CatBoost | RF | Croston | SBA | TSB"
        "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
