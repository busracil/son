"""
MAN TÜRKİYE — STOK PARAMETRESİ OPTİMİZASYONU
Amaç: Malzeme yönetimi ekibine parça bazlı R ve Q değerlerini göstermek.
R = Yeniden Sipariş Noktası (stok bu seviyeye düştüğünde sipariş ver)
Q = Sipariş Miktarı (her siparişte bu kadar ısmarla)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="MAN Türkiye | Stok Yönetimi",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .hero-r {
        background: linear-gradient(135deg, #1b5e20, #43a047);
        padding: 32px 20px; border-radius: 18px; color: white;
        text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .hero-q {
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        padding: 32px 20px; border-radius: 18px; color: white;
        text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .hero-label { font-size: 13px; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .hero-value { font-size: 64px; font-weight: 900; line-height: 1; }
    .hero-unit  { font-size: 16px; opacity: 0.8; margin-top: 4px; }
    .hero-desc  { font-size: 13px; opacity: 0.7; margin-top: 10px; }
    .rule-box {
        background: #e8f5e9; border-left: 6px solid #2e7d32;
        border-radius: 10px; padding: 18px 22px; margin: 16px 0;
        font-size: 16px; color: #1b5e20;
    }
    .info-card {
        background: #f5f5f5; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .info-label { font-size: 12px; color: #666; margin-bottom: 4px; }
    .info-value { font-size: 24px; font-weight: 700; color: #1e3c72; }
    .info-sub   { font-size: 11px; color: #888; margin-top: 2px; }
    .section-title {
        font-size: 17px; font-weight: 700; color: #1e3c72;
        border-left: 4px solid #2a5298; padding-left: 10px;
        margin: 28px 0 14px 0;
    }
    .wape-ok  { color: #2e7d32; font-weight: 700; }
    .wape-bad { color: #c62828; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Yardımcı ─────────────────────────────────────────────────────────────────
def wape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    d = np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt - yp)) / d * 100) if d > 0 else np.nan


# ── Veri yükleme ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(base, "final_predictions.csv")
    ip = os.path.join(base, "inventory_parameters.csv")
    mp = os.path.join(base, "model_selection_summary.csv")
    if not os.path.exists(ip):
        return None, None, None
    inv    = pd.read_csv(ip)
    preds  = pd.read_csv(fp, parse_dates=["Tarih"]) if os.path.exists(fp) else pd.DataFrame()
    models = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame()
    return inv, preds, models


# ── Demo ──────────────────────────────────────────────────────────────────────
def generate_demo():
    np.random.seed(42)
    parts = [f"P{str(i).zfill(4)}" for i in range(1, 301)]
    segs  = ["A-X", "A-Y", "B-X", "B-Y", "C-X", "C-Z"]
    rows  = []
    for p in parts:
        seg = np.random.choice(segs)
        avg = np.random.uniform(2, 120)
        std = avg * np.random.uniform(0.2, 0.7)
        lt  = float(np.random.choice([14, 21, 30, 45, 60]))
        oc  = float(np.random.uniform(200, 1200))
        hc  = float(np.random.uniform(10, 80))
        bm  = float(np.random.uniform(300, 8000))
        eoq = np.sqrt((2 * avg * 12 * oc) / (hc * 12)) if avg > 0 else 0
        ss  = 1.645 * std * np.sqrt(lt / 30)
        rop = avg * (lt / 30) + ss
        pw  = np.random.uniform(5, 45)
        rows.append({
            "Parça_Kodu": p, "Segment": seg,
            "Ort_Talep_Ay": round(avg, 1), "Std_Talep_Ay": round(std, 1),
            "Lead_Time_Gun": lt, "Güvenlik_Stoku": round(ss, 0),
            "Yeniden_Sipariş_N": round(rop, 0), "EOQ": round(eoq, 0),
            "Birim_Maliyet_TL": round(bm, 0),
            "Siparis_Maliyeti": oc, "Elde_Tutma_Maliyet": hc,
            "Model_WAPE": round(pw, 1),
            "Selected_Model": np.random.choice(
                ["LightGBM","XGBoost","CatBoost","Croston","SBA"])
        })
    inv = pd.DataFrame(rows)

    # Gerçek vs tahmin verisi
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    pred_rows = []
    for p in parts[:50]:
        avg = inv[inv["Parça_Kodu"] == p]["Ort_Talep_Ay"].iloc[0]
        for m in months:
            d = max(0, int(np.random.poisson(avg)))
            pred_rows.append({"Parça_Kodu": p, "Tarih": m,
                              "Talep": d, "Tahmin": max(0, d * np.random.uniform(0.78, 1.22))})
    preds = pd.DataFrame(pred_rows)
    return inv, preds, pd.DataFrame()


# ── Ana uygulama ──────────────────────────────────────────────────────────────
def main():
    # Başlık
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3c72,#2a5298);
                padding:20px 28px;border-radius:14px;margin-bottom:20px;'>
        <h1 style='color:white;margin:0;font-size:24px;'>
            🚛 MAN TÜRKİYE — STOK YÖNETİMİ
        </h1>
        <p style='color:rgba(255,255,255,0.75);margin:5px 0 0;font-size:13px;'>
            Her parça için optimal <b>sipariş noktası (R)</b> ve <b>sipariş miktarı (Q)</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    inv, preds, models = load_data()
    if inv is None:
        st.info("📊 Veriler bulunamadı — Demo modu aktif. Notebook'u çalıştırıp CSV'leri ekleyin.")
        inv, preds, models = generate_demo()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Parça Seç")
        sort_opt = st.radio("Sıralama", ["A→Z", "R (büyük→küçük)", "WAPE (kötü→iyi)"],
                            horizontal=False)
        if sort_opt == "R (büyük→küçük)" and "Yeniden_Sipariş_N" in inv.columns:
            part_list = inv.sort_values("Yeniden_Sipariş_N", ascending=False)["Parça_Kodu"].tolist()
        elif sort_opt == "WAPE (kötü→iyi)" and "Model_WAPE" in inv.columns:
            part_list = inv.sort_values("Model_WAPE", ascending=False)["Parça_Kodu"].tolist()
        else:
            part_list = sorted(inv["Parça_Kodu"].tolist())

        selected = st.selectbox("Parça Kodu", part_list)

        st.markdown("---")

        if "Segment" in inv.columns:
            segs = sorted(inv["Segment"].dropna().unique().tolist())
            sel_seg = st.multiselect("Segment Filtresi", segs, placeholder="Tümü")
        else:
            sel_seg = []

        wape_max = st.slider("Maks. WAPE (%)", 0, 100, 100, 5) if "Model_WAPE" in inv.columns else 100

        st.markdown("---")
        st.markdown("### 📥 İndir")
        st.download_button("⬇ Stok Parametreleri (CSV)",
                           inv.to_csv(index=False).encode(),
                           "inventory_parameters.csv", "text/csv")
        if not preds.empty:
            st.download_button("⬇ Tahmin Sonuçları (CSV)",
                               preds.to_csv(index=False).encode(),
                               "final_predictions.csv", "text/csv")

    # ── Filtre uygula ─────────────────────────────────────────────────────────
    fi = inv.copy()
    if sel_seg:
        fi = fi[fi["Segment"].isin(sel_seg)]
    if "Model_WAPE" in fi.columns:
        fi = fi[fi["Model_WAPE"] <= wape_max]

    # ════════════════════════════════════════════════════════════════════════
    # BÖLÜM 1 — SEÇİLEN PARÇA: R ve Q
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(f"<div class='section-title'>📦 Sipariş Kararı: {selected}</div>",
                unsafe_allow_html=True)

    row = inv[inv["Parça_Kodu"] == selected]
    if len(row) == 0:
        st.warning("Bu parça için veri bulunamadı.")
        return

    row = row.iloc[0]
    R   = row.get("Yeniden_Sipariş_N", 0)
    Q   = row.get("EOQ", 0)
    SS  = row.get("Güvenlik_Stoku", 0)
    LT  = row.get("Lead_Time_Gun", 0)
    avg = row.get("Ort_Talep_Ay", 0)
    std = row.get("Std_Talep_Ay", 0)
    bm  = row.get("Birim_Maliyet_TL", 0)
    oc  = row.get("Siparis_Maliyeti", 0)
    hc  = row.get("Elde_Tutma_Maliyet", 0)
    pw  = row.get("Model_WAPE", np.nan)
    seg = row.get("Segment", "-")
    sel_model = row.get("Selected_Model", "—")

    # Hero: R ve Q
    col_r, col_q = st.columns(2)
    with col_r:
        st.markdown(f"""
        <div class='hero-r'>
            <div class='hero-label'>📍 R — Yeniden Sipariş Noktası</div>
            <div class='hero-value'>{int(R)}</div>
            <div class='hero-unit'>adet</div>
            <div class='hero-desc'>Stok bu seviyeye düştüğünde sipariş ver</div>
        </div>""", unsafe_allow_html=True)
    with col_q:
        st.markdown(f"""
        <div class='hero-q'>
            <div class='hero-label'>🛒 Q — Sipariş Miktarı (EOQ)</div>
            <div class='hero-value'>{int(Q)}</div>
            <div class='hero-unit'>adet</div>
            <div class='hero-desc'>Her siparişte bu kadar ısmarla</div>
        </div>""", unsafe_allow_html=True)

    # Açık kural kutusu
    lt_text = f"{int(LT)} gün" if LT > 0 else "—"
    st.markdown(f"""
    <div class='rule-box'>
        <b>📋 Sipariş Kuralı:</b><br>
        Stok <b>{int(R)} adete</b> düştüğünde, <b>{int(Q)} adet</b> sipariş verin.<br>
        Sipariş {lt_text} sonra gelir. &nbsp;|&nbsp;
        Güvenlik stoğu: <b>{int(SS)} adet</b> &nbsp;(%95 servis seviyesi)
    </div>
    """, unsafe_allow_html=True)

    # Detay kartlar
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Ort. Talep/Ay", f"{avg:.1f} adet", ""),
        ("Talep Std Sap.", f"{std:.1f} adet", ""),
        ("Lead Time", f"{int(LT)} gün", ""),
        ("Birim Maliyet", f"{bm:,.0f} TL", ""),
        ("Tahmin WAPE", f"%{pw:.1f}" if not np.isnan(pw) else "—",
         "wape-ok" if (not np.isnan(pw) and pw < 25) else "wape-bad"),
    ]
    for col, (lbl, val, cls) in zip([c1, c2, c3, c4, c5], cards):
        with col:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>{lbl}</div>
                <div class='info-value {cls}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # Gerçek vs Tahmin grafiği (varsa)
    if not preds.empty and selected in preds["Parça_Kodu"].values:
        st.markdown("<br>", unsafe_allow_html=True)
        part_preds = preds[preds["Parça_Kodu"] == selected].sort_values("Tarih")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=part_preds["Tarih"], y=part_preds["Talep"],
                             name="Gerçek Talep", marker_color="rgba(25,118,210,0.55)"))
        fig.add_trace(go.Scatter(x=part_preds["Tarih"], y=part_preds["Tahmin"],
                                 name=f"Model Tahmini", mode="lines+markers",
                                 line=dict(color="#e65100", width=2.5),
                                 marker=dict(size=8, symbol="diamond")))
        fig.add_hline(y=R, line_dash="dash", line_color="#2e7d32",
                      annotation_text=f"R={int(R)}", annotation_position="right")
        fig.update_layout(
            title=f"{selected} — Gerçek Talep vs Tahmin (Test Dönemi)",
            xaxis_title="Tarih", yaxis_title="Adet",
            height=340, hovermode="x unified", barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # BÖLÜM 2 — GENEL ÖZET METRİKLER
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📊 Genel Durum</div>", unsafe_allow_html=True)

    total    = len(fi)
    ok_count = int((fi["Model_WAPE"] < 25).sum()) if "Model_WAPE" in fi.columns else "-"
    avg_wape = fi["Model_WAPE"].mean() if "Model_WAPE" in fi.columns else np.nan
    avg_R    = fi["Yeniden_Sipariş_N"].mean() if "Yeniden_Sipariş_N" in fi.columns else 0
    avg_Q    = fi["EOQ"].mean() if "EOQ" in fi.columns else 0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sum_cards = [
        ("Toplam Parça", f"{total:,}", ""),
        ("Hedef Tutan Parça\n(WAPE<%25)", f"{ok_count}", "wape-ok"),
        ("Ort. WAPE", f"%{avg_wape:.1f}" if not np.isnan(avg_wape) else "—",
         "wape-ok" if (not np.isnan(avg_wape) and avg_wape < 25) else "wape-bad"),
        ("Ort. R", f"{avg_R:.0f} adet", ""),
        ("Ort. Q", f"{avg_Q:.0f} adet", ""),
    ]
    for col, (lbl, val, cls) in zip([sc1, sc2, sc3, sc4, sc5], sum_cards):
        with col:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>{lbl}</div>
                <div class='info-value {cls}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # WAPE dağılımı histogramı
    if "Model_WAPE" in fi.columns:
        cw1, cw2 = st.columns([2, 1])
        with cw1:
            fig_hist = px.histogram(fi, x="Model_WAPE", nbins=30,
                                    title="Tahmin Doğruluğu Dağılımı (WAPE)",
                                    color_discrete_sequence=["#1976d2"],
                                    labels={"Model_WAPE": "WAPE (%)", "count": "Parça Sayısı"})
            fig_hist.add_vline(x=25, line_dash="dash", line_color="red",
                               annotation_text="Hedef %25", annotation_position="top right")
            fig_hist.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with cw2:
            if "Segment" in fi.columns:
                seg_wape = (fi.groupby("Segment")["Model_WAPE"]
                            .mean().reset_index()
                            .rename(columns={"Model_WAPE": "Ort_WAPE"})
                            .sort_values("Ort_WAPE"))
                fig_seg = px.bar(seg_wape, x="Segment", y="Ort_WAPE",
                                 title="Segment Bazlı Ort. WAPE",
                                 color="Ort_WAPE", color_continuous_scale="RdYlGn_r",
                                 text=seg_wape["Ort_WAPE"].round(1).astype(str) + "%")
                fig_seg.add_hline(y=25, line_dash="dash", line_color="red")
                fig_seg.update_layout(height=300, showlegend=False,
                                      yaxis_title="Ort. WAPE (%)")
                st.plotly_chart(fig_seg, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # BÖLÜM 3 — TÜM PARÇALAR TABLOSU
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📋 Tüm Parçalar — Sipariş Parametreleri</div>",
                unsafe_allow_html=True)

    show_cols = ["Parça_Kodu"]
    col_labels = {"Parça_Kodu": "Parça"}
    for c, lbl in [("Segment", "Segment"),
                   ("Selected_Model", "Model"),
                   ("Model_WAPE", "WAPE (%)"),
                   ("Ort_Talep_Ay", "Ort. Talep/Ay"),
                   ("Lead_Time_Gun", "Lead Time (gün)"),
                   ("Güvenlik_Stoku", "Güvenlik Stoğu"),
                   ("Yeniden_Sipariş_N", "R — Sipariş Noktası"),
                   ("EOQ", "Q — Sipariş Miktarı"),
                   ("Birim_Maliyet_TL", "Birim Maliyet (TL)")]:
        if c in fi.columns:
            show_cols.append(c)
            col_labels[c] = lbl

    disp = fi[show_cols].rename(columns=col_labels).copy()
    if "WAPE (%)" in disp.columns:
        disp["WAPE (%)"] = disp["WAPE (%)"].round(1)
    if "R — Sipariş Noktası" in disp.columns:
        disp = disp.sort_values("R — Sipariş Noktası", ascending=False)

    st.dataframe(disp, use_container_width=True, hide_index=True, height=420)
    st.caption(f"{len(disp):,} parça listeleniyor.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#90a4ae;font-size:11px;'>"
        "🚛 MAN Türkiye Malzeme Yönetimi &nbsp;|&nbsp; "
        "(R, Q) Stok Politikası &nbsp;|&nbsp; %95 Servis Seviyesi"
        "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
