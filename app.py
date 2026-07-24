import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, List, Dict
import requests

# --- [ใหม่] ฟังก์ชันโหลดข้อมูลแบบมี Cache และทำ Data Cleaning ---
# @st.cache_data(ttl=3600)
# def fetch_market_data(tickers: List[str], period: str = '2y') -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
#     # 1. โหลดราคาและจัดการ NaN
#     prices = yf.download(tickers, period=period)['Close']
#     if isinstance(prices, pd.Series):
#         prices = prices.to_frame(name=tickers[0])
#     prices = prices.ffill().bfill() # ป้องกันข้อมูลแหว่งจากหุ้นใหม่
    
#     # 2. โหลด Metadata (ทำรอบเดียวเก็บ Cache ไว้เลย)
#     caps = []
#     sectors = []
#     for t in tickers:
#         try:
#             info = yf.Ticker(t).info
#             caps.append(info.get('marketCap', info.get('totalAssets', 1e9)))
#             sectors.append(info.get('sector', 'Financials/Other'))
#         except Exception:
#             caps.append(1e9)
#             sectors.append('Unknown')

#     total_cap = sum(caps) if sum(caps) > 0 else 1e9
#     w_mkt_arr = np.array(caps) / total_cap
    
#     return prices, w_mkt_arr, sectors


@st.cache_data(ttl=3600)
def fetch_market_data(tickers: List[str], period: str = '2y') -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    # 1. สร้าง Session และใส่ Header เพื่อปลอมตัวเป็น Browser ของมนุษย์
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    })

    # 2. โหลดราคาโดยแนบ Session เข้าไปด้วย
    prices = yf.download(tickers, period=period, session=session)['Close']
    
    # ดักจับ Error เผื่อถูกบล็อกระดับขั้นสูงสุด
    if prices.empty or prices.isna().all().all():
        st.error("⚠️ Server ถูกจำกัดการเชื่อมต่อจาก Yahoo Finance ชั่วคราว กรุณารอสักครู่แล้วกดล้าง Cache (Clear cache)")
        st.stop()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.ffill().bfill() 
    
    # 3. โหลด Metadata โดยแนบ Session
    caps = []
    sectors = []
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t, session=session) # <--- แนบ session ตรงนี้ด้วย
            info = ticker_obj.info
            caps.append(info.get('marketCap', info.get('totalAssets', 1e9)))
            sectors.append(info.get('sector', 'Financials/Other'))
        except Exception:
            caps.append(1e9)
            sectors.append('Unknown')

    total_cap = sum(caps) if sum(caps) > 0 else 1e9
    w_mkt_arr = np.array(caps) / total_cap
    
    return prices, w_mkt_arr, sectors


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    # ใช้ utf-8-sig เพื่อให้เปิดใน Excel แล้วสัญลักษณ์/ภาษาไทยไม่เพี้ยน
    return df.to_csv(index=True).encode('utf-8-sig')





class BlackLittermanEngine:
    """Advanced quantitative engine supporting dynamic multi-asset investor views and metadata."""

    @classmethod
    def compute_bl_returns(
        cls, prices: pd.DataFrame, tickers: List[str], lambda_val: float, tau: float, 
        views_df: pd.DataFrame, w_mkt: np.ndarray, rf_rate: float # เพิ่ม rf_rate ตรงนี้
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

        log_returns = np.log(prices / prices.shift(1)).dropna()
        sigma = log_returns.cov() * 252

        # [แก้] บวก rf_rate เพื่อให้กลายเป็น Total Implied Return ป้องกันการหักลบซ้ำซ้อนตอนหา Sharpe
        pi = lambda_val * (sigma @ w_mkt) + rf_rate

        active_views = views_df[views_df["ใส่ความเห็น?"] == True]
        k = len(active_views)

        if k == 0:
            return sigma, pd.Series(pi.values.flatten(), index=tickers), pd.Series(pi.values.flatten(), index=tickers)

        n = len(tickers)
        p = np.zeros((k, n))
        q = np.zeros((k, 1))

        for idx, (_, row) in enumerate(active_views.iterrows()):
            ticker_idx = tickers.index(row["Ticker"])
            p[idx, ticker_idx] = 1.0

            # [แก้] ป้องกันบั๊ก Matrix พัง ถ้า user เลือก Relative แต่ลืมเลือกหุ้นที่จะเทียบ
            if row["ประเภท"] == "Relative":
                if row["เทียบกับ"] != "-" and row["เทียบกับ"] in tickers and row["เทียบกับ"] != row["Ticker"]:
                    target_idx = tickers.index(row["เทียบกับ"])
                    p[idx, target_idx] = -1.0
                else:
                    pass # ถ้าระบุไม่ครบ ให้กลืนไปเป็น Absolute view ชั่วคราว

            q[idx, 0] = row["คาดการณ์ (%)"] / 100.0

        tau_sigma = tau * sigma.values
        omega = np.diag(np.diag(p @ tau_sigma @ p.T))

        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega) if np.linalg.det(omega) != 0 else np.eye(k) * 1e4

        post_cov = np.linalg.inv(inv_tau_sigma + p.T @ inv_omega @ p)
        post_mu = post_cov @ (inv_tau_sigma @ pi.values + p.T @ inv_omega @ q.flatten())

        return (
            sigma,
            pd.Series(pi.values.flatten(), index=tickers),
            pd.Series(post_mu.flatten(), index=tickers)
        )

class PortfolioOptimizer:
    """Convex optimization solver using Scipy SLSQP."""

    @staticmethod
    def max_sharpe(mu: pd.Series, sigma: pd.DataFrame, rf: float) -> Tuple[np.ndarray, bool, str]:
        n = len(mu)

        def objective(weights):
            p_ret = np.sum(mu * weights)
            p_vol = np.sqrt(weights.T @ sigma @ weights)
            excess_ret = p_ret - rf # ตอนนี้ p_ret เป็น Total Return แล้ว ลบ rf ตรงนี้จึงถูกต้อง 100%
            return -excess_ret / p_vol if p_vol > 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        initial_weights = [1.0 / n] * n

        res = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x, res.success, res.message





def main():
    st.set_page_config(layout="wide", page_title="BL Institutional Dashboard")
    st.title("🏛️ Institutional Asset Allocation & Risk Analytics Engine")
    st.caption("Production-Ready Multi-View Black-Litterman Framework with Advanced Risk & Attribution")

    # --- Sidebar Configuration ---
    st.sidebar.header("1. Portfolio Asset Universe")
    tickers_raw = st.sidebar.text_input("ระบุชื่อหุ้น (คั่นด้วยคอมม่า)", "AAPL,MSFT,GOOGL,AMZN,JPM,V")
    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]

    total_capital = st.sidebar.number_input("เงินลงทุนเริ่มต้นปัจจุบัน (Current Capital Valuation)", min_value=1000, value=1000000, step=50000)

    st.sidebar.subheader("Parameters & Risk Constraints")
    rf_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100.0
    lambda_val = st.sidebar.slider("Risk Aversion (λ)", 1.0, 5.0, 2.5)
    tau = st.sidebar.slider("Model Uncertainty (τ)", 0.01, 0.10, 0.05)
    conf_level = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95) / 100.0
    ir_alpha_level = st.sidebar.slider("IR Significance Test Level (%)", 90, 99, 95) / 100.0

    # --- Stress Testing Scenario Configuration ---
    st.sidebar.header("2. Macro Stress Testing Scenarios")
    stress_mode = st.sidebar.selectbox(
        "เลือกโหมด Stress Test (Historical Macro Shocks)",
        ["None / Normal Market Scenario", "1987 Black Monday", "1997 Asian Financial Crisis", 
         "2000 Dot-com Bubble Burst", "2008 Global Financial Crisis (GFC)", "2020 COVID-19 Market Crash", "Custom Stress Scenario"]
    )

    stress_scenarios = {
        "None / Normal Market Scenario": (0.0, 1.0), "1987 Black Monday": (-22.5, 3.0),
        "1997 Asian Financial Crisis": (-18.0, 1.6), "2000 Dot-com Bubble Burst": (-25.0, 1.8),
        "2008 Global Financial Crisis (GFC)": (-35.0, 2.5), "2020 COVID-19 Market Crash": (-15.0, 2.0)
    }

    if stress_mode == "Custom Stress Scenario":
        custom_return_shock = st.sidebar.slider("Custom Expected Return Shock (%)", -50, 0, -20)
        custom_vol_mult = st.sidebar.slider("Custom Volatility Multiplier (x)", 1.0, 4.0, 1.5, step=0.1)
        active_shock = (custom_return_shock, custom_vol_mult)
    else:
        active_shock = stress_scenarios[stress_mode]

    # --- Sidebar Dynamic Table Input ---
    st.sidebar.header("3. Multi-Investor Views Configuration")

    init_views_df = pd.DataFrame({
        "Ticker": tickers, "ใส่ความเห็น?": [False] * len(tickers),
        "ประเภท": ["Absolute"] * len(tickers), "เทียบกับ": ["-"] * len(tickers),
        "คาดการณ์ (%)": [5.0] * len(tickers)
    })

    edited_views = st.sidebar.data_editor(
        init_views_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "ใส่ความเห็น?": st.column_config.CheckboxColumn("Active?"),
            "ประเภท": st.column_config.SelectboxColumn("View Type", options=["Absolute", "Relative"]),
            "เทียบกับ": st.column_config.SelectboxColumn("Target (Relative)", options=["-"] + tickers),
            "คาดการณ์ (%)": st.column_config.NumberColumn("Forecast (%)", min_value=-100.0, max_value=100.0, format="%.1f%%")
        },
        hide_index=True, use_container_width=True
    )

    if st.sidebar.button("Execute Asset Allocation & Stress Test"):
        with st.status("Fetching market metrics and processing matrices...", expanded=True) as status:
            
            # [แก้] เรียกใช้ Cache Function ตรงนี้ เร็วกว่าเดิมมาก
            prices, w_mkt_arr, sectors = fetch_market_data(tickers)
            w_mkt = pd.Series(w_mkt_arr, index=tickers)

            # [แก้] ส่ง rf_rate เข้าไปด้วยตามที่แก้สมการไว้
            sigma, pi, mu_bl = BlackLittermanEngine.compute_bl_returns(
                prices, tickers, lambda_val, tau, edited_views, w_mkt_arr, rf_rate
            )

            w_opt, success, msg = PortfolioOptimizer.max_sharpe(mu_bl, sigma, rf_rate)
            status.update(label="Optimization Engine Complete. Analyzing Stress Regime...", state="complete")

        if not success:
            st.error(f"Optimization Engine failed to converge: {msg}")
            return

        p_ret = np.sum(mu_bl * w_opt)
        p_vol = np.sqrt(w_opt.T @ sigma @ w_opt)
        bmk_ret = np.sum(w_mkt * pi)
        bmk_vol = np.sqrt(w_mkt.T @ sigma @ w_mkt)

        sharpe_ratio = (p_ret - rf_rate) / p_vol if p_vol > 0 else 0
        bmk_sharpe = (bmk_ret - rf_rate) / bmk_vol if bmk_vol > 0 else 0

        active_weights = w_opt - w_mkt.values
        te = np.sqrt(active_weights.T @ sigma @ active_weights)
        active_return = p_ret - bmk_ret
        information_ratio = active_return / te if te > 1e-6 else 0

        # [แก้] คำนวณ t-stat จากระยะเวลาจริงที่โหลดมาได้ (Dynamic n years)
        trading_days = len(prices)
        years_of_data = trading_days / 252.0 if trading_days > 0 else 1.0
        t_stat = information_ratio * np.sqrt(years_of_data)
        
        dynamic_ir_z = norm.ppf(1 - (1 - ir_alpha_level) / 2)
        skill_eval = f"Skill 🧠 (at {ir_alpha_level:.0%})" if t_stat > dynamic_ir_z else f"Luck 🍀 (at {ir_alpha_level:.0%})"

        z_score = norm.ppf(conf_level)
        pdf_z = norm.pdf(z_score)
        cvar_mult = pdf_z / (1 - conf_level)
        var_p = z_score * p_vol - p_ret
        cvar_p = cvar_mult * p_vol - p_ret
        relative_var = (z_score * te) - active_return

        # [แก้] เปลี่ยน Hardcode 0.012 เป็น 0.0 เพื่อความแม่นยำของโมเดล
        fx_returns = pd.Series([0.0] * len(tickers), index=tickers)
        currency_effect = w_opt * fx_returns
        total_currency_effect = np.sum(currency_effect)

        shock_pct, vol_multiplier = active_shock
        stressed_p_ret = p_ret + (shock_pct / 100.0)
        stressed_p_vol = p_vol * vol_multiplier
        stressed_var = (z_score * stressed_p_vol) - stressed_p_ret

        expected_ending_val = total_capital * (1 + p_ret)
        stressed_ending_val = total_capital * (1 + stressed_p_ret)
        asset_capital_values = w_opt * total_capital



        # =========================================================
        # --- Main Layout: Performance Analytics UI ---
        # =========================================================
        st.markdown("### 📊 Portfolio Metrics Dashboard")

        st.markdown("#### 💰 Capital Valuation Overview")
        v1, v2, v3 = st.columns(3)
        v1.metric("Current Portfolio Value (Initial)", f"${total_capital:,.2f}")
        v2.metric("Expected Ending Value", f"${expected_ending_val:,.2f}", f"Net Gain: ${(expected_ending_val - total_capital):+,.2f}")
        v3.metric("Asset Capital Value (Check Sum)", f"${np.sum(asset_capital_values):,.2f}", "Must equal Initial Capital")

        st.write("---")

        st.markdown("#### 📈 Core Performance Analytics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Optimized Expected Return", f"{p_ret:.2%}", f"Active Return: {active_return:+.2%}")
        k2.metric("Portfolio Volatility", f"{p_vol:.2%}", f"Bmk Vol: {bmk_vol:.2%}", delta_color="inverse")
        sharpe_delta = sharpe_ratio - bmk_sharpe
        k3.metric("Sharpe Ratio (Ex-Ante)", f"{sharpe_ratio:.2f}", f"vs Bmk: {bmk_sharpe:.2f} (Δ: {sharpe_delta:+.2f})", delta_color="normal" if sharpe_delta >= 0 else "inverse")
        k4.metric("Information Ratio (IR)", f"{information_ratio:.2f}", skill_eval)

        st.markdown("#### 🛡️ Advanced Risk & Currency Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Tracking Error (Ex-Ante TE)", f"{te:.2%}")
        r2.metric("Portfolio CVaR", f"{cvar_p:.2%}")
        r3.metric("Relative VaR (vs Benchmark)", f"{relative_var:.2%}")
        r4.metric("Total Currency Effect", f"{total_currency_effect:.2%}")

        if stress_mode != "None / Normal Market Scenario":
            st.markdown(f"### ⚠️ Risk Analytics Forward Stress Regime: <span style='color:#e74c3c;'>{stress_mode}</span>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Stressed Expected Return", f"{stressed_p_ret:.2%}", f"Delta: {shock_pct:+.1f}%")
            s2.metric("Stressed Volatility", f"{stressed_p_vol:.2%}", f"Multiplier: {vol_multiplier}x", delta_color="inverse")
            s3.metric("Stressed Value-at-Risk", f"{stressed_var:.2%}", " regime-shifted limit", delta_color="inverse")
            s4.metric("Stressed Portfolio Value", f"${stressed_ending_val:,.2f}", f"Loss Impact: ${stressed_ending_val - total_capital:,.2f}", delta_color="inverse")

        col_matrix_title, col_matrix_btn = st.columns([4, 1])
        with col_matrix_title:
         st.markdown("### 💸 Asset Allocation Matrix & Brinson Attribution")

        allocation_effect = active_weights * (pi.values - bmk_ret)
        selection_effect = w_mkt.values * (mu_bl.values - pi.values)
        interaction_effect = active_weights * (mu_bl.values - pi.values)

        summary_df = pd.DataFrame(index=tickers)
        summary_df["Sector / Industry"] = sectors
        summary_df["Market Weight (Bmk)"] = w_mkt
        summary_df["Optimal Weight (BL)"] = w_opt
        summary_df["Active Weight"] = active_weights
        summary_df["Adjusted Return (μ̂)"] = mu_bl
        summary_df["Alloc. Effect"] = allocation_effect
        summary_df["Select. Effect"] = selection_effect
        summary_df["Interact. Effect"] = interaction_effect
        summary_df["Currency Effect"] = currency_effect
        summary_df["Asset Capital Value"] = asset_capital_values

        with col_matrix_btn:
          st.write("") 
          csv_data = convert_df_to_csv(summary_df)
          st.download_button(
          label="📥 Download (CSV)",
          data=csv_data,
          file_name="asset_allocation_matrix.csv",
          mime="text/csv",
          use_container_width=True
    )

        st.dataframe(summary_df.style.format({
            "Market Weight (Bmk)": "{:.2%}", "Optimal Weight (BL)": "{:.2%}", "Active Weight": "{:.2%}",
            "Adjusted Return (μ̂)": "{:.2%}", "Alloc. Effect": "{:.2%}",
            "Select. Effect": "{:.2%}", "Interact. Effect": "{:.2%}",
            "Currency Effect": "{:.2%}", "Asset Capital Value": "${:,.2f}"
        }).background_gradient(subset=["Alloc. Effect", "Select. Effect", "Active Weight"], cmap="RdYlGn"), use_container_width=True)

        st.markdown("### 📈 Sector & Allocation Visual Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            sector_df = summary_df.groupby("Sector / Industry")["Optimal Weight (BL)"].sum().reset_index()
            fig_pie_sec = px.pie(sector_df, values="Optimal Weight (BL)", names="Sector / Industry", hole=0.4, title="Dynamic Sector Exposure Allocation")
            st.plotly_chart(fig_pie_sec, use_container_width=True)
        with c2:
            fig_pie_ind = px.pie(summary_df, values="Optimal Weight (BL)", names=summary_df.index, hole=0.4, title="Individual Stock Weight Allocation")
            st.plotly_chart(fig_pie_ind, use_container_width=True)

        st.markdown("#### 📊 Active Weight Shifts Representation")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=tickers, y=w_mkt*100, name="Market Bench Weight", marker_color="#34495e"))
        fig_bar.add_trace(go.Bar(x=tickers, y=w_opt*100, name="Optimized Target Weight", marker_color="#e74c3c"))
        fig_bar.update_layout(title="Active Weight Shifts: Benchmark vs BL Optimized", barmode="group", yaxis_title="% Weight Allocation")
        st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()


