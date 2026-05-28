"""
==============================================================================
 APROKSIMASI PERILAKU FUNGSI TEKANAN DENGAN POLINOM TAYLOR
 Kalkulus Teknik Perminyakan — Streamlit Interactive App
==============================================================================

STRUKTUR KODE:
  1. compute_pressure()    — Nilai eksak P(z) = P0 * exp(-alpha * z)
  2. taylor_order2()       — Polinom Taylor Orde-2 di sekitar z = a
  3. taylor_order3()       — Polinom Taylor Orde-3 di sekitar z = a
  4. relative_error()      — Galat relatif (%) antara eksak dan aproksimasi
  5. build_dataframe()     — DataFrame ringkasan pada kedalaman tertentu
  6. plot_pressure()       — Plot perbandingan P(z) eksak vs Taylor O-2 vs O-3
  7. plot_error()          — Plot galat relatif (%) dengan skala Y logaritmik
  8. main()                — Entry point; sidebar input + layout Streamlit
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNGSI MATEMATIS INTI
# ─────────────────────────────────────────────────────────────────────────────

def compute_pressure(P0: float, alpha: float, z: np.ndarray) -> np.ndarray:
    """
    Menghitung nilai eksak fungsi tekanan eksponensial.

    Model tekanan reservoir:
        P(z) = P0 * exp(-alpha * z)

    Parameter
    ----------
    P0    : Tekanan permukaan (psi)
    alpha : Parameter peluruhan reservoir (1/m)
    z     : Array kedalaman (meter)

    Return
    ------
    Array nilai tekanan eksak (psi)
    """
    return P0 * np.exp(-alpha * z)


def taylor_order2(P0: float, alpha: float, a: float, z: np.ndarray) -> np.ndarray:
    """
    Polinom Taylor Orde-2 dari P(z) = P0*exp(-alpha*z) di sekitar z = a.

    Rumus Taylor Orde-2:
        T2(z) = P(a)
               + P'(a)  * (z - a)
               + P''(a) / 2! * (z - a)^2

    Dimana:
        P(a)   = P0 * exp(-alpha*a)
        P'(a)  = -alpha * P0 * exp(-alpha*a)
        P''(a) = alpha^2 * P0 * exp(-alpha*a)

    Parameter
    ----------
    P0    : Tekanan permukaan (psi)
    alpha : Parameter peluruhan reservoir (1/m)
    a     : Titik ekspansi (meter)
    z     : Array kedalaman (meter)

    Return
    ------
    Array nilai aproksimasi Taylor Orde-2 (psi)
    """
    Pa   = P0 * np.exp(-alpha * a)          # P(a)
    dz   = z - a                            # (z - a)

    T2 = (Pa
          + (-alpha * Pa) * dz
          + (alpha**2 * Pa / 2.0) * dz**2)
    return T2


def taylor_order3(P0: float, alpha: float, a: float, z: np.ndarray) -> np.ndarray:
    """
    Polinom Taylor Orde-3 dari P(z) = P0*exp(-alpha*z) di sekitar z = a.

    Rumus Taylor Orde-3:
        T3(z) = P(a)
               + P'(a)   * (z - a)
               + P''(a)  / 2! * (z - a)^2
               + P'''(a) / 3! * (z - a)^3

    Dimana:
        P'''(a) = -alpha^3 * P0 * exp(-alpha*a)

    Parameter
    ----------
    P0    : Tekanan permukaan (psi)
    alpha : Parameter peluruhan reservoir (1/m)
    a     : Titik ekspansi (meter)
    z     : Array kedalaman (meter)

    Return
    ------
    Array nilai aproksimasi Taylor Orde-3 (psi)
    """
    Pa   = P0 * np.exp(-alpha * a)
    dz   = z - a

    T3 = (Pa
          + (-alpha * Pa) * dz
          + (alpha**2 * Pa / 2.0) * dz**2
          + (-alpha**3 * Pa / 6.0) * dz**3)
    return T3


def relative_error(exact: np.ndarray, approx: np.ndarray) -> np.ndarray:
    """
    Menghitung galat relatif (%) antara nilai eksak dan aproksimasi.

        Galat (%) = |eksak - aproksimasi| / |eksak| * 100

    Nilai NaN dikembalikan di tempat pembagian-nol (eksak = 0).

    Parameter
    ----------
    exact  : Array nilai eksak
    approx : Array nilai aproksimasi

    Return
    ------
    Array galat relatif (%)
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        err = np.where(
            exact != 0,
            np.abs((exact - approx) / exact) * 100.0,
            np.nan
        )
    return err


# ─────────────────────────────────────────────────────────────────────────────
# 2. PEMBUATAN TABEL ANALISIS GALAT
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(P0: float, alpha: float, a: float,
                    depths: list[float]) -> pd.DataFrame:
    """
    Membangun DataFrame ringkasan analisis galat pada kedalaman tertentu.

    Kolom yang dihasilkan:
        Kedalaman (m), Eksak (psi), Taylor-2 (psi), Taylor-3 (psi),
        Galat-2 (%), Galat-3 (%)

    Parameter
    ----------
    P0     : Tekanan permukaan (psi)
    alpha  : Parameter peluruhan reservoir (1/m)
    a      : Titik ekspansi (meter)
    depths : List kedalaman evaluasi (meter)

    Return
    ------
    pd.DataFrame berisi ringkasan analisis galat
    """
    z_pts = np.array(depths, dtype=float)

    exact = compute_pressure(P0, alpha, z_pts)
    t2    = taylor_order2(P0, alpha, a, z_pts)
    t3    = taylor_order3(P0, alpha, a, z_pts)
    e2    = relative_error(exact, t2)
    e3    = relative_error(exact, t3)

    df = pd.DataFrame({
        "Kedalaman (m)" : z_pts.astype(int),
        "Eksak (psi)"   : np.round(exact, 4),
        "Taylor-2 (psi)": np.round(t2, 4),
        "Taylor-3 (psi)": np.round(t3, 4),
        "Galat-2 (%)"   : np.round(e2, 6),
        "Galat-3 (%)"   : np.round(e3, 6),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_pressure(z: np.ndarray, exact: np.ndarray,
                  t2: np.ndarray, t3: np.ndarray,
                  a: float) -> plt.Figure:
    """
    Grafik 1 — Perbandingan Tekanan vs Kedalaman.

    Menampilkan 3 kurva:
        • Solusi Eksak   (P0 * exp(-alpha*z))
        • Taylor Orde-2  (parabola)
        • Taylor Orde-3  (kubik)

    Garis vertikal putus-putus menandai titik ekspansi a.

    Parameter
    ----------
    z     : Array kedalaman (m)
    exact : Array tekanan eksak (psi)
    t2    : Array aproksimasi Orde-2 (psi)
    t3    : Array aproksimasi Orde-3 (psi)
    a     : Titik ekspansi (m)

    Return
    ------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    ax.plot(z, exact, color='#00C8FF', linewidth=2.5,
            label='Solusi Eksak  $P_0 e^{-\\alpha z}$')
    ax.plot(z, t2,    color='#FF8C42', linewidth=2.0, linestyle='--',
            label='Taylor Orde-2')
    ax.plot(z, t3,    color='#6EE7B7', linewidth=2.0, linestyle='-.',
            label='Taylor Orde-3')

    ax.axvline(x=a, color='#FBBF24', linewidth=1.5, linestyle=':',
               label=f'Titik ekspansi  $a = {a:.0f}$ m')

    ax.set_xlabel('Kedalaman, $z$ (meter)', color='#E2E8F0', fontsize=12)
    ax.set_ylabel('Tekanan, $P$ (psi)',     color='#E2E8F0', fontsize=12)
    ax.set_title('Perbandingan Tekanan: Eksak vs Polinom Taylor',
                 color='#F8FAFC', fontsize=13, fontweight='bold', pad=12)

    ax.tick_params(colors='#94A3B8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    ax.legend(facecolor='#1E293B', edgecolor='#475569',
              labelcolor='#E2E8F0', fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(color='#1E293B', linestyle='--', linewidth=0.7, alpha=0.8)

    fig.tight_layout()
    return fig


def plot_error(z: np.ndarray, e2: np.ndarray, e3: np.ndarray,
               a: float) -> plt.Figure:
    """
    Grafik 2 — Galat Relatif (%) vs Kedalaman dengan skala-Y logaritmik.

    Skala logaritmik memperlihatkan perbedaan galat di rentang kedalaman
    dekat dan jauh dari titik ekspansi secara bersamaan.

    Parameter
    ----------
    z  : Array kedalaman (m)
    e2 : Array galat relatif Orde-2 (%)
    e3 : Array galat relatif Orde-3 (%)
    a  : Titik ekspansi (m)

    Return
    ------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Hindari log(0) — filter nilai ≤ 0 sebelum plot
    mask2 = e2 > 0
    mask3 = e3 > 0

    ax.semilogy(z[mask2], e2[mask2], color='#FF8C42', linewidth=2.0,
                linestyle='--', label='Galat Taylor Orde-2')
    ax.semilogy(z[mask3], e3[mask3], color='#6EE7B7', linewidth=2.0,
                linestyle='-.', label='Galat Taylor Orde-3')

    ax.axvline(x=a, color='#FBBF24', linewidth=1.5, linestyle=':',
               label=f'Titik ekspansi  $a = {a:.0f}$ m')

    ax.set_xlabel('Kedalaman, $z$ (meter)', color='#E2E8F0', fontsize=12)
    ax.set_ylabel('Galat Relatif (%)',       color='#E2E8F0', fontsize=12)
    ax.set_title('Galat Relatif Polinom Taylor (Skala Log)',
                 color='#F8FAFC', fontsize=13, fontweight='bold', pad=12)

    ax.tick_params(colors='#94A3B8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    ax.legend(facecolor='#1E293B', edgecolor='#475569',
              labelcolor='#E2E8F0', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(color='#1E293B', linestyle='--', linewidth=0.7, alpha=0.8,
            which='both')

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN — STREAMLIT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Entry point aplikasi Streamlit.

    Alur:
        1. Konfigurasi halaman dan judul
        2. Sidebar — input parameter interaktif dari pengguna
        3. Komputasi — hitung eksak, Taylor-2, Taylor-3, galat
        4. Metrik ringkasan — kartu statistik di atas grafik
        5. Grafik 1 — kurva tekanan vs kedalaman
        6. Grafik 2 — galat relatif (skala log)
        7. Tabel — DataFrame analisis galat pada kedalaman tertentu
    """

    # ── Konfigurasi Halaman ────────────────────────────────────────────────
    st.set_page_config(
        page_title="Aproksimasi Taylor",
        page_icon="🛢️",
        layout="wide",
    )

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; color:#00C8FF;'>
        🛢️ Aproksimasi Perilaku Fungsi Tekanan<br>dengan Polinom Taylor
    </h1>
    <p style='text-align:center; color:#94A3B8; font-size:14px;'>
        Fungsi Tekanan &nbsp;|&nbsp; P(z) = P₀ e<sup>−αz</sup>
    </p>
    <hr style='border-color:#334155;'>
    """, unsafe_allow_html=True)

    # ── Sidebar — Input Pengguna ───────────────────────────────────────────
    st.sidebar.header("⚙️ Parameter Reservoir")

    P0 = st.sidebar.number_input(
        "P₀ — Tekanan Permukaan (psi)",
        min_value=100.0, max_value=10_000.0,
        value=3000.0, step=100.0,
        help="Tekanan di permukaan sumur (z = 0)"
    )

    alpha = st.sidebar.number_input(
        "α — Parameter Reservoir (1/m)",
        min_value=1e-6, max_value=1e-2,
        value=0.0005, step=0.0001, format="%.4f",
        help="Koefisien peluruhan tekanan terhadap kedalaman"
    )

    a = st.sidebar.slider(
        "a — Titik Ekspansi Taylor (m)",
        min_value=0, max_value=3000,
        value=1000, step=50,
        help="Titik referensi di mana deret Taylor dikembangkan"
    )

    st.sidebar.markdown("---")
    z_min, z_max = st.sidebar.slider(
        "Rentang Kedalaman z (meter)",
        min_value=0, max_value=5000,
        value=(0, 3000), step=50,
        help="Batas bawah dan atas kedalaman yang ditampilkan"
    )

    # ── Komputasi ──────────────────────────────────────────────────────────
    z      = np.linspace(z_min, z_max, 800)
    exact  = compute_pressure(P0, alpha, z)
    t2     = taylor_order2(P0, alpha, a, z)
    t3     = taylor_order3(P0, alpha, a, z)
    e2     = relative_error(exact, t2)
    e3     = relative_error(exact, t3)

    # ── Metrik Ringkasan ───────────────────────────────────────────────────
    Pa_exact = float(compute_pressure(P0, alpha, np.array([a]))[0])
    Pa_t2    = float(taylor_order2(P0, alpha, a, np.array([a]))[0])
    Pa_t3    = float(taylor_order3(P0, alpha, a, np.array([a]))[0])

    st.subheader("📊 Nilai Tekanan di Titik Ekspansi (z = a)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eksak  P(a)",    f"{Pa_exact:,.2f} psi")
    c2.metric("Taylor Orde-2",  f"{Pa_t2:,.2f} psi",)
    c3.metric("Taylor Orde-3",  f"{Pa_t3:,.2f} psi",)
    c4.metric("α · a (tanpa dimensi)", f"{alpha * a:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plot 1: Tekanan vs Kedalaman ───────────────────────────────────────
    st.subheader("📈 Plot 1 — Tekanan vs Kedalaman")
    fig1 = plot_pressure(z, exact, t2, t3, a)
    st.pyplot(fig1, use_container_width=True)

    # ── Plot 2: Galat Relatif ──────────────────────────────────────────────
    st.subheader("📉 Plot 2 — Galat Relatif (Skala Logaritmik)")
    fig2 = plot_error(z, e2, e3, a)
    st.pyplot(fig2, use_container_width=True)

    # ── Tabel Analisis Galat ───────────────────────────────────────────────
    st.subheader("📋 Analisis Galat pada Kedalaman Tertentu")
    sample_depths = [500, 1000, 1500, 2000, 2500]
    df = build_dataframe(P0, alpha, a, sample_depths)

    st.dataframe(
        df.style
          .format({
              "Eksak (psi)"   : "{:,.4f}",
              "Taylor-2 (psi)": "{:,.4f}",
              "Taylor-3 (psi)": "{:,.4f}",
              "Galat-2 (%)"   : "{:.6f}",
              "Galat-3 (%)"   : "{:.6f}",
          })
          .background_gradient(subset=["Galat-2 (%)", "Galat-3 (%)"],
                                cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Rumus Matematis (Expander) ─────────────────────────────────────────
    with st.expander("📐 Rumus Matematika — Polinom Taylor"):
        st.latex(r"P(z) = P_0 \, e^{-\alpha z}")
        st.markdown("**Taylor Orde-2** di sekitar $z = a$:")
        st.latex(r"""
            T_2(z) = P(a) - \alpha P(a)(z-a) + \frac{\alpha^2 P(a)}{2!}(z-a)^2
        """)
        st.markdown("**Taylor Orde-3** di sekitar $z = a$:")
        st.latex(r"""
            T_3(z) = T_2(z) - \frac{\alpha^3 P(a)}{3!}(z-a)^3
        """)
        st.markdown("**Galat Relatif:**")
        st.latex(r"""
            \varepsilon_r = \left| \frac{P(z) - T_n(z)}{P(z)} \right| \times 100\%
        """)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style='border-color:#334155;'>
    <p style='text-align:center; color:#475569; font-size:12px;'>
        Kalkulus Teknik Perminyakan &nbsp;·&nbsp;
        Aproksimasi Fungsi Tekanan dengan Polinom Taylor
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
