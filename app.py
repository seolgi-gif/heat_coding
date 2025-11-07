import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 ---
# (이전과 동일)
try:
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((f for f in font_path if 'NanumGothic' in f), None)
    malgun_gothic = next((f for f in font_path if 'Malgun' in f), None)
    if nanum_gothic:
        font_prop = fm.FontProperties(fname=nanum_gothic)
        plt.rc('font', family='NanumGothic')
    elif malgun_gothic:
        font_prop = fm.FontProperties(fname=malgun_gothic)
        plt.rc('font', family='Malgun Gothic')
    else:
        font_prop = fm.FontProperties(size=12)
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    font_prop = fm.FontProperties(size=12)

# --- 2. 2D 열전달 시뮬레이션 함수 ---
# (이전과 동일, 안정성 높음)
def run_2d_heat_simulation(k, L_x, rho, cp=1000, T_hot=1000+273.15, T_initial=20+273.15, sim_time_minutes=5):
    sim_time_seconds = sim_time_minutes * 60
    L_y = 0.1
    alpha = k / (rho * cp)
    nx, ny = 50, 25
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)
    dt = 0.2 * (1 / (alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None

    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial
    TARGET_TEMP_KELVIN = 120 + 273.15
    time_to_target = None

    for t_step in range(nt):
        T_old = T.copy()
        T[:, 0] = T_hot; T[:, -1] = T[:, -2]; T[0, :] = T[1, :]; T[-1, :] = T[-2, :]
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                term1 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dy**2
                term2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dx**2
                T[i, j] = T_old[i, j] + alpha * dt * (term1 + term2)
        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        if time_to_target is None and current_inner_temp_k >= TARGET_TEMP_KELVIN:
            time_to_target = time_points[t_step] / 60
    return time_points, temp_history_celsius, T - 273.15, time_to_target

# --- 3. 시나리오(재료) 정의 (알루미늄 추가) ---
scenarios = {
    '에어로겔 (최상급 단열재)': {'k': 0.02, 'rho': 80, 'cp': 1000},
    '세라믹 섬유 (고성능 단열재)': {'k': 0.1, 'rho': 150, 'cp': 1000},
    '내화 벽돌 (일반 단열재)': {'k': 1.0, 'rho': 2000, 'cp': 1000},
    '알루미늄 (열 전도체 비교용)': {'k': 200.0, 'rho': 2700, 'cp': 900},
}

# --- 4. Streamlit UI 구성 (5분 챌린지 버전) ---
st.set_page_config(layout="wide")
st.title("🔥 단열재 5분 버티기 챌린지")
st.markdown("외부 1000°C 환경에서 선택한 재료가 **5분**간 내부 온도를 120°C 이하로 버텨낼 수 있을까요? **두께**와 **시간**을 조절하며 직접 확인해보세요!")

st.sidebar.header("⚙️ 챌린지 설정")
selected_material_name = st.sidebar.selectbox("1. 챌린지 재료 선택", options=list(scenarios.keys()))
thickness_cm = st.sidebar.slider("2. 재료 두께 (cm)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
sim_time_minutes = st.sidebar.slider("3. 관찰 시간 (분)", min_value=1, max_value=15, value=5, step=1)

thickness_m = thickness_cm / 100.0
material_props = scenarios[selected_material_name]
k = material_props['k']; rho = material_props['rho']; cp = material_props['cp']

if st.sidebar.button("🚀 챌린지 시작!"):
    with st.spinner(f"'{selected_material_name}'(두께: {thickness_cm}cm)으로 {sim_time_minutes}분간 버티기 테스트 중..."):
        time_pts, temp_hist, final_temp_dist, time_to_target = run_2d_heat_simulation(
            k=k, L_x=thickness_m, rho=rho, cp=cp, sim_time_minutes=sim_time_minutes
        )

    st.subheader(f"📊 {sim_time_minutes}분 챌린지 결과")
    
    with st.expander("🔬 선택 재료의 물리적 특성 보기"):
        st.markdown(f"- **열전도율 (k)**: `{k}` W/m·K (낮을수록 단열 성능 좋음)")

    if time_pts is None:
        st.error("시뮬레이션 조건이 너무 극단적이라 계산이 불가능합니다.")
    else:
        final_temp = temp_hist[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric(f"최종 온도 ({sim_time_minutes}분 후)", f"{final_temp:.1f} °C")
        
        # 5분 버티기 목표에 대한 결과 표시
        if time_to_target is None or time_to_target > 5:
             col2.metric("5분 버티기 목표", "🏆 성공!")
        else:
             col2.metric("5분 버티기 목표", "💥 실패!")

        if time_to_target is not None:
            col3.metric("120°C 도달 시간", f"{time_to_target:.1f} 분")
        else:
            col3.metric("120°C 도달 시간", f"{sim_time_minutes}분 이상")

        # --- 5. 결과 시각화 ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(time_pts / 60, temp_hist, label=f"{selected_material_name} ({thickness_cm}cm)", lw=2.5)
        ax1.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
        if time_to_target is not None and time_to_target <= 5: # 5분 버티기 실패 시 강조
            ax1.axvline(x=5, color='orange', linestyle=':', label='5분 목표 지점')
        
        ax1.set_title(f'내부 표면 온도 변화', fontproperties=font_prop, fontsize=16)
        ax1.set_xlabel('시간 (분)', fontproperties=font_prop)
        ax1.set_ylabel('평균 온도 (°C)', fontproperties=font_prop)
        ax1.legend(prop=font_prop); ax1.grid(True, linestyle=':'); ax1.set_xlim(0, sim_time_minutes)
        max_temp_visual = max(temp_hist)
        ax1.set_ylim(15, max(150, max_temp_visual * 1.2))
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        im = ax2.imshow(final_temp_dist, cmap='inferno', aspect='auto', extent=[0, thickness_cm, 0, 10], vmin=20, vmax=1000)
        fig2.colorbar(im, ax=ax2, label='온도 (°C)'); ax2.set_title(f'최종 시간에서의 2D 온도 분포', fontproperties=font_prop, fontsize=16)
        ax2.set_xlabel('두께 방향 (cm)', fontproperties=font_prop); ax2.set_ylabel('높이 방향 (cm)', fontproperties=font_prop)
        st.pyplot(fig2)

else:
    st.info("사이드바에서 재료, 두께, 시간을 설정한 후 '챌린지 시작!' 버튼을 눌러주세요.")

