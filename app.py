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
except Exception as e:
    st.warning(f"한글 폰트를 로드하는 데 실패했습니다. 영문으로 표시될 수 있습니다. 오류: {e}")
    font_prop = fm.FontProperties(size=12)


# --- 2. 2D 열전달 시뮬레이션 함수 (업그레이드) ---
def run_2d_heat_simulation(k, L_x, rho, cp=1000, T_hot=1000+273.15, T_initial=20+273.15, sim_time_minutes=30):
    """
    2D 평판 열전달 시뮬레이션.
    - 각 재료의 실제 밀도(rho)를 인자로 받도록 수정
    - 시뮬레이션 시간을 분 단위로 받도록 수정
    - 목표 온도 도달 시간을 계산하는 로직 추가
    """
    sim_time_seconds = sim_time_minutes * 60
    L_y = 0.1 # 평판 높이 (m), 시뮬레이션에 큰 영향 없음

    alpha = k / (rho * cp)
    nx, ny = 50, 25
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)
    
    dt = 0.2 * (1 / (alpha * (1/dx**2 + 1/dy**2)))
    nt = int(sim_time_seconds / dt)

    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial

    TARGET_TEMP_KELVIN = 120 + 273.15
    time_to_target = None

    for t_step in range(nt):
        T_old = T.copy()
        T[:, 0] = T_hot
        T[:, -1] = T[:, -2]
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                term1 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dy**2
                term2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dx**2
                T[i, j] = T_old[i, j] + alpha * dt * (term1 + term2)
        
        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        
        # 목표 온도(120도) 최초 도달 시간 기록
        if time_to_target is None and current_inner_temp_k >= TARGET_TEMP_KELVIN:
            time_to_target = time_points[t_step] / 60 # 분 단위로 저장

    final_temp_distribution_celsius = T - 273.15
    return time_points, temp_history_celsius, final_temp_distribution_celsius, time_to_target

# --- 3. 시나리오(재료) 정의 ---
scenarios = {
    '에어로겔': {'k': 0.02, 'rho': 80},
    '세라믹 섬유': {'k': 0.1, 'rho': 150},
    '내화 벽돌': {'k': 1.0, 'rho': 2000},
}

# --- 4. Streamlit UI 구성 (업그레이드) ---
st.title("🔥 2D 열전달 시뮬레이션 v2.0")
st.markdown("""
**두께**를 조절하며 1000°C의 외부 열로부터 내부 표면을 **120°C 이하**로 얼마나 오래 방어할 수 있는지 확인해보세요.
- **시뮬레이션 시간**: 30분
- **외부 조건**: 왼쪽 면 1000°C 고정
- **측정**: 오른쪽 면(내부 표면)의 평균 온도 변화
""")

st.sidebar.header("⚙️ 시뮬레이션 설정")
selected_material_name = st.sidebar.selectbox("1. 단열재 종류 선택", options=list(scenarios.keys()))
thickness_cm = st.sidebar.slider("2. 단열재 두께 (cm)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
thickness_m = thickness_cm / 100.0

material_props = scenarios[selected_material_name]
k = material_props['k']
rho = material_props['rho']

if st.sidebar.button("🚀 시뮬레이션 실행"):
    with st.spinner(f"'{selected_material_name}'(두께: {thickness_cm}cm) 시나리오로 30분간 시뮬레이션 중..."):
        time_pts, temp_hist, final_temp_dist, time_to_target = run_2d_heat_simulation(
            k=k, L_x=thickness_m, rho=rho, sim_time_minutes=30
        )
        final_temp = temp_hist[-1]

        st.subheader("📊 시뮬레이션 결과")
        
        # --- 결과 분석 ---
        col1, col2, col3 = st.columns(3)
        col1.metric("최종 온도 (30분 후)", f"{final_temp:.1f} °C")

        if final_temp < 120:
            col2.metric("목표 달성 여부", "✅ Pass")
        else:
            col2.metric("목표 달성 여부", "❌ Fail")

        if time_to_target is not None:
            col3.metric("120°C 도달 시간", f"{time_to_target:.1f} 분")
        else:
            col3.metric("120°C 도달 시간", "30분 이상")

        # --- 5. 결과 시각화 (업그레이드) ---
        # 그래프 1: 시간에 따른 온도 변화
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(time_pts / 60, temp_hist, label=f"{selected_material_name} (두께: {thickness_cm}cm)", lw=2)
        ax1.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
        ax1.set_title(f'내부 표면 온도 변화 (두께: {thickness_cm}cm)', fontproperties=font_prop, fontsize=16)
        ax1.set_xlabel('시간 (분)', fontproperties=font_prop)
        ax1.set_ylabel('평균 온도 (°C)', fontproperties=font_prop)
        ax1.legend(prop=font_prop)
        ax1.grid(True, linestyle=':')
        ax1.set_xlim(0, 30)
        
        # Y축 자동 조절 로직
        max_temp_visual = max(temp_hist)
        if max_temp_visual < 150: # 성공적인 경우, 그래프 확대
            ax1.set_ylim(15, 150)
        else: # 실패한 경우, 전체를 다 보여줌
            ax1.set_ylim(15, max_temp_visual * 1.2)
            
        st.pyplot(fig1)

        # 그래프 2: 2D 온도 분포 히트맵
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        im = ax2.imshow(final_temp_dist, cmap='inferno', aspect='auto', extent=[0, thickness_cm, 0, 10], vmin=20, vmax=1000)
        fig2.colorbar(im, ax=ax2, label='온도 (°C)')
        ax2.set_title(f'최종 시간(30분)에서의 2D 온도 분포', fontproperties=font_prop, fontsize=16)
        ax2.set_xlabel('두께 방향 (cm)', fontproperties=font_prop)
        ax2.set_ylabel('높이 방향 (cm)', fontproperties=font_prop)
        st.pyplot(fig2)

else:
    st.info("사이드바에서 설정을 마친 후 '시뮬레이션 실행' 버튼을 눌러주세요.")

