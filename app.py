import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 (Streamlit Cloud에 최적화) ---
# packages.txt를 통해 설치된 나눔 폰트를 사용하도록 설정
@st.cache_data
def font_setup():
    # matplotlib 폰트 캐시를 다시 빌드
    fm._rebuild()
    
    # 설치된 나눔고딕 폰트 경로 확인
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic_files = [f for f in font_files if 'NanumGothic' in f]
    
    if nanum_gothic_files:
        # 나눔고딕 폰트를 기본 폰트로 설정
        plt.rc('font', family='NanumGothic')
        font_prop = fm.FontProperties(fname=nanum_gothic_files[0]) # 첫 번째 찾은 폰트 사용
    else:
        # 폰트가 없는 경우 기본값 사용 (경고 메시지 표시)
        st.warning("나눔고딕 폰트를 찾을 수 없습니다. packages.txt 파일이 올바르게 설정되었는지 확인하세요. 글자가 깨질 수 있습니다.")
        font_prop = fm.FontProperties(size=12) # 폴백
        
    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

font_prop = font_setup()


# --- 2. 2D 열전달 시뮬레이션 함수 ---
# (이전과 동일, 안정성 높음)
def run_2d_heat_simulation(k, L_x, rho, cp=1000, T_hot=1000+273.15, T_initial=20+273.15, sim_time_minutes=15):
    sim_time_seconds = sim_time_minutes * 60
    L_y = 0.1
    alpha = k / (rho * cp)
    nx, ny = 50, 25
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)
    # 안정성 조건(Courant-Friedrichs-Lewy condition)을 고려한 dt 계산
    dt = 0.2 * (1 / (alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5 # dt가 너무 크지 않도록 상한 설정
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None

    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial
    TARGET_TEMP_KELVIN = 120 + 273.15
    time_to_target = None

    for t_step in range(nt):
        T_old = T.copy()
        # 경계 조건 (Boundary Conditions)
        T[:, 0] = T_hot      # 왼쪽: 고온
        T[:, -1] = T[:, -2]  # 오른쪽: 단열 (Neumann)
        T[0, :] = T[1, :]    # 위쪽: 단열 (Neumann)
        T[-1, :] = T[-2, :]  # 아래쪽: 단열 (Neumann)
        
        # 유한 차분법을 이용한 내부 온도 계산
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

# --- 3. 시나리오(재료) 정의 ---
scenarios = {
    '에어로겔 (최상급 단열재)': {'k': 0.02, 'rho': 80, 'cp': 1000},
    '고강도 경량 단열 타일 (우주왕복선)': {'k': 0.06, 'rho': 145, 'cp': 1000},
    '세라믹 섬유 (고성능 단열재)': {'k': 0.1, 'rho': 150, 'cp': 1000},
    '알루미늄 (열 전도체 비교용)': {'k': 200.0, 'rho': 2700, 'cp': 900},
}

# --- 4. Streamlit UI 구성 (15분 고정 시간 버전) ---
st.set_page_config(layout="wide")
st.title("🌡️ 2D 열전달 시뮬레이션")
st.markdown("외부 1000°C 환경에서 **15분** 동안, 재료의 **두께**에 따라 내부 온도가 어떻게 변하는지 관찰합니다.")

st.sidebar.header("⚙️ 시뮬레이션 설정")
selected_material_name = st.sidebar.selectbox("1. 재료 선택", options=list(scenarios.keys()))
thickness_mm = st.sidebar.slider("2. 재료 두께 (mm)", min_value=10.0, max_value=200.0, value=50.0, step=1.0)

thickness_m = thickness_mm / 1000.0
material_props = scenarios[selected_material_name]
k = material_props['k']; rho = material_props['rho']; cp = material_props['cp']
SIMULATION_TIME_MINUTES = 15

if st.sidebar.button("🚀 시뮬레이션 실행"):
    with st.spinner(f"'{selected_material_name}'(두께: {thickness_mm}mm)으로 {SIMULATION_TIME_MINUTES}분간 시뮬레이션 중..."):
        time_pts, temp_hist, final_temp_dist, time_to_target = run_2d_heat_simulation(
            k=k, L_x=thickness_m, rho=rho, cp=cp, sim_time_minutes=SIMULATION_TIME_MINUTES
        )

    st.subheader(f"📊 {SIMULATION_TIME_MINUTES}분 시뮬레이션 결과")
    
    with st.expander("🔬 선택 재료의 물리적 특성 보기"):
        st.markdown(f"- **열전도율 (k)**: `{k}` W/m·K (낮을수록 단열 성능 좋음)")

    if time_pts is None:
        st.error("시뮬레이션 조건이 너무 극단적이라 계산이 불가능합니다.")
    else:
        final_temp = temp_hist[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric(f"최종 온도 ({SIMULATION_TIME_MINUTES}분 후)", f"{final_temp:.1f} °C")
        
        if final_temp < 120:
             col2.metric("목표(120°C) 달성", "✅ 성공")
        else:
             col2.metric("목표(120°C) 달성", "❌ 실패")

        if time_to_target is not None:
            col3.metric("120°C 도달 시간", f"{time_to_target:.1f} 분")
        else:
            col3.metric("120°C 도달 시간", f"{SIMULATION_TIME_MINUTES}분 이상")

        # --- 5. 결과 시각화 ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(time_pts / 60, temp_hist, label=f"{selected_material_name} ({thickness_mm}mm)", lw=2.5)
        ax1.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
        
        ax1.set_title(f'내부 표면 온도 변화', fontproperties=font_prop, fontsize=16)
        ax1.set_xlabel('시간 (분)', fontproperties=font_prop)
        ax1.set_ylabel('평균 온도 (°C)', fontproperties=font_prop)
        ax1.legend(prop=font_prop); ax1.grid(True, linestyle=':'); ax1.set_xlim(0, SIMULATION_TIME_MINUTES)
        max_temp_visual = max(temp_hist)
        ax1.set_ylim(15, max(150, max_temp_visual * 1.2))
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        im = ax2.imshow(final_temp_dist, cmap='inferno', aspect='auto', extent=[0, thickness_mm, 0, 10], vmin=20, vmax=1000)
        fig2.colorbar(im, ax=ax2, label='온도 (°C)'); ax2.set_title(f'최종 시간에서의 2D 온도 분포', fontproperties=font_prop, fontsize=16)
        ax2.set_xlabel('두께 방향 (mm)', fontproperties=font_prop); ax2.set_ylabel('높이 방향 (cm)', fontproperties=font_prop)
        st.pyplot(fig2)

else:
    st.info("사이드바에서 재료와 두께를 설정한 후 '시뮬레이션 실행' 버튼을 눌러주세요.")
