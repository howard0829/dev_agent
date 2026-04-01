"""
멀티 앱 런처 - Streamlit UI
메인 화면 상단 카드 바에서 앱을 선택하면, 해당 앱의 page 모듈이
사이드바와 메인 화면을 독립적으로 렌더링합니다.
"""

import streamlit as st

from apps import discover_apps
from core.styles import apply_common_styles, apply_custom_css


# ──────────────────────────────────────────────
# 앱 레지스트리 로드
# ──────────────────────────────────────────────
APP_REGISTRY = discover_apps()

if not APP_REGISTRY:
    st.error("❌ 등록된 앱이 없습니다. apps/ 디렉토리를 확인하세요.")
    st.stop()

APP_IDS = list(APP_REGISTRY.keys())

# ──────────────────────────────────────────────
# 현재 선택된 앱 결정
# ──────────────────────────────────────────────
if "_selected_app_id" not in st.session_state:
    st.session_state["_selected_app_id"] = APP_IDS[0]

current_app_id = st.session_state["_selected_app_id"]
current_entry = APP_REGISTRY[current_app_id]
current_app = current_entry["config"]
current_page = current_entry["page"]

# ──────────────────────────────────────────────
# 페이지 설정 (반드시 첫 번째 Streamlit 명령)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=current_app["name"],
    page_icon=current_app["icon"],
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 스타일 적용
# ──────────────────────────────────────────────
apply_common_styles()
apply_custom_css(current_app.get("custom_css", ""))

# ──────────────────────────────────────────────
# 앱별 세션 초기화
# ──────────────────────────────────────────────
prefix = current_app_id
current_page.init_app_session(prefix)

# ──────────────────────────────────────────────
# 메인 화면 - 앱 스위처 카드 바
# ──────────────────────────────────────────────
switcher_cols = st.columns(len(APP_IDS))
for i, aid in enumerate(APP_IDS):
    cfg = APP_REGISTRY[aid]["config"]
    is_active = aid == current_app_id
    with switcher_cols[i]:
        st.markdown('<div class="app-card-btn">', unsafe_allow_html=True)
        if st.button(
            f"{cfg['icon']}\n{cfg['name']}",
            key=f"_sw_{aid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state["_selected_app_id"] = aid
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="app-card-desc">{cfg["description"]}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ──────────────────────────────────────────────
# 앱 위임: 사이드바 → 메인 화면
# ──────────────────────────────────────────────
with st.sidebar:
    sidebar_cfg = current_page.render_sidebar(prefix)

current_page.render_main(prefix, sidebar_cfg)
