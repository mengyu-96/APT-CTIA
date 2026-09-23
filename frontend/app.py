from __future__ import annotations

import os
from pathlib import Path
import time

import streamlit as st
from streamlit_option_menu import option_menu

from apt_ui.pages.attribution import render_attribution
from apt_ui.pages.clustering import render_clustering
from apt_ui.pages.datasets import render_datasets
from apt_ui.pages.features import render_features
from apt_ui.pages.models import render_models
from apt_ui.pages.report import render_report
from apt_ui.pages.upload import render_upload
from apt_ui.services.api_client import get_dashboard_counts, get_system_health
from apt_ui.services import ui
from apt_ui.services.security import UserStore, load_auth_config, verify_credentials
from apt_ui.services.streamlit_compat import install_streamlit_width_compatibility


AUTH_CONFIG = load_auth_config(os.environ)
AUTH_USER_STORE = UserStore(AUTH_CONFIG.users_file)
install_streamlit_width_compatibility(st)
SYSTEM_NAME = "HERA"
SYSTEM_SUBTITLE = "基于异构证据推理与语义—结构双流学习的APT组织归因平台"
SYSTEM_SUBTITLE_EN = (
    "APT Organization Attribution Platform Based on Heterogeneous Evidence "
    "Reasoning and Semantic-Structural Dual-stream Learning"
)
MENU_ENTRIES = [
    ("home", "主页", "house"),
    ("tasks", "预处理任务管理", "list-task"),
    ("datasets", "数据集管理", "database"),
    ("features", "特征提取与可视化", "bar-chart"),
    ("training", "模型训练", "diagram-3"),
    ("models", "模型管理", "cpu"),
    ("attribution", "APT归因结果", "bullseye"),
    ("report", "报告生成与导出", "file-text"),
]
PAGE_BY_MENU_LABEL = {label: page for page, label, _ in MENU_ENTRIES}
MENU_LABEL_BY_PAGE = {page: label for page, label, _ in MENU_ENTRIES}


def _sync_navigation_from_widget(key: str) -> None:
    selected_page = PAGE_BY_MENU_LABEL.get(st.session_state.get(key))
    if selected_page is None:
        return
    st.session_state["_main_navigation_page"] = selected_page
    st.query_params["page"] = selected_page


def _finish_authentication(username: str) -> None:
    """Start every new login on the home page, including stale shared links."""

    st.session_state["password_correct"] = True
    st.session_state["authenticated_at"] = time.time()
    st.session_state["auth_username"] = username
    st.session_state["main_navigation"] = MENU_LABEL_BY_PAGE["home"]
    st.session_state["_main_navigation_page"] = "home"
    st.query_params["page"] = "home"
    st.rerun()


@st.cache_data(show_spinner=False)
def _read_css(file_path: str, mtime_ns: int) -> str:
    del mtime_ns
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def check_password() -> bool:
    """Authenticate the current Streamlit session without URL credentials."""

    if not AUTH_CONFIG.enabled:
        st.session_state.setdefault("auth_username", "本地预览")
        return True
    if AUTH_CONFIG.error:
        st.error(f"登录功能配置错误：{AUTH_CONFIG.error}")
        st.caption("请修改部署环境变量后重新启动前端服务。")
        return False
    if st.session_state.get("password_correct"):
        authenticated_at = float(st.session_state.get("authenticated_at", 0) or 0)
        if time.time() - authenticated_at < AUTH_CONFIG.session_ttl_seconds:
            return True
        st.session_state.pop("password_correct", None)
        st.session_state.pop("authenticated_at", None)
        st.warning("登录会话已过期，请重新登录。")

    _, login_col, _ = st.columns([1, 2, 1])
    with login_col:
        auth_slot = st.empty()
        auth_view = st.session_state.setdefault("auth_view", "login")
        previous_auth_view = st.session_state.get("_rendered_auth_view")
        if previous_auth_view is not None and previous_auth_view != auth_view:
            # Send the deletion in its own run. Replacing the container in the
            # same run leaves the old form's trailing elements in the DOM.
            auth_slot.empty()
            st.session_state["_rendered_auth_view"] = auth_view
            st.rerun()
        st.session_state["_rendered_auth_view"] = auth_view
        with auth_slot.container():
            _render_auth_form()
    return False


def _render_auth_form() -> None:
        st.markdown(
            """
            <div class="login-hero">
                <div class="login-icon"><i class="fas fa-shield-alt"></i></div>
                <div class="login-brand-title">{SYSTEM_NAME}</div>
                <p>{SYSTEM_SUBTITLE}</p>
                <p class="login-subtitle-en">{SYSTEM_SUBTITLE_EN}</p>
            </div>
            """.format(
                SYSTEM_NAME=SYSTEM_NAME,
                SYSTEM_SUBTITLE=SYSTEM_SUBTITLE,
                SYSTEM_SUBTITLE_EN=SYSTEM_SUBTITLE_EN,
            ),
            unsafe_allow_html=True,
        )
        st.session_state.setdefault("auth_view", "login")
        if st.session_state.auth_view == "login":
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button(
                    "Login", type="primary", use_container_width=True
                )
            if submitted:
                if verify_credentials(username, password, AUTH_CONFIG, AUTH_USER_STORE):
                    st.session_state.pop("login_password", None)
                    st.session_state.pop("login_username", None)
                    _finish_authentication(username.strip())
                st.session_state["password_correct"] = False
            if st.session_state.get("password_correct") is False:
                st.error("用户名或密码不正确。")
            if AUTH_CONFIG.allow_registration:
                switch_label_col, switch_action_col = st.columns([1, 1], gap="small")
                with switch_label_col:
                    st.markdown('<div class="auth-switch-label">没有账号？</div>', unsafe_allow_html=True)
                with switch_action_col:
                    st.markdown('<span class="auth-switch-action-anchor"></span>', unsafe_allow_html=True)
                    if st.button("注册账号", key="show_registration", use_container_width=False):
                        st.session_state.auth_view = "register"
                        st.rerun()
        else:
            st.markdown('<div class="auth-register-title">注册账号</div>', unsafe_allow_html=True)
            with st.form("registration_form"):
                new_username = st.text_input("Username", key="register_username")
                new_password = st.text_input("Password", type="password", key="register_password")
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="register_password_confirm",
                )
                registered = st.form_submit_button(
                    "Create account", type="primary", use_container_width=True
                )
            st.caption("用户名为 3–32 个字符；密码至少 12 个字符。")
            if registered:
                if new_password != confirm_password:
                    st.error("两次输入的密码不一致。")
                elif AUTH_CONFIG.username and new_username.strip().casefold() == AUTH_CONFIG.username.casefold():
                    st.error("该用户名已存在。")
                else:
                    try:
                        account = AUTH_USER_STORE.register(new_username, new_password)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        _finish_authentication(account["username"])
            if st.button("返回登录", key="show_login", use_container_width=True):
                st.session_state.auth_view = "login"
                st.rerun()


def load_css(file_path: Path) -> None:
    stat = Path(file_path).stat()
    st.markdown(f'<style>{_read_css(str(file_path), stat.st_mtime_ns)}</style>', unsafe_allow_html=True)


@st.dialog("软件说明", width="large")
def render_software_guide() -> None:
    st.markdown(
        """
        **快速开始**

        1. 登录已有账号，或通过“注册账号”创建新账号。
        2. 在“预处理任务管理”中填写数据集名称，上传 TXT、JSON 或 PDF 文件并提交任务。
        3. 任务完成后，在“数据集管理”中检查处理结果；如页面未更新，可点击“刷新”。
        4. 在“特征提取与可视化”中查看实体、关系和图结构统计。
        5. 在“模型训练”中选择数据集和训练参数，提交任务后从任务记录查看结果。
        6. 在“APT归因结果”中选择模型和待分析数据，核对候选组织、置信度与解释证据。
        7. 在“报告生成与导出”中选择归因结果，生成并下载分析报告。

        **其他功能**

        - “模型管理”用于查看、比较和维护训练模型。
        - 退出账号请展开左侧底部的“会话管理”，然后点击“退出登录”。
        """
    )


def render_software_guide_entry() -> None:
    st.markdown('<span class="software-guide-anchor"></span>', unsafe_allow_html=True)
    if st.button("软件说明", key="software_guide_button"):
        render_software_guide()


FEATURE_MODULES = [
    ("fa-tasks", "任务管理", "分析任务与预处理", "blue"),
    ("fa-chart-bar", "特征提取", "多维特征可视化", "purple"),
    ("fa-project-diagram", "模型训练", "多层异构图与双流学习", "indigo"),
    ("fa-bullseye", "APT 归因", "攻击组织溯源", "warning"),
    ("fa-stream", "归因记录", "历史结果与解释证据", "warning"),
    ("fa-database", "数据集", "样本数据管理", "cyan"),
    ("fa-file-alt", "报告生成", "分析报告导出", "success"),
    ("fa-brain", "模型管理", "算法模型管理", "purple"),
]


def _feature_card(icon: str, title: str, desc: str, tone: str) -> str:
    return (
        f'<div class="feature-module feature-module--{tone}">'
        f'<div class="feature-module-icon"><i class="fas {icon}"></i></div>'
        f'<div class="feature-module-title">{title}</div>'
        f'<p>{desc}</p>'
        f'<span class="feature-module-arrow" aria-hidden="true">→</span>'
        f"</div>"
    )


def _stat_card(icon: str, value: object, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-card-icon"><i class="fas {icon}"></i></div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>"
    )


def render_home() -> None:
    st.markdown(
        """
        <section class="home-hero">
            <div class="home-hero__content">
                <div class="home-hero__eyebrow"><i class="fas fa-shield-alt"></i><span>APT INTELLIGENCE PLATFORM</span></div>
                <div class="home-brand-title">{SYSTEM_NAME}</div>
                <p class="home-hero__headline">面向高级持续性威胁的智能分析与攻击组织归因平台</p>
                <p class="home-hero__subtitle">{SYSTEM_SUBTITLE}</p>
                <p class="home-hero__description">覆盖威胁情报预处理、特征建模、模型训练、APT归因与报告生成。</p>
            </div>
            <div class="home-hero__visual" aria-hidden="true">
                <div class="threat-network">
                    <span class="network-ring network-ring--one"></span>
                    <span class="network-ring network-ring--two"></span>
                    <span class="network-line network-line--one"></span>
                    <span class="network-line network-line--two"></span>
                    <span class="network-line network-line--three"></span>
                    <span class="network-node network-node--core"><i class="fas fa-shield-halved"></i></span>
                    <span class="network-node network-node--one"></span>
                    <span class="network-node network-node--two"></span>
                    <span class="network-node network-node--three"></span>
                    <span class="network-node network-node--four"></span>
                </div>
                <div class="network-caption"><span></span>异构证据关联分析</div>
            </div>
        </section>
        """.format(SYSTEM_NAME=SYSTEM_NAME, SYSTEM_SUBTITLE=SYSTEM_SUBTITLE),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title"><i class="fas fa-th-large"></i><span>核心功能</span></div>', unsafe_allow_html=True)
    for row_start in range(0, len(FEATURE_MODULES), 4):
        cols = st.columns(4)
        for col, (icon, title, desc, tone) in zip(cols, FEATURE_MODULES[row_start:row_start + 4]):
            col.markdown(_feature_card(icon, title, desc, tone), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"><i class="fas fa-gauge-high"></i><span>系统资源概览</span></div>', unsafe_allow_html=True)

    counts = get_dashboard_counts()
    cards = [
        ("fa-database", counts["datasets"], "数据集"),
        ("fa-brain", counts["models"], "已训练模型"),
        ("fa-stream", counts["results"], "归因结果"),
        ("fa-check-circle", get_system_health(), "系统状态"),
    ]
    for col, (icon, value, label) in zip(st.columns(4), cards):
        col.markdown(_stat_card(icon, value, label), unsafe_allow_html=True)


def _clear_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=SYSTEM_NAME,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True,
    )

    css_file = Path(__file__).parent / "css" / "styles.css"
    if css_file.exists():
        load_css(css_file)
    ui.apply_accessibility_metadata()
    render_software_guide_entry()

    if not check_password():
        return

    valid_pages = {item[0] for item in MENU_ENTRIES}
    requested_page = st.query_params.get("page", "home")
    if requested_page not in valid_pages:
        requested_page = "home"
    default_index = next(i for i, item in enumerate(MENU_ENTRIES) if item[0] == requested_page)

    if st.session_state.get("_main_navigation_page") != requested_page:
        st.session_state["main_navigation"] = MENU_LABEL_BY_PAGE[requested_page]
        st.session_state["_main_navigation_page"] = requested_page

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon"><i class="fas fa-shield-alt"></i></div>
                <div class="sidebar-brand-title">{SYSTEM_NAME}</div>
                <p class="sidebar-brand-subtitle">{SYSTEM_SUBTITLE}</p>
            </div>
            """.format(SYSTEM_NAME=SYSTEM_NAME, SYSTEM_SUBTITLE=SYSTEM_SUBTITLE),
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=[item[1] for item in MENU_ENTRIES],
            icons=[item[2] for item in MENU_ENTRIES],
            menu_icon="cast",
            default_index=default_index,
            key="main_navigation",
            on_change=_sync_navigation_from_widget,
            styles={
                "container": {"padding": "0!important", "background-color": "#07131F"},
                "icon": {"color": "#38BDF8", "font-size": "15px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "3px 8px",
                    "padding": "0.55rem 0.8rem",
                    "color": "#8393A7",
                    "background-color": "transparent",
                    "--hover-color": "rgba(56, 189, 248, 0.06)",
                    "border-radius": "8px",
                    "transition": "background-color 0.18s ease, color 0.18s ease",
                },
                "nav-link-selected": {
                    "background-color": "rgba(56, 189, 248, 0.11)",
                    "color": "#F8FAFC",
                    "border-left": "3px solid #38BDF8",
                    "font-weight": "600",
                },
            },
        )

        st.divider()
        with st.expander("🔧 会话管理"):
            st.caption("登录状态仅保存在当前浏览器会话中，不写入网址。")
            if st.button("退出登录", type="secondary", use_container_width=True):
                _clear_session()

    selected_key = PAGE_BY_MENU_LABEL.get(selected, "home")
    if selected_key != requested_page:
        st.session_state["_main_navigation_page"] = selected_key
        st.query_params["page"] = selected_key
    _render_page(selected_key)


def _render_page(selected_key: str) -> None:
    if selected_key == "home":
        render_home()
    elif selected_key == "tasks":
        render_upload()
    elif selected_key == "features":
        render_features()
    elif selected_key == "training":
        render_clustering()
    elif selected_key == "attribution":
        render_attribution()
    elif selected_key == "datasets":
        render_datasets()
    elif selected_key == "report":
        render_report()
    elif selected_key == "models":
        render_models()


if __name__ == "__main__":
    main()
