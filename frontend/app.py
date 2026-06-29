from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

from apt_ui.pages.attribution import render_attribution
from apt_ui.pages.clustering import render_clustering
from apt_ui.pages.datasets import render_datasets
from apt_ui.pages.features import render_features
from apt_ui.pages.models import render_models
from apt_ui.pages.report import render_report
from apt_ui.pages.upload import render_upload
from apt_ui.services.api_client import get_dashboard_counts, get_runtime_config
from apt_ui.services import ui


AUTH_QUERY_KEY = "auth"
AUTH_ENABLED = os.getenv("ENABLE_UI_AUTH", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin")
AUTH_SECRET = os.getenv("AUTH_SESSION_SECRET", "rgapt-session-secret")


@st.cache_data(show_spinner=False)
def _read_css(file_path: str, mtime_ns: int) -> str:
    del mtime_ns
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _get_query_params() -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    for key in st.query_params:
        values = st.query_params.get_all(key)
        if values:
            params[key] = values
    return params


def _set_query_params(params: dict[str, list[str] | str]) -> None:
    st.query_params.from_dict(params)


def _auth_signature(username: str) -> str:
    return hmac.new(
        AUTH_SECRET.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _build_auth_token(username: str) -> str:
    return f"{username}:{_auth_signature(username)}"


def _is_valid_auth_token(token: str | None) -> bool:
    if not token or ":" not in token:
        return False
    username, signature = token.split(":", 1)
    if username != AUTH_USERNAME:
        return False
    return hmac.compare_digest(signature, _auth_signature(username))


def _get_auth_token_from_query() -> str | None:
    params = _get_query_params()
    values = params.get(AUTH_QUERY_KEY, [])
    return values[0] if values else None


def _set_auth_token(token: str | None) -> None:
    params = _get_query_params()
    if token:
        if params.get(AUTH_QUERY_KEY, [None])[0] == token:
            return
        params[AUTH_QUERY_KEY] = token
    else:
        if AUTH_QUERY_KEY not in params:
            return
        params.pop(AUTH_QUERY_KEY, None)
    _set_query_params(params)


def check_password() -> bool:
    """Returns `True` if the user had a correct password."""

    if not AUTH_ENABLED:
        return True

    def password_entered() -> None:
        if (
            st.session_state["username"] == AUTH_USERNAME
            and st.session_state["password"] == AUTH_PASSWORD
        ):
            st.session_state["password_correct"] = True
            _set_auth_token(_build_auth_token(AUTH_USERNAME))
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
            _set_auth_token(None)

    if _is_valid_auth_token(_get_auth_token_from_query()):
        st.session_state["password_correct"] = True
        return True

    if "password_correct" not in st.session_state:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 2rem; margin-top: 5rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h1 style="color: #00d4ff; letter-spacing: 2px;">RGAPT</h1>
                    <p style="color: #a0aab5;">Semantic-Enhanced APT Threat Graph Attribution System</p>
                    <p style="color: #6c757d; font-size: 0.9em;">基于关系感知图注意力网络的 APT 归因分析系统</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", width="stretch")
            if AUTH_USERNAME == "admin" and AUTH_PASSWORD == "admin":
                st.info("Default: admin / admin")

        return False

    if not st.session_state["password_correct"]:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 2rem; margin-top: 5rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: #00d4ff;">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h1 style="color: #00d4ff; letter-spacing: 2px;">RGAPT</h1>
                    <p style="color: #a0aab5;">Semantic-Enhanced APT Threat Graph Attribution System</p>
                    <p style="color: #6c757d; font-size: 0.9em;">基于关系感知图注意力网络的 APT 归因分析系统</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", width="stretch")
            st.error("😕 User not known or password incorrect")
        return False

    return True


def load_css(file_path: Path) -> None:
    stat = Path(file_path).stat()
    st.markdown(f'<style>{_read_css(str(file_path), stat.st_mtime_ns)}</style>', unsafe_allow_html=True)


FEATURE_MODULES = [
    ("fa-tasks", "任务管理", "分析任务与预处理"),
    ("fa-chart-bar", "特征提取", "多维特征可视化"),
    ("fa-project-diagram", "模型训练", "关系感知图注意力网络"),
    ("fa-bullseye", "APT 归因", "攻击组织溯源"),
    ("fa-stream", "归因记录", "历史结果与解释证据"),
    ("fa-database", "数据集", "样本数据管理"),
    ("fa-file-alt", "报告生成", "分析报告导出"),
    ("fa-brain", "模型管理", "算法模型管理"),
]


def _feature_modules(training_ui_enabled: bool) -> list[tuple[str, str, str]]:
    if training_ui_enabled:
        return FEATURE_MODULES
    return [item for item in FEATURE_MODULES if item[1] != "模型训练"]


def _feature_card(icon: str, title: str, desc: str) -> str:
    return (
        f'<div class="feature-module">'
        f'<i class="fas {icon}"></i>'
        f'<h3>{title}</h3>'
        f'<p>{desc}</p>'
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


def render_home(training_ui_enabled: bool) -> None:
    capability_text = "集成特征提取、模型训练与 APT 归因能力。" if training_ui_enabled else "集成特征提取、APT 归因与报告导出能力。"
    st.markdown(
        f"""
        <div style="text-align: center; margin: 1rem 0 2.6rem 0; padding: 2.4rem 1rem;
             background: linear-gradient(160deg, rgba(8,32,52,0.6), rgba(4,20,33,0.4));
             border: 1px solid var(--border-color); border-radius: 16px;">
            <div style="font-size: 2.6rem; color: #00d4ff; margin-bottom: 0.6rem;
                 filter: drop-shadow(0 0 12px rgba(0,212,255,0.4));">
                <i class="fas fa-shield-alt"></i>
            </div>
            <h1 style="font-size: 1.7rem; margin: 0 0 0.4rem 0;">基于关系感知图注意力网络的 APT 归因分析系统</h1>
            <p style="color: #00d4ff; opacity: 0.75; letter-spacing: 1.5px; text-transform: uppercase;
               font-size: 0.82rem; margin: 0 0 1.2rem 0;">Semantic-Enhanced APT Threat Graph Attribution System</p>
            <p style="font-size: 1rem; color: var(--text-muted); max-width: 760px; margin: 0 auto; line-height: 1.7;">
                帮助安全专家快速、准确地识别与分析高级持续性威胁，{capability_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title"><i class="fas fa-th-large"></i><span>核心功能</span></div>', unsafe_allow_html=True)
    feature_modules = _feature_modules(training_ui_enabled)
    for row_start in range(0, len(feature_modules), 4):
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, feature_modules[row_start:row_start + 4]):
            col.markdown(_feature_card(icon, title, desc), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"><i class="fas fa-gauge-high"></i><span>系统资源概览</span></div>', unsafe_allow_html=True)

    counts = get_dashboard_counts()
    cards = [
        ("fa-database", counts["datasets"], "数据集"),
        ("fa-brain", counts["models"], "已训练模型"),
        ("fa-stream", counts["results"], "归因结果"),
        ("fa-check-circle", "Ready", "系统状态"),
    ]
    for col, (icon, value, label) in zip(st.columns(4), cards):
        col.markdown(_stat_card(icon, value, label), unsafe_allow_html=True)


def _clear_session() -> None:
    _set_auth_token(None)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="APT归因分析系统",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True,
    )

    if not check_password():
        return

    css_file = Path(__file__).parent / "css" / "styles.css"
    if css_file.exists():
        load_css(css_file)

    runtime_config = get_runtime_config()
    training_ui_enabled = runtime_config.get("training_ui_enabled", True)

    menu_entries = [
        ("home", "主页", "house"),
        ("tasks", "分析任务管理", "list-task"),
        ("datasets", "数据集管理", "database"),
        ("features", "特征提取与可视化", "bar-chart"),
        ("models", "模型管理", "cpu"),
        ("attribution", "APT归因结果", "bullseye"),
        ("report", "报告生成与导出", "file-text"),
    ]
    if training_ui_enabled:
        menu_entries.insert(4, ("training", "模型训练", "diagram-3"))

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem; padding: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));"><i class="fas fa-shield-alt"></i></div>
                <h2 style="margin:0; color: white; letter-spacing: 2px;">RGAPT</h2>
                <p style="color: #00d4ff; font-size: 0.8rem; letter-spacing: 1px; opacity: 0.8;">Semantic-Enhanced APT Attribution</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=[item[1] for item in menu_entries],
            icons=[item[2] for item in menu_entries],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#00d4ff", "font-size": "15px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "3px 8px",
                    "padding": "0.55rem 0.8rem",
                    "color": "#93a4b3",
                    "border-radius": "8px",
                    "transition": "background-color 0.18s ease, color 0.18s ease",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 212, 255, 0.12)",
                    "color": "#ffffff",
                    "border-left": "3px solid #00d4ff",
                    "font-weight": "600",
                },
            },
        )

        st.divider()
        with st.expander("🔧 会话管理"):
            st.caption("刷新浏览器后仍会保留当前登录状态，直到主动退出。")
            if st.button("🗑️ 清空当前会话", type="secondary", width="stretch"):
                _clear_session()

    selected_key = next((item[0] for item in menu_entries if item[1] == selected), "home")

    if selected_key == "home":
        render_home(training_ui_enabled)
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
