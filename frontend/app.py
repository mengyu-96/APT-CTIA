import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import os
from apt_ui.services.api_client import get_json

# Import pages
from apt_ui.pages.upload import render_upload
from apt_ui.pages.clustering import render_clustering
from apt_ui.pages.features import render_features
from apt_ui.pages.attribution import render_attribution
from apt_ui.pages.report import render_report
from apt_ui.pages.gangs import render_gangs
from apt_ui.pages.datasets import render_datasets
from apt_ui.pages.models import render_models

# --- Helper Functions ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")

@st.cache_data(show_spinner=False)
def _read_css(file_path: str, mtime_ns: int) -> str:
    del mtime_ns
    with open(file_path) as f:
        return f.read()

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] == "admin"
            and st.session_state["password"] == "admin"
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.markdown(
            """
            <style>
            .login-container {
                display: flex;
                justify_content: center;
                align_items: center;
                height: 100vh;
            }
            .login-box {
                width: 400px;
                padding: 40px;
                background: rgba(16, 20, 24, 0.8);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem; margin-top: 5rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <h1 style="color: #00d4ff; letter-spacing: 2px;">RGAPT</h1>
                <p style="color: #a0aab5;">Semantic-Enhanced APT Threat Graph Attribution System</p>
                <p style="color: #6c757d; font-size: 0.9em;">基于关系感知图注意力网络的APT归因分析系统</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", use_container_width=True)
            
            st.info("Default: admin / admin")
            
        return False
        
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem; margin-top: 5rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem; color: #00d4ff;">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <h1 style="color: #00d4ff; letter-spacing: 2px;">RGAPT</h1>
                <p style="color: #a0aab5;">Semantic-Enhanced APT Threat Graph Attribution System</p>
                <p style="color: #6c757d; font-size: 0.9em;">基于关系感知图注意力网络的APT归因分析系统</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", use_container_width=True)
            st.error("😕 User not known or password incorrect")
        return False
        
    else:
        # Password correct.
        return True

def load_css(file_path):
    """Loads a CSS file into the Streamlit app."""
    stat = Path(file_path).stat()
    st.markdown(f'<style>{_read_css(str(file_path), stat.st_mtime_ns)}</style>', unsafe_allow_html=True)

def render_home():
    """Renders the Home/Dashboard page with module overview."""
    
    # Hero Section
    st.markdown("""
    <div class="welcome-content" style="text-align: center; margin-bottom: 3rem; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; color: #00d4ff; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);">
            <i class="fas fa-shield-alt"></i> 
        </h1>
        <h2 style="color: #00d4ff; font-weight: 300; letter-spacing: 2px; margin-bottom: 2rem; font-size: 1.5rem;">
            基于关系感知图注意力网络的APT归因分析系统
            <span style="font-size: 1rem; color: #a0aab5; display: block; margin-top: 5px;">Semantic-Enhanced APT Threat Graph Attribution System</span>
        </h2>
        <p style="font-size: 1.1rem; color: #a0aab5; max-width: 800px; margin: 0 auto; line-height: 1.6;">
            致力于帮助安全专家快速、准确地识别和分析高级持续性威胁。
            系统集成了先进的特征提取、模型训练和 APT 归因技术，为您的网络安全保驾护航。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Modules Grid (2 rows x 4 columns)
    st.markdown("### 核心功能")
    
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-tasks"></i>
            <h3>任务管理</h3>
            <p style="font-size: 0.8rem; color: #aaa;">分析任务与预处理</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-chart-bar"></i>
            <h3>特征提取</h3>
            <p style="font-size: 0.8rem; color: #aaa;">多维特征可视化</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-project-diagram"></i>
            <h3>模型训练</h3>
            <p style="font-size: 0.8rem; color: #aaa;">基于关系感知图注意力网络的无监督学习</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-bullseye"></i>
            <h3>APT 归因</h3>
            <p style="font-size: 0.8rem; color: #aaa;">攻击组织溯源</p>
        </div>
        """, unsafe_allow_html=True)

    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-users"></i>
            <h3>团伙库</h3>
            <p style="font-size: 0.8rem; color: #aaa;">已知组织档案</p>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-database"></i>
            <h3>数据集</h3>
            <p style="font-size: 0.8rem; color: #aaa;">样本数据管理</p>
        </div>
        """, unsafe_allow_html=True)
    with c7:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-file-alt"></i>
            <h3>报告生成</h3>
            <p style="font-size: 0.8rem; color: #aaa;">分析报告导出</p>
        </div>
        """, unsafe_allow_html=True)
    with c8:
        st.markdown("""
        <div class="feature-module">
            <i class="fas fa-brain"></i>
            <h3>模型管理</h3>
            <p style="font-size: 0.8rem; color: #aaa;">算法模型训练</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # System Status (Real Data)
    st.markdown("### 系统资源概览")
    
    # Fetch real stats
    n_datasets = 0
    n_models = 0
    n_gangs = 0
    
    n_datasets = len(get_json("/api/datasets", timeout=1, default=[]))
    n_models = len(get_json("/api/models", timeout=1, default=[]))
    n_gangs = len(get_json("/api/gangs", timeout=1, default=[]))

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-card-icon"><i class="fas fa-database"></i></div><div class="stat-value">{n_datasets}</div><div class="stat-label">数据集</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-card-icon"><i class="fas fa-brain"></i></div><div class="stat-value">{n_models}</div><div class="stat-label">已训练模型</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-card-icon"><i class="fas fa-users"></i></div><div class="stat-value">{n_gangs}</div><div class="stat-label">情报库组织</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-card"><div class="stat-card-icon"><i class="fas fa-check-circle"></i></div><div class="stat-value">Ready</div><div class="stat-label">系统状态</div></div>', unsafe_allow_html=True)

def placeholder_page(title, icon, description="该模块正在开发中..."):
    """Renders a placeholder for pages that are under construction."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));">{icon}</div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">{title}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center; padding: 4rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem; color: rgba(255,255,255,0.2);">🚧</div>
        <h3 style="color: #00d4ff;">模块建设中</h3>
        <p style="color: #aaa;">{description}</p>
        <p style="font-size: 0.8rem; color: #666; margin-top: 1rem;">Coming Soon in v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main App --- 

def main():
    # Page configuration
    st.set_page_config(
        page_title="APT归因分析系统",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS and FontAwesome
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)
    
    if not check_password():
        return

    css_file = Path(__file__).parent / "css" / "styles.css"
    if css_file.exists():
        load_css(css_file)
    
    # Sidebar navigation menu
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));"><i class="fas fa-shield-alt"></i></div>
            <h2 style="margin:0; color: white; letter-spacing: 2px;">RGAPT</h2>
            <p style="color: #00d4ff; font-size: 0.8rem; letter-spacing: 1px; opacity: 0.8;">Semantic-Enhanced APT Attribution</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=[
                "主页", 
                "分析任务管理", 
                "数据集管理",
                "特征提取与可视化", 
                "模型训练", 
                "模型管理",
                "APT归因结果", 
                "报告生成与导出",
                "团伙库"
            ],
            icons=[
                "house", 
                "list-task", 
                "database", 
                "bar-chart", 
                "diagram-3", 
                "cpu",
                "bullseye", 
                "file-text",
                "people-group"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#00d4ff", "font-size": "16px"}, 
                "nav-link": {
                    "font-size": "14px", 
                    "text-align": "left", 
                    "margin": "5px 10px", 
                    "color": "#a0aab5",
                    "border-radius": "8px",
                    "transition": "all 0.3s ease",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 212, 255, 0.15)", 
                    "color": "#ffffff", 
                    "border-left": "3px solid #00d4ff"
                },
            }
        )

        st.divider()
        
        # Session State Management
        with st.expander("🔧 会话管理"):
            st.caption("会话数据仅保存在当前浏览器会话中。")
            if st.button("🗑️ 清空当前会话", type="secondary", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    # Page routing
    if selected == "主页":
        render_home()
    elif selected == "分析任务管理":
        render_upload()
    elif selected == "特征提取与可视化":
        render_features()
    elif selected == "模型训练":
        render_clustering()
    elif selected == "APT归因结果":
        render_attribution()
    elif selected == "团伙库":
        render_gangs()
    elif selected == "数据集管理":
        render_datasets()
    elif selected == "报告生成与导出":
        render_report()
    elif selected == "模型管理":
        render_models()

if __name__ == "__main__":
    main()
