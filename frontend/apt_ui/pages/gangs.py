import streamlit as st
import pandas as pd
import requests
import os
from apt_ui.services.api_client import get_json

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")

def get_gangs_data():
    return get_json("/api/gangs", timeout=3, default=[])

def render_gangs():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-users"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">团伙库</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Known APT Groups Database</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch Data from Backend
    if 'gangs_data' not in st.session_state:
        st.session_state['gangs_data'] = get_gangs_data()
    
    gangs_data = st.session_state['gangs_data']

    # Search and Filter
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("搜索组织 (名称/别名)", placeholder="例如: APT28, Lazarus...")
    with col_filter:
        # Dynamic filter options
        origins = sorted(list(set([g['origin'] for g in gangs_data]))) if gangs_data else []
        origin_filter = st.selectbox("来源筛选", ["全部"] + origins)

    # Filter Logic
    filtered_gangs = []
    if gangs_data:
        for gang in gangs_data:
            match_search = search_term.lower() in gang['id'].lower() or search_term.lower() in gang['aliases'].lower()
            match_origin = origin_filter == "全部" or origin_filter == gang['origin']
            
            if match_search and match_origin:
                filtered_gangs.append(gang)

    st.markdown("###")
    
    # Display Grid
    if not filtered_gangs:
        if not gangs_data:
             st.warning("未能加载团伙数据，请检查后端服务。")
        else:
             st.info("没有找到匹配的组织信息。")
    else:
        # 2 columns per row
        for i in range(0, len(filtered_gangs), 2):
            cols = st.columns(2)
            # Card 1
            with cols[0]:
                gang = filtered_gangs[i]
                render_gang_card(gang)
            
            # Card 2 (if exists)
            if i + 1 < len(filtered_gangs):
                with cols[1]:
                    gang = filtered_gangs[i+1]
                    render_gang_card(gang)
            else:
                with cols[1]:
                    pass # Empty column for alignment

def render_gang_card(gang):
    # Using expander for details to make it actionable
    with st.container():
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                <h3 style="color: #00d4ff; margin: 0;">{gang['id']}</h3>
                <span style="background: rgba(0, 212, 255, 0.1); color: #00d4ff; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid rgba(0, 212, 255, 0.3);">{gang['origin']}</span>
            </div>
            <p style="color: #aaa; font-size: 0.9rem; margin-bottom: 1rem;"><strong>别名:</strong> {gang['aliases']}</p>
            <p style="color: #ddd; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem;">{gang['description']}</p>
            
            <div style="background: rgba(0, 0, 0, 0.2); padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.85rem; color: #aaa;"><strong>🎯 目标行业:</strong> <span style="color: #fff;">{gang['targets']}</span></p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #aaa;"><strong>🛠️ 常用工具:</strong> <span style="color: #fff;">{gang['tools']}</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Interactive Elements (using Streamlit widgets inside the card logic)
        with st.expander("查看关联情报 (IOCs & Samples)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🛑 威胁指标 (IOCs)**")
                if gang.get('iocs'):
                    for ioc in gang['iocs']:
                        st.markdown(f"- `{ioc}`")
                else:
                    st.caption("暂无 IOC 数据")
            
            with c2:
                st.markdown("**🔗 系统内关联样本**")
                if gang.get('related_samples'):
                    for sample in gang['related_samples']:
                        st.markdown(f"- 📄 {sample}")
                    if st.button(f"查看 {gang['id']} 分析报告", key=f"btn_{gang['id']}"):
                         st.toast(f"正在生成 {gang['id']} 的详细归因报告...", icon="📄")
                else:
                    st.caption("暂无关联样本")

        st.markdown(f"""
            <div style="text-align: right; font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                最近活跃: {gang['last_seen']}
            </div>
        </div>
        """, unsafe_allow_html=True)
