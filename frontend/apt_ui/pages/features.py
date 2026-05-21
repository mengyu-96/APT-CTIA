import streamlit as st
import pandas as pd
import plotly.express as px
import requests

BACKEND_URL = "http://127.0.0.1:5001"

def get_datasets():
    try:
        response = requests.get(f"{BACKEND_URL}/api/datasets", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def get_dataset_stats(dataset_id):
    try:
        response = requests.get(f"{BACKEND_URL}/api/datasets/{dataset_id}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

# Entity Categories Mapping
ENTITY_CATEGORIES = {
    '静态特征': ['FILE_PATH', 'REGISTRY', 'FILE_EXT', 'HASH_MD5', 'HASH_SHA1', 'HASH_SHA256', 'SSL_CERT'],
    '动态特征': ['PROCESS', 'SERVICE', 'USER_AGENT', 'USER_ACCOUNT', 'MUTEX', 'PIPE', 'CMD_LINE'],
    '网络特征': ['IP', 'DOMAIN', 'URL', 'EMAIL', 'PORT', 'HOSTNAME', 'PROTOCOL'],
    '威胁情报': ['MITRE_TECH', 'CVE', 'CWE', 'MALWARE', 'TOOL', 'THREAT_INTEL_SOURCE', 'APT_GROUP', 'CAMPAIGN', 'ORG', 'INDUSTRY', 'COUNTRY', 'CITY']
}

def render_features():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-chart-bar"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">特征提取与可视化</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Feature Extraction & Visualization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Select Dataset
    datasets = get_datasets()
    if not datasets:
        st.info("暂无数据集。请先前往 [分析任务管理] 上传并预处理数据。")
        return

    # Filter only processed graphs
    processed_datasets = [d for d in datasets if "Graph" in d.get('type', '')]
    if not processed_datasets:
        st.warning("暂无已处理的图数据集。")
        return

    dataset_options = {d['name']: d['id'] for d in processed_datasets}
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_name = st.selectbox("选择数据集", list(dataset_options.keys()))
        with c2:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True) 
            if st.button("🔄 刷新数据"):
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    selected_id = dataset_options[selected_name]
    
    # Fetch Stats
    stats_data = get_dataset_stats(selected_id)
    
    if not stats_data:
        st.warning("该数据集暂无详细统计信息（可能是旧版本生成的，或预处理尚未完成）。请尝试重新预处理。")
        return
        
    # Check if stats is a list (old format) or dict (new aggregated format from api)
    # The API now returns the aggregated stats directly? 
    # Let's check api.py... get_dataset_stats returns json.load(f) which is LIST of per-graph stats
    
    # Wait, in datasets.py we are aggregating manually.
    # In features.py we were also aggregating manually.
    
    stats = stats_data # It is a list of dicts
        
    # Aggregate Stats
    total_entities = 0
    total_reports = len(stats)
    category_counts = {k: 0 for k in ENTITY_CATEGORIES.keys()}
    category_counts['其他'] = 0
    
    entity_type_counts = {}
    
    # Advanced: Node Degree Distribution (Mocked for now as we don't have full graph structure here)
    # But we have num_nodes and num_edges per graph
    avg_nodes = 0
    avg_edges = 0
    if total_reports > 0:
        avg_nodes = sum(s.get('num_nodes', 0) for s in stats) / total_reports
        avg_edges = sum(s.get('num_edges', 0) for s in stats) / total_reports
    
    for report in stats:
        counts = report.get('entity_counts', {})
        for etype, count in counts.items():
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + count
            total_entities += count
            
            # Categorize
            found = False
            for cat, types in ENTITY_CATEGORIES.items():
                if etype in types:
                    category_counts[cat] += count
                    found = True
                    break
            if not found:
                category_counts['其他'] += count

    st.markdown("###")
    
    # Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本报告数", total_reports)
    c2.metric("平均图节点数", f"{avg_nodes:.1f}")
    c3.metric("平均图边数", f"{avg_edges:.1f}")
    c4.metric("提取实体总数", total_entities)
    
    st.markdown("###")

    # Tabs
    tab_all, tab_detail, tab_insight = st.tabs(["总体分布", "详细统计", "深度洞察"])
    
    # Color Palette
    colors = ['#00E676', '#FF9800', '#9C27B0', '#2196F3', '#607D8B']

    with tab_all:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("特征类别分布")
        
        df_cat = pd.DataFrame({
            'Category': list(category_counts.keys()),
            'Count': list(category_counts.values())
        })
        df_cat = df_cat[df_cat['Count'] > 0] # Filter empty
        
        c_pie, c_bar = st.columns([1, 1])
        
        with c_pie:
            fig_pie = px.pie(df_cat, values='Count', names='Category', 
                             color_discrete_sequence=colors,
                             hole=0.4)
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c_bar:
            # Top 10 Entity Types
            sorted_types = sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            df_top = pd.DataFrame(sorted_types, columns=['Type', 'Count'])
            
            fig_bar = px.bar(df_top, x='Count', y='Type', orientation='h',
                             color='Count', color_continuous_scale='Viridis')
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                yaxis=dict(autorange="reversed"),
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_detail:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("各类特征详细统计")
        
        # Prepare detailed dataframe
        all_types = []
        for etype, count in entity_type_counts.items():
            cat = "其他"
            for c, ts in ENTITY_CATEGORIES.items():
                if etype in ts:
                    cat = c
                    break
            all_types.append({'Type': etype, 'Count': count, 'Category': cat})
            
        df_detail = pd.DataFrame(all_types).sort_values('Count', ascending=False)
        
        st.dataframe(
            df_detail, 
            column_config={
                "Type": "实体类型",
                "Count": st.column_config.ProgressColumn("数量", format="%d", min_value=0, max_value=max(df_detail['Count']) if not df_detail.empty else 100),
                "Category": "所属类别"
            },
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_insight:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("图结构洞察")
        
        # Plot Node vs Edge distribution
        if stats:
            df_graphs = pd.DataFrame(stats)
            
            # Enhance: Add APT Group distribution analysis if available
            if 'apt_group' in df_graphs.columns and df_graphs['apt_group'].notna().any():
                st.markdown("#### APT 组织图谱特征分布")
                # Box plot for nodes/edges per APT group
                c_box1, c_box2 = st.columns(2)
                with c_box1:
                    fig_box_n = px.box(df_graphs, x='apt_group', y='num_nodes', color='apt_group',
                                      title="各组织样本节点数分布")
                    fig_box_n.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
                    st.plotly_chart(fig_box_n, use_container_width=True)
                with c_box2:
                    fig_box_e = px.box(df_graphs, x='apt_group', y='num_edges', color='apt_group',
                                      title="各组织样本边数分布")
                    fig_box_e.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
                    st.plotly_chart(fig_box_e, use_container_width=True)
                st.divider()

            if 'num_nodes' in df_graphs.columns and 'num_edges' in df_graphs.columns:
                c1, c2 = st.columns(2)
                with c1:
                    fig_scatter = px.scatter(
                        df_graphs, 
                        x='num_nodes', 
                        y='num_edges',
                        color='apt_group' if 'apt_group' in df_graphs.columns else None,
                        hover_data=['report_id'],
                        title="节点数 vs 边数分布 (Graph Complexity)"
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#ffffff'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with c2:
                    # Histogram of node counts
                    fig_hist = px.histogram(
                        df_graphs, 
                        x='num_nodes',
                        nbins=20,
                        title="图规模分布 (Node Counts Histogram)"
                    )
                    fig_hist.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#ffffff',
                        bargap=0.1
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("数据中缺少图结构统计信息。")
        else:
            st.info("暂无数据。")
            
        st.markdown('</div>', unsafe_allow_html=True)
