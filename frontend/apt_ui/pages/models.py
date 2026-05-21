import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

BACKEND_URL = "http://127.0.0.1:5001"

def get_models():
    try:
        response = requests.get(f"{BACKEND_URL}/api/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"无法连接到后端: {e}")
        return []

def rename_model(model_id, new_name):
    try:
        response = requests.put(f"{BACKEND_URL}/api/models/{model_id}/rename", json={"new_name": new_name}, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def delete_model(model_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/models/{model_id}", timeout=5)
        if response.status_code == 200:
            return True, "删除成功"
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def render_models():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-brain"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">模型管理</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Model Registry & Performance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch Models
    if 'models_list' not in st.session_state:
        st.session_state['models_list'] = get_models()
    
    models = st.session_state['models_list']

    # Refresh Button
    if st.button("🔄 刷新列表"):
        st.session_state['models_list'] = get_models()
        st.rerun()

    if not models:
         st.info("暂无已训练的模型。请前往 [模型训练] 页面进行模型训练。")
         return

    # Model Performance Chart
    st.subheader("模型性能对比")
    df_perf = pd.DataFrame(models)
    
    col_chart, col_info = st.columns([2, 1])
    
    with col_chart:
        if not df_perf.empty:
            fig = px.bar(df_perf, x='name', y=['accuracy', 'f1_score'], barmode='group',
                        color_discrete_sequence=['#00d4ff', '#0099cc'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                xaxis_title="模型名称",
                yaxis_title="分数",
                legend_title="指标",
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 最新模型")
        # Just pick the first one as they are sorted by date desc
        latest_model = models[0] if models else None
        if latest_model:
            st.metric("模型名称", latest_model['name'])
            st.metric("准确率 (Accuracy)", f"{latest_model['accuracy']:.3f}")
            st.metric("F1 分数", f"{latest_model['f1_score']:.3f}")
            st.caption(f"创建时间: {latest_model['created']}")
        else:
            st.warning("无模型")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("###")
    st.subheader("模型列表")

    # Filter/Search
    c1, c2 = st.columns([3, 1])
    with c1:
        search_term = st.text_input("搜索模型", placeholder="模型名称...")
    with c2:
        status_filter = st.selectbox("状态筛选", ["全部", "Completed"]) # Only Completed for now

    # Filter Logic
    filtered_models = []
    for m in models:
        match_search = search_term.lower() in m['name'].lower()
        match_status = status_filter == "全部" or status_filter == m['status']
        if match_search and match_status:
            filtered_models.append(m)

    # Detailed Analysis View (Drill-down)
    if 'selected_model_id' in st.session_state and st.session_state['selected_model_id']:
        sel_model = next((m for m in filtered_models if m['id'] == st.session_state['selected_model_id']), None)
        if sel_model:
            with st.expander(f"📊 {sel_model['name']} - 详细评估报告", expanded=True):
                mc1, mc2 = st.columns([1, 1])
                with mc1:
                    st.markdown("#### 混淆矩阵数据")
                    # Check if confusion matrix data is available in classification_report or similar
                    # Backend currently returns classification_report dict. 
                    # We might not have raw CM unless saved. 
                    # But we can visualize the per-class precision/recall/f1 from classification_report.
                    
                    report = sel_model.get('classification_report', {})
                    if report:
                        # Convert to DataFrame for heatmap-like display
                        # Rows: Classes, Cols: Precision, Recall, F1, Support
                        class_data = []
                        for k, v in report.items():
                            if k not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(v, dict):
                                class_data.append({
                                    'Class': k,
                                    'Precision': v['precision'],
                                    'Recall': v['recall'],
                                    'F1-Score': v['f1-score'],
                                    'Support': v['support']
                                })
                        
                        if class_data:
                            df_cls = pd.DataFrame(class_data)
                            fig_cls = px.bar(df_cls, x='Class', y=['Precision', 'Recall', 'F1-Score'],
                                            barmode='group', title="各类别性能指标")
                            fig_cls.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#ffffff'
                            )
                            st.plotly_chart(fig_cls, use_container_width=True)
                        else:
                            st.info("无详细分类报告数据")
                    else:
                        st.info("无分类报告")

                with mc2:
                    st.markdown("#### 综合指标雷达图")
                    # Radar chart for overall metrics
                    if report:
                        macro = report.get('macro avg', {})
                        weighted = report.get('weighted avg', {})
                        acc = report.get('accuracy', 0)
                        
                        categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1']
                        values = [
                            acc,
                            macro.get('precision', 0),
                            macro.get('recall', 0),
                            macro.get('f1-score', 0),
                            weighted.get('f1-score', 0)
                        ]
                        
                        df_radar = pd.DataFrame(dict(
                            r=values,
                            theta=categories
                        ))
                        fig_radar = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                        fig_radar.update_traces(fill='toself')
                        fig_radar.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            polar=dict(
                                bgcolor='rgba(0,0,0,0)',
                                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                                angularaxis=dict(color='white')
                            ),
                            font_color='#ffffff',
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                if st.button("关闭详情", key="close_detail"):
                    st.session_state['selected_model_id'] = None
                    st.rerun()
            st.divider()

    # Table Header
    cols = st.columns([3, 2, 2, 1, 1, 1, 2, 3])
    cols[0].markdown("**模型名称**")
    cols[1].markdown("**类型**")
    cols[2].markdown("**训练数据集**")
    cols[3].markdown("**准确率**")
    cols[4].markdown("**F1**")
    cols[5].markdown("**轮数**")
    cols[6].markdown("**状态**")
    cols[7].markdown("**操作**")
    st.divider()

    for idx, model in enumerate(filtered_models):
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([3, 2, 2, 1, 1, 1, 2, 3])
        
        with c1:
            st.markdown(f"**{model['name']}**")
            # Add View Details button/link
            if st.button("👁️ 详情", key=f"view_{idx}", help="查看详细评估指标"):
                st.session_state['selected_model_id'] = model['id']
                st.rerun()
        with c2:
            st.caption(model['type'])
        with c3:
            st.caption(model.get('dataset_name', model.get('dataset_id', 'Unknown')))
        with c4:
            st.markdown(f"{model['accuracy']:.3f}")
        with c5:
            st.markdown(f"{model['f1_score']:.3f}")
        with c6:
            st.markdown(model['epochs'])
        with c7:
            status_color = "#4CAF50" if model['status'] == "Completed" else "#9E9E9E"
            st.markdown(f"<span style='color:{status_color}; font-weight:bold;'>● {model['status']}</span>", unsafe_allow_html=True)
        with c8:
            col_a, col_b = st.columns(2)
            with col_a:
                # Edit/Rename
                with st.expander("✏️"):
                    new_name = st.text_input("重命名", value=model['name'], key=f"ren_model_{idx}")
                    if st.button("确认", key=f"ren_model_btn_{idx}"):
                         ok, res = rename_model(model['id'], new_name)
                         if ok:
                             st.success("成功")
                             st.session_state['models_list'] = get_models()
                             time.sleep(1)
                             st.rerun()
                         else:
                             st.error(f"失败: {res}")

            with col_b:
                if st.button("🗑️", key=f"del_m_{idx}"):
                    success, msg = delete_model(model['id'])
                    if success:
                        st.toast(f"模型 {model['name']} 已删除", icon="🗑️")
                        st.session_state['models_list'] = [m for m in st.session_state['models_list'] if m['id'] != model['id']]
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"删除失败: {msg}")
        
        st.markdown("---")
