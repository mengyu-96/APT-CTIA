
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import json
import os
from apt_ui.services.api_client import clear_api_cache, get_json

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")

def get_models():
    return get_json("/api/models", timeout=5, default=[])

def get_datasets():
    return get_json("/api/datasets", timeout=2, default=[])

def get_attribution_results():
    return get_json("/api/attribution_results", timeout=5, default=[])

def get_attribution_detail(result_id):
    return get_json(f"/api/attribution_results/{result_id}", timeout=5, default=None, ttl="slow")

def get_server_tasks():
    """Fetch active inference tasks from backend"""
    tasks = get_json("/api/tasks", timeout=2, default=[], ttl="fast")
    # Filter for inference tasks and sort by creation time (newest first)
    return [t for t in tasks if t.get('type') == 'inference']

@st.cache_data(ttl=300, show_spinner=False)
def _attribution_distribution_df(dist):
    return pd.DataFrame(list(dist.items()), columns=['APT Group', 'Count'])

@st.cache_data(ttl=300, show_spinner=False)
def _format_attribution_results(raw_results):
    formatted_results = []
    for res in raw_results:
        top3_str = ""
        if res.get('top3'):
            top3_items = [f"{item.get('label', 'Unknown')} ({item.get('score', 0):.2%})" for item in res.get('top3', [])]
            top3_str = ", ".join(top3_items)

        formatted_results.append({
            "report_id": res.get('report_id'),
            "predicted_label": res.get('predicted_label'),
            "confidence": res.get('confidence'),
            "top3": top3_str
        })
    return pd.DataFrame(formatted_results)

def render_explanation_panel(sample_result):
    explanation = sample_result.get('explanation') or {}
    if not explanation:
        st.info("该样本暂无可解释证据。")
        return

    decision_mode = explanation.get('decision_mode', {})
    c1, c2, c3 = st.columns(3)
    c1.metric("预测归因", sample_result.get('predicted_label', 'Unknown'))
    c2.metric("置信度", f"{sample_result.get('confidence', 0):.2%}")
    c3.metric("主导证据", decision_mode.get('dominant_signal', 'unknown'))

    if decision_mode:
        st.caption(
            f"语义分支 {decision_mode.get('semantic_gate', 0):.2%} | "
            f"结构分支 {decision_mode.get('structural_gate', 0):.2%} | "
            f"{decision_mode.get('description', '')}"
        )

    key_nodes = explanation.get('key_nodes', [])
    if key_nodes:
        st.markdown("**关键实体节点**")
        node_df = pd.DataFrame(key_nodes)
        st.dataframe(node_df, use_container_width=True, hide_index=True)
        top_nodes = node_df.head(8).copy()
        if 'text' in top_nodes.columns and 'attention' in top_nodes.columns:
            fig_nodes = px.bar(
                top_nodes,
                x='text',
                y='attention',
                color='type' if 'type' in top_nodes.columns else None,
                title='Top Evidence Nodes',
            )
            fig_nodes.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_nodes, use_container_width=True)

    key_edges = explanation.get('key_edges', [])
    if key_edges:
        st.markdown("**关键证据边**")
        st.dataframe(pd.DataFrame(key_edges), use_container_width=True, hide_index=True)

    evidence_paths = explanation.get('evidence_paths', [])
    if evidence_paths:
        st.markdown("**核心决策路径**")
        for item in evidence_paths[:5]:
            path_text = " -> ".join(item.get('path_texts', []))
            st.markdown(f"- `{path_text}`")

    mitre_attack = explanation.get('mitre_attack', {})
    techniques = mitre_attack.get('techniques', [])
    if techniques:
        st.markdown("**MITRE ATT&CK 摘要**")
        tech_df = pd.DataFrame(techniques)
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
        tactics = mitre_attack.get('top_tactics', [])
        if tactics:
            tactic_df = pd.DataFrame(tactics)
            fig_tactics = px.bar(
                tactic_df,
                x='name',
                y='score',
                title='Top ATT&CK Tactics',
                color='score',
                color_continuous_scale='Blues'
            )
            fig_tactics.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_tactics, use_container_width=True)

def render_attribution():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-bullseye"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">APT 归因分析</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">APT Attribution & Inference</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_new, tab_history = st.tabs(["🚀 新建归因任务", "📜 历史归因记录"])

    # --- Tab 1: New Task ---
    with tab_new:
        col_form, col_status = st.columns([1, 1], gap="large")
        
        with col_form:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("任务配置")
            
            with st.form("attribution_form"):
                # 1. Select Model
                models = get_models()
                model_options = {m['name']: m['id'] for m in models} if models else {}
                
                if not model_options:
                    st.warning("无可用模型，请先进行模型训练训练。")
                    selected_model_id = None
                else:
                    selected_model_name = st.selectbox("选择归因模型", list(model_options.keys()))
                    selected_model_id = model_options[selected_model_name]

                # 2. Select Dataset
                datasets = get_datasets()
                # Filter only graph datasets
                graph_datasets = [d for d in datasets if "Graph" in d.get('type', '')]
                dataset_options = {d['name']: d['id'] for d in graph_datasets} if graph_datasets else {}
                
                if not dataset_options:
                    st.warning("无可用数据集，请先进行数据预处理。")
                    selected_dataset_id = None
                else:
                    selected_dataset_name = st.selectbox("选择待归因数据集", list(dataset_options.keys()))
                    selected_dataset_id = dataset_options[selected_dataset_name]
                
                st.markdown("---")
                submitted = st.form_submit_button("开始归因", type="primary", use_container_width=True)
                
            if submitted:
                if not selected_model_id or not selected_dataset_id:
                    st.error("请选择有效的模型和数据集")
                else:
                    payload = {
                        "model_id": selected_model_id,
                        "dataset_id": selected_dataset_id
                    }
                    try:
                        with st.spinner("正在提交归因任务..."):
                            res = requests.post(f"{BACKEND_URL}/api/inference", json=payload, timeout=5)
                            if res.status_code == 202:
                                clear_api_cache()
                                data = res.json()
                                task_id = data.get("task_id")
                                st.success(f"任务已提交! ID: {task_id}")
                                
                                # Add to session state for tracking
                                if 'attribution_tasks' not in st.session_state:
                                    st.session_state['attribution_tasks'] = []
                                st.session_state['attribution_tasks'].insert(0, {
                                    "id": task_id,
                                    "status": "pending",
                                    "model": selected_model_name,
                                    "dataset": selected_dataset_name,
                                    "created": time.strftime("%H:%M:%S")
                                })
                                st.rerun()
                            else:
                                st.error(f"提交失败: {res.text}")
                    except Exception as e:
                        st.error(f"连接错误: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col_status:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("正在进行的任务")
            
            # Fetch tasks directly from backend to ensure persistence across refreshes
            active_tasks = get_server_tasks()
            
            if not active_tasks:
                st.info("暂无任务记录")
            else:
                # Auto-refresh if there are running tasks
                if any(t['status'] in ['pending', 'running'] for t in active_tasks):
                    time.sleep(2)
                    st.rerun()
                    
                for task in active_tasks:
                    status = task.get('status', 'unknown')
                    status_color = "#4CAF50" if status == "completed" else "#F44336" if status == "failed" else "#2196F3"
                    created_time = task.get('created', '')
                    
                    c_task_info, c_task_del = st.columns([6, 1])
                    
                    with c_task_info:
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px; border-left: 3px solid {status_color};">
                            <div style="display:flex; justify-content:space-between;">
                                <strong>{task.get('name', 'Attribution Task')}</strong>
                                <span style="color:{status_color}">{status.upper()}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #aaa;">
                                ID: {task['id'][:8]}... | Time: {created_time}
                            </div>
                            {f'<div style="color: #ff6b6b; font-size: 0.8rem; margin-top:5px; word-wrap: break-word;">Error: {task.get("error")}</div>' if task.get('error') else ''}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c_task_del:
                        st.write("") # Spacer for alignment
                        if st.button("🗑️", key=f"del_task_{task['id']}", help="删除任务记录"):
                            try:
                                resp = requests.delete(f"{BACKEND_URL}/api/tasks/{task['id']}", timeout=2)
                                if resp.status_code == 200:
                                    clear_api_cache()
                                    st.toast("删除成功", icon="✅")
                                    time.sleep(0.5)
                                    st.rerun()
                                elif resp.status_code == 405:
                                    st.error("后端接口未更新，请重启后端服务。")
                                else:
                                    st.error(f"删除失败: {resp.status_code}")
                            except Exception as e:
                                st.error(f"请求失败: {e}")
                    
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: History & Results ---
    with tab_history:
        results = get_attribution_results()
        
        if not results:
            st.info("暂无历史归因记录")
        else:
            # Layout: List on left, Detail on right
            c_list, c_detail = st.columns([1, 3], gap="large")
            
            with c_list:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("记录列表")
                
                # Radio button to select result
                options = {f"{r['created']} ({r['total_samples']} samples)": r['id'] for r in results}
                selected_label = st.radio("选择记录", list(options.keys()), label_visibility="collapsed")
                selected_result_id = options[selected_label]
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_detail:
                if selected_result_id:
                    detail = get_attribution_detail(selected_result_id)
                    
                    if detail:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        
                        # Title row with delete button
                        c_head_title, c_head_del = st.columns([5, 1])
                        with c_head_title:
                            st.subheader(f"归因结果详情: {selected_label}")
                        with c_head_del:
                             if st.button("🗑️ 删除", type="secondary", key="del_attr_rec", help="永久删除该条归因记录"):
                                 try:
                                     del_res = requests.delete(f"{BACKEND_URL}/api/attribution_results/{selected_result_id}")
                                     if del_res.status_code == 200:
                                         clear_api_cache()
                                         st.toast("删除成功!", icon="✅")
                                         time.sleep(1)
                                         st.rerun()
                                     else:
                                         st.error(f"删除失败: {del_res.text}")
                                 except Exception as e:
                                     st.error(f"请求错误: {e}")
                        
                        # 1. Summary Metrics
                        total = detail.get('total_samples', 0)
                        dist = detail.get('label_distribution', {})
                        
                        # Pie Chart
                        if dist:
                            df_dist = _attribution_distribution_df(dist)
                            fig = px.pie(df_dist, values='Count', names='APT Group', title='归因分布 (Attribution Distribution)',
                                         color_discrete_sequence=px.colors.sequential.RdBu)
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#ffffff')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        # 2. Detailed Table
                        st.subheader("样本详情")
                        raw_results = detail.get('results', [])
                        if raw_results:
                            df_res = _format_attribution_results(raw_results)
                            
                            st.dataframe(
                                df_res[['report_id', 'predicted_label', 'confidence', 'top3']],
                                column_config={
                                    "report_id": "样本ID (Report)",
                                    "predicted_label": "预测归因 (Predicted APT)",
                                    "confidence": st.column_config.ProgressColumn(
                                        "置信度 (Confidence)",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                    "top3": "Top-3 候选 (Candidates)"
                                },
                                use_container_width=True,
                                height=400
                            )

                            st.divider()
                            st.subheader("可解释归因")
                            sample_options = [res.get('report_id') for res in raw_results]
                            selected_sample_id = st.selectbox("选择样本查看证据链", sample_options, key="explain_sample_id")
                            selected_sample = next((res for res in raw_results if res.get('report_id') == selected_sample_id), None)
                            if selected_sample:
                                render_explanation_panel(selected_sample)
                            
                            # Action Buttons
                            c_export, c_report = st.columns([1, 1])
                            
                            with c_export:
                                csv = df_res.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 导出结果 (CSV)",
                                    csv,
                                    "attribution_results.csv",
                                    "text/csv",
                                    key='download-csv',
                                    use_container_width=True
                                )
                                
                            with c_report:
                                if st.button("📄 生成 PDF 研判报告", use_container_width=True, key="gen_pdf_btn"):
                                    with st.spinner("正在生成报告..."):
                                        try:
                                            # We need to construct analysis_results payload or let backend reconstruct it
                                            # Ideally backend reconstructs from task_id or result_id
                                            # Since result_id in this context is directory name, which often contains task_id.
                                            # Let's pass the result object as analysis_results to be safe
                                            
                                            # Construct attribution summary for report
                                            dist = detail.get('label_distribution', {})
                                            total = detail.get('total_samples', 1)
                                            attrs = []
                                            if dist:
                                                for name, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                                                    attrs.append({
                                                        "name": name,
                                                        "score": count / total,
                                                        "risk": "High" if count/total > 0.5 else "Medium"
                                                    })
                                            
                                            payload = {
                                                "task_id": selected_result_id, # Use result ID as task ID
                                                "analysis_results": {
                                                    "top_attribution": attrs[0]['name'] if attrs else "Unknown",
                                                    "attributions": attrs,
                                                    "total_samples": total,
                                                    "graph_data": None, # Could fetch sample graph if needed
                                                    "iocs": {} # Could extract IOCs from results if available
                                                }
                                            }
                                            
                                            gen_res = requests.post(f"{BACKEND_URL}/api/generate_report", json=payload, timeout=30)
                                            if gen_res.status_code == 200:
                                                rep_data = gen_res.json()
                                                report_url = f"{BACKEND_URL}{rep_data['report_url']}"
                                                st.success("报告生成成功！")
                                                st.markdown(f'<a href="{report_url}" target="_blank" style="display:inline-block; padding:0.5em 1em; background:#4CAF50; color:white; border-radius:4px; text-decoration:none;">⬇️ 点击下载 PDF 报告</a>', unsafe_allow_html=True)
                                            else:
                                                st.error(f"生成失败: {gen_res.text}")
                                        except Exception as e:
                                            st.error(f"请求错误: {e}")
                        else:
                            st.info("无样本详情数据")
                            
                        st.markdown('</div>', unsafe_allow_html=True)
