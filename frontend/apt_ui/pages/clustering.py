import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import time
import json
from pathlib import Path

# Backend API endpoint
BACKEND_URL = "http://127.0.0.1:5001"

def get_datasets():
    try:
        response = requests.get(f"{BACKEND_URL}/api/datasets", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def load_artifact_image(path: str):
    if not path:
        return None

    try:
        p = Path(path)
        if p.exists() and p.is_file():
            return p.read_bytes()
    except Exception:
        pass

    try:
        r = requests.get(f"{BACKEND_URL}/api/artifact", params={"path": path}, timeout=5)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass

    return None

def list_training_tasks():
    try:
        r = requests.get(f"{BACKEND_URL}/api/tasks", timeout=5)
        if r.status_code != 200:
            return []
        tasks = r.json()
        return [t for t in tasks if t.get("type") == "train"]
    except Exception:
        return []

def format_task_created(created_value):
    if not created_value:
        return ""
    s = str(created_value)
    if len(s) >= 15 and s[8] == "-" and s[0:8].isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[9:11]}:{s[11:13]}:{s[13:15]}"
    return s

def delete_task(task_id: str):
    if not task_id:
        return False
    try:
        r = requests.delete(f"{BACKEND_URL}/api/tasks/{task_id}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def render_clustering():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-project-diagram"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">模型训练与分析</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Model Training & Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"""
    **数据存储位置信息：**
    - 📂 **数据集 (Datasets):** `backend/uploads`
    - ⚙️ **预处理结果 (Processed Data):** `backend/results_archive/processed_data`
    - 🧠 **模型文件 (Models):** `backend/results_archive/training_runs`
    - 📊 **分析结果 (Results):** `backend/results_archive/reports`
    """)

    # Fetch available datasets
    datasets = get_datasets()
    # Filter only processed graphs
    graph_datasets = [d for d in datasets if "Graph" in d.get('type', '')]
    
    # Check if preprocessing has been run in session
    if 'latest_preprocessing_run' in st.session_state:
        latest = st.session_state['latest_preprocessing_run']
        # Add to list if not present (optimistic)
        if latest.get('output_path'):
             # check if already in list
             path = latest.get('output_path')
             if not any(d['path'] == path for d in graph_datasets):
                 graph_datasets.insert(0, {'name': 'Latest Run', 'path': path, 'id': 'latest'})

    col_control, col_result = st.columns([1, 2], gap="large")

    with col_control:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("分析配置")
        
        with st.form("clustering_config"):
            # Dataset Selection
            if not graph_datasets:
                st.warning("无可用图数据集")
                st.caption("请先前往 [分析任务管理] 进行预处理")
                selected_dataset = None
            else:
                # Map name to full object
                options = {d['name']: d for d in graph_datasets}
                selected_name = st.selectbox("选择数据集", list(options.keys()))
                selected_dataset = options[selected_name]
            
            algorithm = st.selectbox("模型架构", ["GAT (图注意力网络)", "GCN (图卷积网络)", "GraphSAGE", "Transformer", "GIN", "Hybrid", "RGAT (Relation-aware GAT)"], index=0)
            
            # Dynamic config based on model
            use_temporal = False
            temporal_hidden = 128
            
            if "RGAT" in algorithm:
                st.info("ℹ️ RGAT (Relation-aware Graph Attention Network) 支持异构关系建模。")
                use_temporal = st.checkbox("启用时序建模 (Temporal Module - GRU)", value=False, help="在图特征提取后增加GRU层处理时序序列")
                if use_temporal:
                    temporal_hidden = st.number_input("时序隐藏层维度", 64, 512, 128)
            
            st.markdown("---")
            st.caption("训练参数")
            
            epochs = st.number_input("训练轮数 (Epochs)", 10, 5000, 100)
            lr = st.number_input("学习率 (LR)", 0.0001, 0.1, 0.001, format="%.4f")
            hidden_dim = st.select_slider("隐藏层维度", options=[64, 128, 256, 512], value=128)
            dropout = st.slider("Dropout", 0.0, 0.9, 0.5)
            
            submitted = st.form_submit_button("🚀 开始训练与分析", type="primary", use_container_width=True)
        
        if submitted:
            if not selected_dataset:
                st.error("请先选择有效的数据集")
            else:
                try:
                    model_type = algorithm.split(" ")[0]
                    config = {
                        "processed_data_path": selected_dataset['path'],
                        "dataset_id": selected_dataset['id'],
                        "dataset_name": selected_dataset['name'],
                        "model_type": model_type,
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": 32,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "seed": 42,
                        "use_temporal": use_temporal,
                        "temporal_hidden_dim": temporal_hidden
                    }
                    
                    with st.spinner("正在提交训练任务..."):
                        response = requests.post(f"{BACKEND_URL}/api/train", json=config, timeout=10)
                        if response.status_code == 202:
                            data = response.json()
                            task_id = data.get("task_id")
                            st.success(f"训练任务已提交！ID: {task_id}")
                        else:
                            st.error(f"提交失败: {response.text}")
                except Exception as e:
                    st.error(f"连接错误: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training History
        st.markdown("###")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("训练任务历史")

        tasks = list_training_tasks()
        has_running = any(t.get("status") in ["pending", "running"] for t in tasks)

        if not tasks:
            st.caption("暂无训练记录")

        for task in tasks:
            status = task.get("status", "unknown")
            status_color = "#4CAF50" if status == "completed" else "#F44336" if status == "failed" else "#2196F3"
            created = format_task_created(task.get("created"))
            model = task.get("model") or task.get("name") or "Train"
            dataset = task.get("dataset") or ""
            task_id = task.get("id", "")

            st.markdown(f"""
            <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                <div style="display:flex; justify-content:space-between; gap: 10px;">
                    <strong style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{model}</strong>
                    <span style="color:{status_color}; flex: 0 0 auto;">{status.upper()}</span>
                </div>
                <div style="font-size: 0.8rem; color: #aaa; margin-top: 2px;">
                    {dataset} {(" | " + created) if created else ""} {(" | ID: " + task_id[:8]) if task_id else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c_view, c_delete = st.columns(2, gap="small")
            with c_view:
                if status == "completed" and task_id:
                    if st.button("查看报告", key=f"view_train_{task_id}"):
                        try:
                            res = requests.get(f"{BACKEND_URL}/api/tasks/{task_id}", timeout=10)
                            if res.status_code == 200:
                                st.session_state['current_model_result'] = res.json().get('result')
                                st.rerun()
                            else:
                                st.warning("无法获取训练结果")
                        except Exception:
                            st.warning("无法获取训练结果")
                else:
                    st.button("查看报告", key=f"view_train_disabled_{task_id}", disabled=True)

            with c_delete:
                deletable = status in ["completed", "failed"]
                if st.button("删除", key=f"delete_train_{task_id}", disabled=not deletable):
                    ok = delete_task(task_id)
                    if ok:
                        st.toast("已删除训练记录", icon="🗑️")
                        st.rerun()
                    else:
                        st.warning("删除失败")

            if status == "failed":
                err = task.get("error")
                if err:
                    st.error(err)

        if has_running:
            time.sleep(2)
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("分析结果看板")
        
        if 'current_model_result' in st.session_state:
            result = st.session_state['current_model_result']
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            acc = result.get('test_accuracy', 0)
            f1 = result.get('test_f1_weighted', 0)
            topk = result.get('topk_accuracy', 0)
            
            c1.metric("准确率 (Accuracy)", f"{acc:.2%}", delta=None)
            c2.metric("F1 Score (Weighted)", f"{f1:.2%}", delta=None)
            if topk:
                c3.metric("Top-3 Accuracy", f"{topk:.2%}", delta=None)
            
            st.divider()
            
            # Config info
            with st.expander("训练配置详情"):
                st.json(result.get('config', {}))
            
            # Training History Visualization
            if result.get('history'):
                st.subheader("训练过程曲线")
                history = result['history']
                
                # Convert to DataFrame for easier plotting
                # Ensure lengths match
                min_len = min(len(history.get('train_loss', [])), len(history.get('val_loss', [])))
                df_loss = pd.DataFrame({
                    'Epoch': range(1, min_len + 1),
                    'Train Loss': history['train_loss'][:min_len],
                    'Val Loss': history['val_loss'][:min_len]
                })
                
                min_len_acc = min(len(history.get('train_acc', [])), len(history.get('val_acc', [])))
                df_acc = pd.DataFrame({
                    'Epoch': range(1, min_len_acc + 1),
                    'Train Acc': history['train_acc'][:min_len_acc],
                    'Val Acc': history['val_acc'][:min_len_acc]
                })
                
                tab_loss, tab_acc = st.tabs(["📉 损失函数 (Loss)", "📈 准确率 (Accuracy)"])
                
                with tab_loss:
                    fig_loss = px.line(df_loss, x='Epoch', y=['Train Loss', 'Val Loss'], 
                                       markers=True, color_discrete_sequence=["#FF5252", "#FFAB40"])
                    fig_loss.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#a0aab5'),
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_loss.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    fig_loss.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_loss, use_container_width=True)

                with tab_acc:
                    fig_acc = px.line(df_acc, x='Epoch', y=['Train Acc', 'Val Acc'], 
                                      markers=True, color_discrete_sequence=["#4CAF50", "#69F0AE"])
                    fig_acc.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#a0aab5'),
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_acc.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    fig_acc.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_acc, use_container_width=True)

            # Confusion Matrix Visualization
            if result.get('confusion_matrix_plot'):
                st.subheader("混淆矩阵")
                cm_path = result['confusion_matrix_plot']
                img = load_artifact_image(cm_path)
                if img:
                    st.image(img, caption="Confusion Matrix", use_column_width=True)
                else:
                    st.warning("无法加载混淆矩阵图像")
                    st.caption(cm_path)
            
            # Classification Report
            if result.get('classification_report'):
                with st.expander("详细分类报告 (Classification Report)"):
                     # Transpose for better readability
                     df_rep = pd.DataFrame(result['classification_report']).transpose()
                     # Style it
                     st.dataframe(df_rep, use_container_width=True)

            st.success(f"模型已保存至: `{result.get('model_path')}`")
            st.markdown("---")
            st.markdown("👉 **下一步操作：**")
            c_next1, c_next2 = st.columns(2)
            with c_next1:
                st.info("前往 [APT归因结果] 查看雷达图分析", icon="🎯")
            with c_next2:
                st.info("前往 [报告生成] 导出完整 PDF 报告", icon="📄")
            
        else:
            st.markdown("""
            <div style="text-align: center; padding: 5rem; color: #aaa;">
                <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.3;">📊</div>
                <p>请在左侧配置并启动训练任务，<br>完成后点击“查看报告”以显示结果。</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
