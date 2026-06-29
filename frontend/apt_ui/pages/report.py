import streamlit as st
import os
from apt_ui.services.api_client import get_json, request
from apt_ui.services.tasks import get_task_detail, list_tasks
from apt_ui.services import ui

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")

def render_report():
    ui.page_header("报告生成与导出", "Report Generation & Export", icon="fa-file-alt")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        with ui.section_card("报告配置", icon="fa-gear"):
            _render_report_config()

    with c2:
        with ui.section_card("报告预览与下载", icon="fa-file-pdf"):
            _render_report_preview()


def _render_report_config():
        # 1. Select Source Type (Training vs Attribution)
        source_type = st.radio("数据来源", ["归因任务 (Inference)", "训练任务 (Training)"], horizontal=True)
        
        selected_task = None
        task_id = None
        
        if source_type == "归因任务 (Inference)":
             # Fetch attribution results
             attr_res = get_json("/api/attribution_results", timeout=5, default=[])
             
             if attr_res:
                 options = {f"{r['created']} - {r['id']} ({r['total_samples']} samples)": r for r in attr_res}
                 selected_label = st.selectbox("选择归因结果", options=list(options.keys()))
                 if selected_label:
                     selected_task = options[selected_label]
                     task_id = selected_task['id'] # Use directory name as ID
             else:
                 st.info("暂无归因结果")
                 
        else:
            # Select from completed training tasks
            completed_tasks = [t for t in list_tasks(task_type="train", limit=40, ttl="default", timeout=5) if t.get('status') == 'completed']
            
            if completed_tasks:
                task_options = {f"{t['id'][:8]} - {t['model']} ({t['dataset']})": t for t in completed_tasks}
                selected_task_label = st.selectbox("选择训练任务", options=list(task_options.keys()))
                if selected_task_label:
                    selected_task = task_options[selected_task_label]
                    task_id = selected_task['id']
            else:
                 st.info("暂无已完成的训练任务")
        
        if not task_id:
             task_id = st.text_input("或手动输入任务/结果 ID", value="")
        
        report_type = st.selectbox("报告类型", ["完整归因报告", "技术细节报告", "高管摘要"])
        report_format = st.radio("导出格式", ["PDF", "HTML (即将支持)", "JSON (即将支持)"], index=0)
        
        include_sections = st.multiselect(
            "包含章节",
            ["执行摘要", "样本分析", "特征提取结果", "模型训练图谱", "归因结论", "IOCs"],
            default=["执行摘要", "归因结论", "IOCs"]
        )
        
        st.divider()
        if st.button("生成报告", type="primary", width="stretch"):
            if not task_id:
                st.error("请选择或输入任务 ID")
            else:
                with st.spinner("正在生成报告..."):
                    # Construct analysis results
                    analysis_results = {}
                    
                    if source_type == "归因任务 (Inference)" and selected_task:
                        # Use the data directly from selection (list_attribution_results returns summary)
                        dist = selected_task.get('label_distribution', {})
                        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                        top_attr = sorted_dist[0][0] if sorted_dist else "Unknown"
                        
                        attrs = []
                        total = selected_task.get('total_samples', 1)
                        for name, count in sorted_dist:
                             attrs.append({
                                 "name": name,
                                 "score": count / total,
                                 "risk": "High"
                             })
                        
                        analysis_results = {
                            "top_attribution": top_attr,
                            "attributions": attrs,
                            "total_samples": total,
                            "source_type": "inference"
                        }
                        
                    elif source_type == "训练任务 (Training)" and selected_task:
                         task_detail = get_task_detail(task_id) if task_id else {}
                         res = task_detail.get('result') or selected_task.get('result')
                         if res:
                            # Extract top attribution from classification report if available
                            top_attr = "Unknown"
                            attributions = []
                            if 'classification_report' in res:
                                # Find class with best f1-score
                                best_f1 = -1
                                report_dict = res['classification_report']
                                for cls_name, metrics in report_dict.items():
                                    if cls_name in ['accuracy', 'macro avg', 'weighted avg']: continue
                                    if isinstance(metrics, dict) and 'f1-score' in metrics:
                                        if metrics['f1-score'] > best_f1:
                                            best_f1 = metrics['f1-score']
                                            top_attr = cls_name
                                        
                                        # Add to attributions list
                                        attributions.append({
                                            "name": cls_name,
                                            "score": metrics.get('precision', 0),
                                            "risk": "High" if metrics.get('f1-score', 0) > 0.8 else "Medium"
                                        })
                            
                            analysis_results = {
                                "top_attribution": top_attr,
                                "attributions": attributions,
                                "confusion_matrix_plot": res.get('confusion_matrix_plot'),
                                "history_plot": res.get('history_plot'),
                                "source_type": "training"
                            }

                    try:
                        response = request(
                            "POST",
                            "/api/generate_report",
                            json_body={
                                "task_id": task_id,
                                "analysis_results": analysis_results,
                            },
                            timeout=30,
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['report_generated'] = result
                            st.success("报告生成成功！")
                        else:
                            st.error(f"生成失败: {response.text}")
                    except Exception as e:
                        st.error(f"连接后端失败: {e}")


def _render_report_preview():
        if st.session_state.get('report_generated'):
            report_info = st.session_state['report_generated']
            report_url = f"{BACKEND_URL}{report_info['report_url']}"

            st.success("✅ PDF 报告已就绪")
            st.markdown(f"**文件路径:** `{report_info['report_path']}`")

            st.markdown(
                f"""
                <div style="text-align: center; padding: 1.5rem 0;">
                    <a href="{report_url}" target="_blank" style="text-decoration: none;">
                        <span style="background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%);
                              color: #00121d; font-weight: 700; padding: 0.7rem 1.6rem;
                              border-radius: 8px; display: inline-block;">
                            📥 点击下载 PDF 报告
                        </span>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("PDF 在新标签页打开；如需嵌入预览，请下载后本地查看。")
        else:
            ui.empty_state("请在左侧配置并点击「生成报告」按钮。", icon="fa-file-pdf")
