import streamlit as st
import requests
import time
import json

# Backend API endpoint
BACKEND_URL = "http://127.0.0.1:5001"

def render_upload() -> None:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-tasks"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">分析任务管理</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Analysis Task Management & Preprocessing</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_create, col_history = st.columns([1, 1], gap="large")
    
    with col_create:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 创建新分析任务")
        st.caption("上传原始日志文件 (TXT/JSON/PDF) 进行预处理。")
        
        with st.form("create_task_form", clear_on_submit=True):
            # Dataset Name Input
            dataset_name = st.text_input(
                "数据集名称 (可选)", 
                help="指定输出数据集的文件夹名称。留空将使用时间戳。"
            )

            # File Uploader
            uploaded_files = st.file_uploader(
                "选择文件", 
                accept_multiple_files=True, 
                type=['txt', 'json', 'pdf'],
                help="支持多个文件上传"
            )
            
            submitted = st.form_submit_button("启动预处理任务", type="primary", use_container_width=True)
            
        if submitted:
            if not uploaded_files:
                st.error("请先选择要上传的文件！")
            else:
                try:
                    # Prepare files for upload
                    files_payload = []
                    for file in uploaded_files:
                        files_payload.append(('files', (file.name, file.getvalue(), file.type)))
                    
                    data_payload = {}
                    if dataset_name:
                        data_payload['output_name'] = dataset_name

                    with st.spinner("正在上传文件并提交任务..."):
                        response = requests.post(
                            f"{BACKEND_URL}/api/preprocess", 
                            files=files_payload, 
                            data=data_payload,
                            timeout=60
                        )
                        
                        if response.status_code == 202:
                            data = response.json()
                            task_id = data.get("task_id")
                            st.success(f"任务提交成功！Task ID: {task_id}")
                            
                            # Add to session state history
                            if 'task_history' not in st.session_state:
                                st.session_state['task_history'] = []
                            
                            display_name = f"{dataset_name} ({len(uploaded_files)} files)" if dataset_name else f"Batch Upload ({len(uploaded_files)} files)"
                            
                            st.session_state['task_history'].insert(0, {
                                "id": task_id,
                                "name": display_name,
                                "status": "pending",
                                "created": time.strftime("%H:%M:%S"),
                                "files": len(uploaded_files)
                            })
                            st.rerun()
                        else:
                            st.error(f"任务提交失败: {response.text}")
                            
                except requests.exceptions.ConnectionError:
                    st.error("无法连接到后端服务，请确认后端已启动。")
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📋 任务列表")
        
        # Fetch tasks from backend (persistent history)
        try:
            res = requests.get(f"{BACKEND_URL}/api/tasks", timeout=3)
            if res.status_code == 200:
                tasks = res.json()
                # Filter only preprocess/upload tasks if needed, or show all
                # Let's show preprocess tasks primarily here
                tasks = [t for t in tasks if t.get('type') in ['preprocess', 'upload']]
            else:
                tasks = []
        except:
             tasks = []
             
        if not tasks:
            st.info("暂无任务记录")
        
        # Auto-refresh logic for running tasks
        has_running = False
        
        for i, task in enumerate(tasks):
            # Check status if running (though list endpoint might be stale if not polled frequently, 
            # individual status check is better for progress bar)
            if task['status'] in ['pending', 'running']:
                has_running = True
                try:
                    res = requests.get(f"{BACKEND_URL}/api/tasks/{task['id']}", timeout=2)
                    if res.status_code == 200:
                        remote_data = res.json()
                        task['progress'] = remote_data.get('progress', 0)
                        task['message'] = remote_data.get('message', '')
                        task['status'] = remote_data.get('status')
                        if task['status'] == 'failed':
                            task['error'] = remote_data.get('error')
                except:
                    pass
            
            # Render Task Item
            status_color = "#4CAF50" if task['status'] == "completed" else "#F44336" if task['status'] == "failed" else "#2196F3"
            icon = "✅" if task['status'] == "completed" else "❌" if task['status'] == "failed" else "🔄"
            
            # Date formatting
            created_str = task.get('created', '')
            if created_str:
                # Simple formatting if needed, or just display
                pass

            with st.container():
                c1, c2 = st.columns([3, 2])
                with c1:
                    # Name logic
                    name = task.get('name', 'Unnamed Task')
                    if name == 'preprocess task':
                         name = f"Batch Upload ({task.get('files', 0)} files)"
                    
                    st.markdown(f"**{name}**")
                    st.caption(f"ID: `{task['id']}` | {created_str}")
                    if task.get('error'):
                        st.error(task['error'])
                    
                    # Progress Bar
                    if task['status'] == 'running':
                        progress = task.get('progress', 0)
                        try:
                            progress_val = int(progress) / 100.0
                        except:
                            progress_val = 0.0
                        
                        st.progress(max(0.0, min(1.0, progress_val)))
                        msg = task.get('message', 'Processing...')
                        st.caption(f"Running: {msg} ({progress}%)")

                with c2:
                    st.markdown(f"<span style='color:{status_color}'>{icon} {task['status'].upper()}</span>", unsafe_allow_html=True)
                    if task['status'] == "completed":
                        st.caption("数据已就绪")
                            
                st.divider()
        
        if has_running:
            time.sleep(2)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
