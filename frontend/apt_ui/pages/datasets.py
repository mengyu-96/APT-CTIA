
import streamlit as st
import pandas as pd
import requests
import time

BACKEND_URL = "http://127.0.0.1:5001"

def get_datasets():
    try:
        response = requests.get(f"{BACKEND_URL}/api/datasets", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"无法连接到后端: {e}")
        return []

def get_raw_datasets():
    try:
        response = requests.get(f"{BACKEND_URL}/api/raw_datasets", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        return []

def get_raw_files(path):
    try:
        response = requests.get(f"{BACKEND_URL}/api/raw_datasets/files", params={"path": path}, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def rename_dataset(dataset_id, new_name):
    try:
        response = requests.put(f"{BACKEND_URL}/api/datasets/{dataset_id}/rename", json={"new_name": new_name}, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def delete_dataset(dataset_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/datasets/{dataset_id}", timeout=5)
        if response.status_code == 200:
            return True, "删除成功"
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def delete_graph(dataset_id, report_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/datasets/{dataset_id}/graphs/{report_id}", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def split_dataset(dataset_id, train_ratio, val_ratio):
    try:
        test_ratio = 1.0 - train_ratio - val_ratio
        payload = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio
        }
        response = requests.post(f"{BACKEND_URL}/api/datasets/{dataset_id}/split", json=payload, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def start_preprocessing(raw_path):
    try:
        payload = {"raw_dataset_path": raw_path}
        response = requests.post(f"{BACKEND_URL}/api/preprocess", json=payload, timeout=5)
        if response.status_code == 202:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def create_raw_dataset(name):
    try:
        response = requests.post(f"{BACKEND_URL}/api/raw_datasets", json={"name": name}, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def delete_raw_dataset(name):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/raw_datasets/{name}", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def rename_raw_dataset(name, new_name):
    try:
        response = requests.put(f"{BACKEND_URL}/api/raw_datasets/{name}", json={"new_name": new_name}, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def copy_raw_dataset(name, new_name):
    try:
        response = requests.post(f"{BACKEND_URL}/api/raw_datasets/{name}/copy", json={"new_name": new_name}, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def delete_raw_file(dataset_name, filename):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/raw_datasets/{dataset_name}/files/{filename}", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def copy_raw_file(dataset_name, filename, target_dataset):
    try:
        response = requests.post(f"{BACKEND_URL}/api/raw_datasets/{dataset_name}/files/{filename}/copy", json={"target_dataset": target_dataset}, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def upload_raw_files(dataset_name, files):
    try:
        files_payload = [('files', (f.name, f.getvalue(), f.type)) for f in files]
        response = requests.post(
            f"{BACKEND_URL}/api/raw_datasets/{dataset_name}/upload",
            files=files_payload,
            timeout=120
        )
        try:
            res_json = response.json()
        except ValueError:
            # Failed to parse JSON, return text (e.g. HTML error)
            error_msg = response.text[:200] if response.text else "Empty response"
            return False, f"Server Error ({response.status_code}): {error_msg}"
            
        if response.status_code == 200:
            return True, res_json
        else:
            return False, res_json.get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def format_bytes(size):
    try:
        size = int(size)
        power = 1024
        n = 0
        power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        while size > power:
            size /= power
            n += 1
        return f"{size:.1f} {power_labels[n]}"
    except:
        return "0 B"

def render_raw_file_explorer():
    # Initialize Session State for Explorer
    if 'explorer_path' not in st.session_state:
        st.session_state['explorer_path'] = [] # List of dataset names. Empty = Root.
    if 'clipboard' not in st.session_state:
        st.session_state['clipboard'] = [] # List of dicts: {'type': 'file'|'dir', 'source_ds': str, 'name': str}
    if 'clipboard_op' not in st.session_state:
        st.session_state['clipboard_op'] = 'copy' # or 'cut'
    
    current_path = st.session_state['explorer_path']
    is_root = len(current_path) == 0
    
    # --- Breadcrumbs & Header ---
    path_str = " 🏠 Home"
    if not is_root:
        path_str += f" / 📂 {current_path[0]}"
    
    st.markdown(f"### {path_str}")
    
    # --- Toolbar ---
    col_t1, col_t2, col_t3, col_t4, col_t5, col_t6 = st.columns([1, 1, 1, 1, 1, 3])
    
    with col_t1:
        if not is_root:
            if st.button("⬆️ 上一级", use_container_width=True):
                st.session_state['explorer_path'].pop()
                st.rerun()
        else:
            if st.button("🔄 刷新", use_container_width=True):
                st.session_state['raw_datasets'] = get_raw_datasets()
                st.rerun()

    # --- Content Area ---
    if is_root:
        _render_root_view(col_t2, col_t3, col_t4, col_t5)
    else:
        # col_t5 is used for Import now, col_t6 is spacer/search
        _render_dataset_view(current_path[0], col_t2, col_t3, col_t4, col_t5)

def _render_root_view(c_new, c_paste, c_del, c_rename):
    # Fetch Data
    if 'raw_datasets' not in st.session_state:
        st.session_state['raw_datasets'] = get_raw_datasets()
    datasets = st.session_state['raw_datasets']
    
    # Toolbar Actions (Root)
    with c_new:
        if st.button("➕ 新建", use_container_width=True):
            st.session_state['show_create_modal'] = True
    
    if st.session_state.get('show_create_modal'):
        with st.form("create_ds_form"):
            new_name = st.text_input("数据集名称")
            if st.form_submit_button("创建"):
                if new_name:
                    ok, res = create_raw_dataset(new_name)
                    if ok:
                        st.success(f"创建成功: {new_name}")
                        st.session_state['raw_datasets'] = get_raw_datasets()
                        st.session_state['show_create_modal'] = False
                        st.rerun()
                    else:
                        st.error(res)
    
    # Display Datasets as Grid
    if not datasets:
        st.info("暂无数据集，请新建。")
        return

    # Use a dataframe for selection if needed, or just cards. 
    # For "Windows Explorer" feel, let's use a selectable table or cards.
    # Since we want to enter folders, cards/buttons are better.
    # But batch operations on folders (delete multiple) require selection.
    
    # Let's use a DataEditor-like approach for selection
    ds_data = []
    for d in datasets:
        ds_data.append({
            "Select": False,
            "Name": d['name'],
            "Files": d['file_count'],
            "Path": d['path']
        })
    
    df = pd.DataFrame(ds_data)
    
    # Show DataEditor
    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn("选择", width="small"),
            "Name": st.column_config.TextColumn("数据集名称", width="medium"),
            "Files": st.column_config.NumberColumn("文件数", width="small"),
            "Path": st.column_config.TextColumn("路径", width="large", disabled=True),
        },
        hide_index=True,
        key="root_editor",
        use_container_width=True
    )
    
    # Handle Selection
    selected_rows = edited_df[edited_df["Select"]]
    
    # Rename Action
    with c_rename:
        if len(selected_rows) == 1:
            if st.button("✏️ 重命名", use_container_width=True, key="rename_raw_btn"):
                st.session_state['renaming_raw_ds'] = selected_rows.iloc[0]['Name']
    
    # Rename Form
    if st.session_state.get('renaming_raw_ds'):
        target_ds = st.session_state['renaming_raw_ds']
        with st.container():
            st.info(f"正在重命名: {target_ds}")
            c_r1, c_r2 = st.columns([3, 1])
            with c_r1:
                new_name_input = st.text_input("新名称", value=target_ds, key="new_name_raw_input")
            with c_r2:
                st.write("") # spacer
                st.write("")
                if st.button("✅ 确认", type="primary"):
                    if new_name_input and new_name_input != target_ds:
                        ok, res = rename_raw_dataset(target_ds, new_name_input)
                        if ok:
                            st.success("重命名成功")
                            st.session_state['raw_datasets'] = get_raw_datasets()
                            del st.session_state['renaming_raw_ds']
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"失败: {res}")
                    else:
                        st.warning("名称未变更")
                if st.button("❌ 取消"):
                    del st.session_state['renaming_raw_ds']
                    st.rerun()
            st.divider()

    # Batch Delete
    with c_del:
        if not selected_rows.empty:
            if st.button("🗑️ 删除", type="primary", use_container_width=True):
                success_count = 0
                for _, row in selected_rows.iterrows():
                    ok, _ = delete_raw_dataset(row['Name'])
                    if ok: success_count += 1
                if success_count > 0:
                    st.toast(f"已删除 {success_count} 个数据集")
                    st.session_state['raw_datasets'] = get_raw_datasets()
                    time.sleep(1)
                    st.rerun()
    
    # Navigation (Click to enter)
    # Since we can't detect click on row in data_editor easily to trigger action without rerun check,
    # we provide a separate "Open Selected" button or just list them as buttons below if single select.
    # OR: We use the `st.dataframe` on_select if available. 
    # Fallback: List buttons for navigation.
    
    st.markdown("---")
    st.markdown("##### 快速导航 (点击进入)")
    cols = st.columns(4)
    for i, ds in enumerate(datasets):
        with cols[i % 4]:
            if st.button(f"📁 {ds['name']}", key=f"nav_{ds['name']}"):
                st.session_state['explorer_path'] = [ds['name']]
                st.rerun()

def _render_dataset_view(ds_name, c_copy, c_paste, c_del, c_import):
    # Get current dataset info
    raw_datasets = st.session_state.get('raw_datasets', get_raw_datasets())
    current_ds = next((d for d in raw_datasets if d['name'] == ds_name), None)
    
    if not current_ds:
        st.error("数据集不存在")
        st.session_state['explorer_path'] = []
        st.rerun()
        return

    # Fetch files
    files_data = get_raw_files(current_ds['path'])
    files = files_data.get('files', [])
    
    # Import Action
    with c_import:
        if st.button("📤 导入", use_container_width=True, help="上传文件到当前数据集"):
             st.session_state[f'show_upload_{ds_name}'] = not st.session_state.get(f'show_upload_{ds_name}', False)

    # Preprocessing Action (Moved to main area or keep in toolbar? Toolbar is full. Let's put pre-process button above table)
    
    # Upload Area
    if st.session_state.get(f'show_upload_{ds_name}'):
         with st.container():
             st.markdown("#### 上传文件")
             st.caption("支持格式: PDF, TXT, JSON, ZIP (自动解压)")
             uploaded = st.file_uploader("选择文件", accept_multiple_files=True, key=f"up_{ds_name}")
             if st.button("确认上传", key=f"do_up_{ds_name}", type="primary"):
                 if uploaded:
                     with st.spinner("正在上传并处理..."):
                         ok, res = upload_raw_files(ds_name, uploaded)
                         if ok:
                             st.success(f"上传成功: {res['message']}")
                             if res.get('errors'):
                                 with st.expander("部分文件出错"):
                                     for err in res['errors']:
                                         st.warning(err)
                             st.session_state[f'show_upload_{ds_name}'] = False
                             time.sleep(1)
                             st.rerun()
                         else:
                             st.error(f"上传失败: {res}")
             st.divider()

    col_pre, _ = st.columns([1, 4])
    with col_pre:
        if st.button("⚙️ 开始预处理", type="primary", use_container_width=True):
             with st.spinner(f"正在提交预处理任务..."):
                ok, res = start_preprocessing(current_ds['path'])
                if ok:
                    st.success(f"任务已提交! ID: {res['task_id']}")
                    st.info("请前往 [分析任务管理] 查看进度。")
                else:
                    st.error(f"提交失败: {res}")

    # File List Dataframe
    file_list = []
    for f in files:
        file_list.append({
            "Select": False,
            "Name": f['name'],
            "Size": format_bytes(f['size']),
            "Modified": f['modified']
        })
    
    if file_list:
        df_files = pd.DataFrame(file_list)
        
        # Editor
        edited_files = st.data_editor(
            df_files,
            column_config={
                "Select": st.column_config.CheckboxColumn("选择", width="small"),
                "Name": st.column_config.TextColumn("文件名", width="large"),
                "Size": st.column_config.TextColumn("大小", width="medium"),
                "Modified": st.column_config.TextColumn("修改时间", width="medium"),
            },
            hide_index=True,
            key=f"files_editor_{ds_name}", # Unique key per folder
            use_container_width=True
        )
        
        selected_files = edited_files[edited_files["Select"]]
        
        # Toolbar Actions (Files)
        with c_copy:
            if not selected_files.empty:
                if st.button("📋 复制", use_container_width=True):
                    st.session_state['clipboard'] = []
                    for _, row in selected_files.iterrows():
                        st.session_state['clipboard'].append({
                            'type': 'file',
                            'source_ds': ds_name,
                            'name': row['Name']
                        })
                    st.toast(f"已复制 {len(selected_files)} 个文件到剪贴板")
        
        with c_paste:
            clipboard = st.session_state.get('clipboard', [])
            if clipboard:
                if st.button(f"📋 粘贴 ({len(clipboard)})", use_container_width=True):
                    success_count = 0
                    for item in clipboard:
                        if item['type'] == 'file':
                            # Copy file to current dataset
                            # API: copy_raw_file(source_ds, filename, target_ds)
                            ok, _ = copy_raw_file(item['source_ds'], item['name'], ds_name)
                            if ok: success_count += 1
                    
                    if success_count > 0:
                        st.toast(f"成功粘贴 {success_count} 个文件")
                        time.sleep(1)
                        st.rerun()
        
        with c_del:
            if not selected_files.empty:
                if st.button("🗑️ 删除", type="primary", use_container_width=True):
                    success_count = 0
                    for _, row in selected_files.iterrows():
                        ok, _ = delete_raw_file(ds_name, row['Name'])
                        if ok: success_count += 1
                    if success_count > 0:
                        st.toast(f"已删除 {success_count} 个文件")
                        time.sleep(1)
                        st.rerun()
    else:
        st.info("暂无文件，请导入。")
        # Empty clipboard/paste logic even if no files
        with c_paste:
            clipboard = st.session_state.get('clipboard', [])
            if clipboard:
                 if st.button(f"📋 粘贴 ({len(clipboard)})", use_container_width=True):
                    success_count = 0
                    for item in clipboard:
                        if item['type'] == 'file':
                            ok, _ = copy_raw_file(item['source_ds'], item['name'], ds_name)
                            if ok: success_count += 1
                    
                    if success_count > 0:
                        st.toast(f"成功粘贴 {success_count} 个文件")
                        time.sleep(1)
                        st.rerun()

def render_processed_explorer():
    # Initialize Session State for Explorer
    if 'proc_explorer_path' not in st.session_state:
        st.session_state['proc_explorer_path'] = [] # List of dataset IDs
    
    current_path = st.session_state['proc_explorer_path']
    is_root = len(current_path) == 0
    
    # --- Breadcrumbs & Header ---
    path_str = " 🏠 Home"
    if not is_root:
        # Get dataset name from ID
        datasets = st.session_state.get('datasets_list', [])
        ds = next((d for d in datasets if d['id'] == current_path[0]), None)
        ds_name = ds['name'] if ds else current_path[0]
        path_str += f" / 📦 {ds_name}"
    
    st.markdown(f"### {path_str}")
    
    # --- Toolbar ---
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([1, 1, 1, 1, 3])
    
    with col_t1:
        if not is_root:
            if st.button("⬆️ 上一级", key="proc_up", use_container_width=True):
                st.session_state['proc_explorer_path'].pop()
                st.rerun()
        else:
            if st.button("🔄 刷新", key="proc_refresh", use_container_width=True):
                st.session_state['datasets_list'] = get_datasets()
                st.rerun()

    # --- Content Area ---
    if is_root:
        _render_processed_root_view(col_t2, col_t3, col_t4)
    else:
        _render_processed_dataset_view(current_path[0], col_t2, col_t3)

def _render_processed_root_view(c_rename, c_del, c_split):
    # Fetch Data
    if 'datasets_list' not in st.session_state:
        st.session_state['datasets_list'] = get_datasets()
    datasets = st.session_state['datasets_list']
    
    if not datasets:
        st.info("暂无已预处理的数据集。")
        return

    # DataEditor
    ds_data = []
    for d in datasets:
        stats = d.get('stats', {})
        ds_data.append({
            "Select": False,
            "Name": d['name'],
            "ID": d['id'],
            "Type": d['type'],
            "Graphs": d['num_graphs'],
            "Nodes": stats.get('total_nodes', '-'),
            "Edges": stats.get('total_edges', '-'),
            "Created": d['created']
        })
    
    df = pd.DataFrame(ds_data)
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn("选择", width="small"),
            "Name": st.column_config.TextColumn("数据集名称", width="medium"),
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Type": st.column_config.TextColumn("类型", width="small"),
            "Graphs": st.column_config.NumberColumn("图数量", width="small"),
            "Nodes": st.column_config.NumberColumn("节点总数", width="small"),
            "Created": st.column_config.TextColumn("创建时间", width="medium"),
        },
        hide_index=True,
        key="proc_root_editor",
        use_container_width=True
    )
    
    selected_rows = edited_df[edited_df["Select"]]
    
    # Actions
    with c_rename:
        if len(selected_rows) == 1:
            if st.button("✏️ 重命名", key="ren_proc_root_btn", use_container_width=True):
                 st.session_state['renaming_proc_ds'] = {
                     'id': selected_rows.iloc[0]['ID'],
                     'name': selected_rows.iloc[0]['Name']
                 }

    with c_del:
        if not selected_rows.empty:
            if st.button("🗑️ 删除", key="del_proc_root_btn", type="primary", use_container_width=True):
                 count = 0
                 for _, row in selected_rows.iterrows():
                     ok, _ = delete_dataset(row['ID'])
                     if ok: count += 1
                 if count > 0:
                     st.toast(f"已删除 {count} 个数据集")
                     st.session_state['datasets_list'] = get_datasets()
                     time.sleep(1)
                     st.rerun()

    # Rename Modal Logic
    if st.session_state.get('renaming_proc_ds'):
        target = st.session_state['renaming_proc_ds']
        with st.container():
             st.info(f"重命名: {target['name']}")
             c_r1, c_r2 = st.columns([3, 1])
             with c_r1:
                 new_name = st.text_input("新名称", value=target['name'], key="new_name_proc_input")
             with c_r2:
                 st.write("")
                 st.write("")
                 if st.button("✅", key="confirm_ren_proc"):
                     if new_name and new_name != target['name']:
                         ok, res = rename_dataset(target['id'], new_name)
                         if ok:
                             st.success("成功")
                             st.session_state['datasets_list'] = get_datasets()
                             del st.session_state['renaming_proc_ds']
                             st.rerun()
                         else:
                             st.error(str(res))
                 if st.button("❌", key="cancel_ren_proc"):
                     del st.session_state['renaming_proc_ds']
                     st.rerun()
             st.divider()

    # Navigation
    st.markdown("---")
    st.markdown("##### 快速导航 (点击进入)")
    cols = st.columns(4)
    for i, ds in enumerate(datasets):
        with cols[i % 4]:
            if st.button(f"📦 {ds['name']}", key=f"nav_proc_{ds['id']}"):
                st.session_state['proc_explorer_path'] = [ds['id']]
                st.rerun()

def _render_processed_dataset_view(dataset_id, c_del, c_split):
    # Get dataset
    datasets = st.session_state.get('datasets_list', get_datasets())
    ds = next((d for d in datasets if d['id'] == dataset_id), None)
    
    if not ds:
        st.error("数据集不存在")
        st.session_state['proc_explorer_path'] = []
        st.rerun()
        return

    # Header Info
    st.caption(f"ID: {ds['id']} | Type: {ds['type']} | Created: {ds['created']}")
    
    # Split Action
    with c_split:
        if st.button("✂️ 划分数据集", use_container_width=True):
             st.session_state['show_split_modal'] = True
             
    if st.session_state.get('show_split_modal'):
        with st.container():
            st.markdown("##### 数据集划分")
            train_ratio = st.slider("训练集比例", 0.1, 0.9, 0.7, 0.05, key="tr_modal")
            val_ratio = st.slider("验证集比例", 0.05, 0.4, 0.15, 0.05, key="vr_modal")
            st.caption(f"测试集比例: {1.0 - train_ratio - val_ratio:.2f}")
            if st.button("确认划分", type="primary"):
                 ok, res = split_dataset(ds['id'], train_ratio, val_ratio)
                 if ok:
                     st.success(f"划分成功! Train: {res['splits']['train']}, Val: {res['splits']['val']}, Test: {res['splits']['test']}")
                     st.session_state['show_split_modal'] = False
                 else:
                     st.error(str(res))
            if st.button("关闭"):
                st.session_state['show_split_modal'] = False
            st.divider()

    # Samples List
    raw_stats = ds.get('stats', {}).get('raw_stats', [])
    
    if raw_stats:
        samples_data = []
        for s in raw_stats:
            samples_data.append({
                "Select": False,
                "Report ID": s.get('report_id'),
                "Group": s.get('apt_group'),
                "Vendor": s.get('source_vendor', '-'),
                "Date": s.get('published_date', '-'),
                "Nodes": s.get('num_nodes'),
                "Edges": s.get('num_edges')
            })
        
        df_samples = pd.DataFrame(samples_data)
        
        edited_samples = st.data_editor(
            df_samples,
            column_config={
                "Select": st.column_config.CheckboxColumn("选择", width="small"),
                "Report ID": st.column_config.TextColumn("报告ID", width="medium"),
                "Group": st.column_config.TextColumn("APT组织", width="medium"),
                "Vendor": st.column_config.TextColumn("来源厂商", width="medium"),
                "Date": st.column_config.TextColumn("发布日期", width="medium"),
                "Nodes": st.column_config.NumberColumn("节点数", width="small"),
                "Edges": st.column_config.NumberColumn("边数", width="small"),
            },
            hide_index=True,
            key=f"samples_editor_{dataset_id}",
            use_container_width=True
        )
        
        selected_samples = edited_samples[edited_samples["Select"]]
        
        with c_del:
            if not selected_samples.empty:
                if st.button("🗑️ 删除样本", type="primary", use_container_width=True):
                    count = 0
                    for _, row in selected_samples.iterrows():
                        ok, _ = delete_graph(dataset_id, row['Report ID'])
                        if ok: count += 1
                    if count > 0:
                        st.toast(f"已删除 {count} 个样本")
                        # Refresh dataset info (requires backend call)
                        st.session_state['datasets_list'] = get_datasets()
                        time.sleep(1)
                        st.rerun()
    else:
        st.info("该数据集无详细样本信息 (可能是旧版本数据或非图数据集)。")


def render_datasets():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-right: 1rem; color: #00d4ff; filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));"><i class="fas fa-database"></i></div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">数据集管理</h1>
            <p style="color: #00d4ff; margin: 0; opacity: 0.8; letter-spacing: 1px;">Dataset & Samples Management</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_processed, tab_raw = st.tabs(["📦 已预处理数据集 (Processed)", "📂 未处理数据集 (Raw)"])

    # ==========================================
    # Tab 1: Processed Datasets
    # ==========================================
    with tab_processed:
        render_processed_explorer()


    # ==========================================
    # Tab 2: Raw Datasets
    # ==========================================
    with tab_raw:
        render_raw_file_explorer()


