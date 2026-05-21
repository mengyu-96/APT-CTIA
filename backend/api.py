# backend/api.py
from flask import Flask, request, jsonify, send_file, abort
from pathlib import Path
import os
import shutil
import datetime
import logging

from core.preprocess import run_preprocessing_pipeline
from core.train import run_training_pipeline
from core.inference import run_inference_pipeline
from core.report_generator import ReportGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Flask app
app = Flask(__name__)

# 定义基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_ARCHIVE = BASE_DIR / 'results_archive'
PROCESSED_DATA_DIR = RESULTS_ARCHIVE / 'processed_data'
TRAINING_RUNS_DIR = RESULTS_ARCHIVE / 'training_runs'
REPORTS_DIR = RESULTS_ARCHIVE / 'reports'
RAW_DATA_PATHS = [
    BASE_DIR / 'dataset_TXT',
    BASE_DIR / 'reports'
]

ATTRIBUTION_RESULTS_DIR = RESULTS_ARCHIVE / 'attribution_results'
ATTRIBUTION_RESULTS_DIR.mkdir(exist_ok=True)

# 设置 app 配置
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# 确保所有需要的目录都存在
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_ARCHIVE.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
TRAINING_RUNS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource Not Found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error", "details": str(error)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """一个简单的接口，用于检查后端服务是否正在运行。"""
    return jsonify({"status": "running", "message": "Backend is active!"})

@app.route('/api/artifact', methods=['GET'])
def get_artifact():
    raw_path = request.args.get('path')
    if not raw_path:
        return jsonify({"error": "Missing path"}), 400

    try:
        p = Path(raw_path)
        if not p.is_absolute():
            p = (BASE_DIR / p)
        resolved = p.resolve()
    except Exception:
        return jsonify({"error": "Invalid path"}), 400

    archive_root = RESULTS_ARCHIVE.resolve()
    try:
        resolved.relative_to(archive_root)
    except Exception:
        abort(403)

    if not resolved.exists() or not resolved.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(str(resolved), as_attachment=False, conditional=True)

import threading
import uuid
import json

# 简单的文件持久化任务队列
TASK_QUEUE_FILE = RESULTS_ARCHIVE / 'tasks.json'
TASK_QUEUE = {}

def load_tasks():
    global TASK_QUEUE
    if TASK_QUEUE_FILE.exists():
        try:
            content = TASK_QUEUE_FILE.read_text(encoding='utf-8')
            if content:
                TASK_QUEUE = json.loads(content)
            
            # Check for stale running tasks
            for tid, task in TASK_QUEUE.items():
                if task.get('status') == 'running':
                    task['status'] = 'failed'
                    task['error'] = 'Server restarted while task was running'
            save_tasks()
            logging.info(f"Loaded {len(TASK_QUEUE)} tasks from persistence.")
            
        except Exception as e:
            logging.error(f"Failed to load tasks: {e}")
            TASK_QUEUE = {}

def save_tasks():
    try:
        temp_file = TASK_QUEUE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(TASK_QUEUE, f, indent=2)
        temp_file.replace(TASK_QUEUE_FILE)
    except Exception as e:
        logging.error(f"Failed to save tasks: {e}")

# Load tasks on startup
load_tasks()

def async_task_wrapper(task_id, func, *args, **kwargs):
    """
    包装器，用于在独立线程中运行任务并更新状态。
    """
    try:
        logging.info(f"Task {task_id} started.")
        TASK_QUEUE[task_id]['status'] = 'running'
        save_tasks()
        result = func(*args, **kwargs)
        TASK_QUEUE[task_id]['status'] = 'completed'
        TASK_QUEUE[task_id]['result'] = result
        logging.info(f"Task {task_id} completed successfully.")
    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
        TASK_QUEUE[task_id]['status'] = 'failed'
        TASK_QUEUE[task_id]['error'] = str(e)
    finally:
        save_tasks()

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """
    获取所有任务列表 (简略信息)
    """
    tasks_list = []
    # Convert dict to list and sort by creation time (descending)
    # We assume keys are UUIDs. 
    # We need to look at 'created_at' or 'created' field.
    
    for t_id, t_data in TASK_QUEUE.items():
        # Basic info
        task_info = {
            "id": t_id,
            "status": t_data.get('status'),
            "type": t_data.get('type', 'unknown'),
            "created": t_data.get('created_at', ''),
            "progress": t_data.get('progress', 0),
            "message": t_data.get('message', ''),
            "error": t_data.get('error'),
            # specific fields
            "name": t_data.get('dataset_name') or t_data.get('output_name') or f"{t_data.get('type')} task",
            "files": len(t_data.get('files', [])) if 'files' in t_data else 0
        }
        
        # Enrich for training tasks
        if t_data.get('type') == 'train':
            task_info['model'] = t_data.get('config', {}).get('model_type')
            task_info['dataset'] = t_data.get('config', {}).get('dataset_name')
            task_info['name'] = f"Train {task_info['model']}"
            
        # Enrich for inference tasks
        if t_data.get('type') == 'inference':
             task_info['name'] = "Attribution Inference"
             
        tasks_list.append(task_info)
        
    # Sort by created time (if available strings)
    try:
        tasks_list.sort(key=lambda x: x['created'], reverse=True)
    except:
        pass
        
    return jsonify(tasks_list)

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    查询异步任务的状态。
    """
    task = TASK_QUEUE.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """
    删除任务记录。
    """
    if task_id in TASK_QUEUE:
        del TASK_QUEUE[task_id]
        save_tasks()
        return jsonify({"message": "Task deleted successfully"})
    else:
        return jsonify({"error": "Task not found"}), 404

@app.route('/api/raw_datasets', methods=['GET'])
def list_raw_datasets():
    """
    列出所有未处理的原始数据集 (dataset_TXT 下的子目录).
    """
    datasets = []
    # Primary root is dataset_TXT
    root = BASE_DIR / 'dataset_TXT'
    
    if root.exists() and root.is_dir():
        for item in root.iterdir():
            if item.is_dir():
                # Count files (recursively)
                file_count = 0
                # Simple heuristic: scan for pdf/txt
                for ext in ['*.pdf', '*.txt', '*.json']:
                    file_count += len(list(item.rglob(ext)))
                
                datasets.append({
                    "id": item.name,
                    "name": item.name,
                    "path": str(item),
                    "file_count": file_count,
                    "type": "FileSystem"
                })
            
    return jsonify(datasets)

@app.route('/api/raw_datasets/files', methods=['GET'])
def list_raw_dataset_files():
    """
    列出原始数据集中的文件。
    Query param: path
    """
    path_str = request.args.get('path')
    if not path_str:
        return jsonify({"error": "Missing path parameter"}), 400
        
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        return jsonify({"error": "Path not found"}), 404
        
    files = []
    # Limit to first 1000 files to avoid payload explosion
    count = 0
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in ['.pdf', '.txt', '.json']:
            files.append({
                "name": p.name,
                "path": str(p),
                "size": p.stat().st_size,
                "modified": datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
            count += 1
            if count >= 1000:
                break
                
    return jsonify({"files": files, "total_shown": count})

@app.route('/api/raw_datasets', methods=['POST'])
def create_raw_dataset():
    """Create a new raw dataset (folder)"""
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"error": "Missing name"}), 400
        
    # Default to creating in dataset_TXT root
    target_path = BASE_DIR / 'dataset_TXT' / name
    
    if target_path.exists():
         return jsonify({"error": "Dataset already exists"}), 400
         
    try:
        target_path.mkdir(parents=True)
        return jsonify({"message": "Dataset created successfully", "path": str(target_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>', methods=['DELETE'])
def delete_raw_dataset(name):
    """Delete a raw dataset folder"""
    target_path = BASE_DIR / 'dataset_TXT' / name
    if not target_path.exists():
        # Try finding in other raw paths if needed, but primary is dataset_TXT
        found = False
        for root in RAW_DATA_PATHS:
             if (root / name).exists():
                 target_path = root / name
                 found = True
                 break
        if not found:
            return jsonify({"error": "Dataset not found"}), 404
            
    try:
        shutil.rmtree(target_path)
        return jsonify({"message": "Dataset deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>', methods=['PUT'])
def rename_raw_dataset(name):
    """Rename a raw dataset folder"""
    data = request.json
    new_name = data.get('new_name')
    if not new_name:
        return jsonify({"error": "Missing new_name"}), 400
        
    target_path = BASE_DIR / 'dataset_TXT' / name
    if not target_path.exists():
        return jsonify({"error": "Dataset not found"}), 404
        
    new_path = target_path.parent / new_name
    if new_path.exists():
        return jsonify({"error": "Target name already exists"}), 400
        
    try:
        target_path.rename(new_path)
        return jsonify({"message": "Renamed successfully", "new_path": str(new_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>/copy', methods=['POST'])
def copy_raw_dataset(name):
    """Copy a raw dataset folder"""
    data = request.json
    new_name = data.get('new_name', f"{name}_copy")
    
    target_path = BASE_DIR / 'dataset_TXT' / name
    if not target_path.exists():
        return jsonify({"error": "Dataset not found"}), 404
        
    new_path = target_path.parent / new_name
    if new_path.exists():
        return jsonify({"error": "Target name already exists"}), 400
        
    try:
        shutil.copytree(target_path, new_path)
        return jsonify({"message": "Copied successfully", "path": str(new_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>/files/<filename>', methods=['DELETE'])
def delete_raw_file(name, filename):
    """Delete a file from a raw dataset"""
    target_path = BASE_DIR / 'dataset_TXT' / name / filename
    if not target_path.exists():
        return jsonify({"error": "File not found"}), 404
        
    try:
        os.remove(target_path)
        return jsonify({"message": "File deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>/files/<filename>/copy', methods=['POST'])
def copy_raw_file(name, filename):
    """Copy a file to another dataset"""
    data = request.json
    target_dataset = data.get('target_dataset')
    if not target_dataset:
        return jsonify({"error": "Missing target_dataset"}), 400
        
    source_path = BASE_DIR / 'dataset_TXT' / name / filename
    if not source_path.exists():
        return jsonify({"error": "Source file not found"}), 404
        
    dest_dir = BASE_DIR / 'dataset_TXT' / target_dataset
    if not dest_dir.exists():
        return jsonify({"error": "Target dataset not found"}), 404
        
    dest_path = dest_dir / filename
    if dest_path.exists():
        # Auto-rename if collision
        base, ext = os.path.splitext(filename)
        dest_path = dest_dir / f"{base}_copy{ext}"
        
    try:
        shutil.copy2(source_path, dest_path)
        return jsonify({"message": "File copied successfully", "path": str(dest_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/raw_datasets/<name>/upload', methods=['POST'])
def upload_to_raw_dataset(name):
    """Upload files to a raw dataset folder"""
    target_path = BASE_DIR / 'dataset_TXT' / name
    if not target_path.exists():
        # Try other paths
        found = False
        for root in RAW_DATA_PATHS:
             if (root / name).exists():
                 target_path = root / name
                 found = True
                 break
        if not found:
            return jsonify({"error": "Dataset not found"}), 404
            
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
        
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400
        
    saved_count = 0
    errors = []
    
    for file in files:
        if file and file.filename:
            # Validate extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ['.pdf', '.txt', '.json', '.zip']:
                errors.append(f"Skipped {file.filename}: Invalid extension {ext}")
                continue
                
            try:
                if ext == '.zip':
                    # Handle zip extraction
                    import zipfile
                    zip_path = target_path / file.filename
                    file.save(str(zip_path))
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Extract only allowed types
                            for member in zip_ref.namelist():
                                m_ext = os.path.splitext(member)[1].lower()
                                if m_ext in ['.pdf', '.txt', '.json'] and not member.startswith('__MACOSX') and not '..' in member:
                                    zip_ref.extract(member, target_path)
                                    saved_count += 1
                    except Exception as ze:
                        errors.append(f"Error extracting {file.filename}: {ze}")
                    finally:
                        # Clean up zip file
                        if zip_path.exists():
                            os.remove(zip_path)
                else:
                    # Save normal file
                    save_path = target_path / file.filename
                    file.save(str(save_path))
                    saved_count += 1
            except Exception as e:
                errors.append(f"Failed to save {file.filename}: {e}")
                
    return jsonify({
        "message": f"Successfully uploaded/extracted {saved_count} files.",
        "errors": errors
    })

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    处理上传的原始日志文件（异步）。
    支持两种模式：
    1. 上传文件 (Multipart form data 'files')
    2. 指定本地路径 (JSON body 'raw_dataset_path')
    """
    # Check for JSON input first (Local Path mode)
    if request.is_json:
        data = request.json
        raw_dataset_path = data.get('raw_dataset_path')
        if not raw_dataset_path:
             return jsonify({"error": "Missing raw_dataset_path"}), 400
             
        path = Path(raw_dataset_path)
        if not path.exists():
            return jsonify({"error": "Path does not exist"}), 404
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Determine output directory name
        output_name = data.get('output_name')
        if output_name:
            # Sanitize name
            safe_name = "".join([c for c in output_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            if safe_name:
                dir_name = f"{safe_name}_{timestamp}"
            else:
                dir_name = timestamp
        else:
            dir_name = timestamp
            
        task_id = str(uuid.uuid4())
        output_dir = PROCESSED_DATA_DIR / dir_name
        output_dir.mkdir(exist_ok=True)
        
        TASK_QUEUE[task_id] = {
            'status': 'pending',
            'created_at': timestamp,
            'type': 'preprocess',
            'source': str(path)
        }
        save_tasks()
        
        def preprocess_worker_local():
            def progress_cb(current, total, msg):
                if total > 0:
                    # Scale file processing to 0-90%
                    if current > total: # Hack for when current is actually percent (e.g. 90, 100)
                        pct = current
                    else:
                        pct = int((current / total) * 90)
                else:
                    pct = 0
                
                # If explicitly passing percentages for phases
                if msg == "Building Graphs...":
                    pct = 90
                elif msg == "Completed":
                    pct = 100
                elif msg.startswith("Done"):
                    pct = 100

                TASK_QUEUE[task_id]['progress'] = pct
                TASK_QUEUE[task_id]['message'] = msg
                save_tasks()

            result_path = run_preprocessing_pipeline(
                raw_data_dir=path,
                base_output_dir=output_dir,
                skip_graphs=False,
                progress_callback=progress_cb
            )
            return {
                "output_path": str(result_path),
                "label_mapping_path": str(output_dir / 'label_mapping.json'),
                "graph_stats_path": str(output_dir / 'graph_stats.json')
            }

        thread = threading.Thread(target=async_task_wrapper, args=(task_id, preprocess_worker_local))
        thread.start()
        
        return jsonify({
            "message": "Preprocessing task submitted (Local Path)",
            "task_id": task_id,
            "status_url": f"/api/tasks/{task_id}"
        }), 202

    # Fallback to File Upload mode
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request or JSON body"}), 400

    files = request.files.getlist('files')

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    # 为本次上传创建一个唯一的临时目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    temp_upload_dir = UPLOAD_FOLDER / f"temp_{timestamp}"
    temp_upload_dir.mkdir(exist_ok=True)

    try:
        saved_files = []
        for file in files:
            if file:
                file_path = temp_upload_dir / file.filename
                file.save(str(file_path))
                saved_files.append(str(file_path))
        
        logging.info(f"Uploaded files saved to {temp_upload_dir}")

        # 创建异步任务
        task_id = str(uuid.uuid4())
        
        # Determine output directory name
        output_name = request.form.get('output_name')
        if output_name:
            safe_name = "".join([c for c in output_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            if safe_name:
                dir_name = f"{safe_name}_{timestamp}"
            else:
                dir_name = timestamp
        else:
            dir_name = timestamp
            
        output_dir = PROCESSED_DATA_DIR / dir_name
        output_dir.mkdir(exist_ok=True)

        TASK_QUEUE[task_id] = {
            'status': 'pending',
            'created_at': timestamp,
            'type': 'preprocess',
            'source': 'upload'
        }
        save_tasks()

        # 定义实际执行的函数
        def preprocess_worker():
            try:
                def progress_cb(current, total, msg):
                    if total > 0:
                         if current > total:
                             pct = current
                         else:
                             pct = int((current / total) * 90)
                    else:
                        pct = 0
                    
                    if msg == "Building Graphs...":
                        pct = 90
                    elif msg == "Completed":
                        pct = 100
                    elif msg.startswith("Done"):
                        pct = 100

                    TASK_QUEUE[task_id]['progress'] = pct
                    TASK_QUEUE[task_id]['message'] = msg
                    save_tasks()

                result_path = run_preprocessing_pipeline(
                    raw_data_dir=temp_upload_dir,
                    base_output_dir=output_dir,
                    skip_graphs=False,
                    progress_callback=progress_cb
                )
                return {
                    "output_path": str(result_path),
                    "label_mapping_path": str(output_dir / 'label_mapping.json'),
                    "graph_stats_path": str(output_dir / 'graph_stats.json')
                }
            finally:
                # 清理临时上传目录
                if temp_upload_dir.exists():
                    shutil.rmtree(temp_upload_dir)
                    logging.info(f"Cleaned up temporary directory: {temp_upload_dir}")

        # 启动线程
        thread = threading.Thread(target=async_task_wrapper, args=(task_id, preprocess_worker))
        thread.start()

        return jsonify({
            "message": "Preprocessing task submitted",
            "task_id": task_id,
            "status_url": f"/api/tasks/{task_id}"
        }), 202

    except Exception as e:
        logging.error(f"An error occurred during upload handling: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """
    列出所有已处理的数据集 (Grouped by preprocessing run)。
    """
    datasets = []
    
    if PROCESSED_DATA_DIR.exists():
        for run_dir in PROCESSED_DATA_DIR.iterdir():
            if run_dir.is_dir():
                # Check for graph stats file
                stats_file = run_dir / "graph_stats.json"
                
                # Check for graphs directory (new format) or graphs.pt (old format)
                graphs_dir = run_dir / "graphs"
                graphs_pt = run_dir / "graphs.pt"
                
                num_graphs = 0
                if graphs_dir.exists() and graphs_dir.is_dir():
                    num_graphs = len(list(graphs_dir.glob("*.pt")))
                elif graphs_pt.exists():
                    # Estimate or load (loading is slow, maybe just rely on stats or file existence)
                    # For quick listing, we might rely on stats if available
                    if stats_file.exists():
                        try:
                             with open(stats_file, 'r', encoding='utf-8') as f:
                                 num_graphs = len(json.load(f))
                        except:
                             num_graphs = 1 # At least one file
                    else:
                        num_graphs = 1
                
                if num_graphs > 0:
                    # Basic info
                    display_name = f"Dataset_{run_dir.name}"
                    
                    # Check metadata for custom name
                    metadata_file = run_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                                if meta.get('name'):
                                    display_name = meta.get('name')
                        except:
                            pass

                    dataset_info = {
                        "id": run_dir.name, # Use directory name as ID (timestamp)
                        "name": display_name,
                        "type": "Graph Collection",
                        "path": str(run_dir),
                        "num_graphs": num_graphs,
                        "created": datetime.datetime.fromtimestamp(run_dir.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Ready",
                        "stats": {}
                    }
                    
                    # Load stats if available
                    if stats_file.exists():
                        try:
                            with open(stats_file, 'r', encoding='utf-8') as f:
                                stats_data = json.load(f)
                                # Aggregate stats
                                total_nodes = sum(s.get('num_nodes', 0) for s in stats_data)
                                total_edges = sum(s.get('num_edges', 0) for s in stats_data)
                                apt_groups = list(set(s.get('apt_group', 'Unknown') for s in stats_data))
                                
                                dataset_info["stats"] = {
                                    "total_nodes": total_nodes,
                                    "total_edges": total_edges,
                                    "apt_groups": apt_groups,
                                    "raw_stats": stats_data
                                }
                        except Exception as e:
                            logging.warning(f"Failed to load stats for {run_dir}: {e}")
                            
                    datasets.append(dataset_info)
    
    # Sort by created desc
    datasets.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(datasets)

@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """
    删除指定的数据集 (整个目录)。
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Dataset not found"}), 404
        
    try:
        shutil.rmtree(target_dir)
        return jsonify({"message": "Dataset deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets/<dataset_id>/rename', methods=['PUT'])
def rename_dataset(dataset_id):
    """
    重命名已处理的数据集 (Update metadata.json).
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    if not target_dir.exists():
        return jsonify({"error": "Dataset not found"}), 404
        
    data = request.json
    new_name = data.get('new_name')
    if not new_name:
        return jsonify({"error": "Missing new_name"}), 400
        
    metadata_file = target_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except:
            pass
            
    metadata['name'] = new_name
    
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        return jsonify({"message": "Dataset renamed successfully", "name": new_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets/<dataset_id>/graphs', methods=['GET'])
def list_graphs_in_dataset(dataset_id):
    """
    列出指定数据集中的所有图 (样本)。
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Dataset not found"}), 404

    graphs = []
    
    # Check for stats file to enrich info
    stats_map = {}
    stats_file = target_dir / "graph_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_list = json.load(f)
                # Assuming order matches or we need a key. 
                # In preprocess.py, we don't save filename in stats explicitly but report_id is there.
                # Let's map report_id -> stat
                for s in stats_list:
                    stats_map[s.get('report_id')] = s
        except:
            pass

    # List all .pt files (or whatever graph format)
    # Actually, in preprocess.py, we save ALL graphs into a single `graphs.pt` file usually for PyG.
    # Let's check preprocess.py...
    # Yes: torch.save(graphs, self.graph_output_path) where graph_output_path is output_dir / "graphs.pt"
    # Wait, so we don't have individual files?
    # If so, we cannot delete individual graphs easily without reloading the big file.
    # BUT, the user requirement says "预处理页面处理过的结果（图）会存储到该页面... 允许数据集内部删掉某个报告预处理的结果"
    
    # If the system currently saves as one big .pt file, we need to support splitting or just virtual listing.
    # Let's check if we can list the 'source' files or if we should modify preprocess to save individually?
    # Modifying preprocess to save individually is better for management but slower for loading.
    # Alternatively, we just list the stats entries as "graphs" since they correspond 1-1 to the list in graphs.pt
    
    # Let's assume we list based on stats (which represent the reports).
    # If user wants to delete one, we might just mark it as deleted in metadata or actually rebuild the graphs.pt?
    # Rebuilding graphs.pt is expensive.
    
    # Let's check the current implementation of preprocess.py again via memory or file read.
    # It saves `graphs.pt` (List[Data]).
    
    # STRATEGY: 
    # For now, we will list the items based on graph_stats.json.
    # To "delete", we will remove the entry from graph_stats.json AND from the graphs.pt list.
    # This requires loading torch, which is heavy for an API endpoint if not careful.
    # But it is the only way to support "real" deletion inside a .pt list.
    
    if not stats_map and stats_file.exists():
         # Reload if we failed above or just use the list
         try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_list = json.load(f)
                return jsonify(stats_list)
         except Exception as e:
             return jsonify({"error": str(e)}), 500
             
    if stats_map:
        return jsonify(list(stats_map.values()))
        
    return jsonify([])

@app.route('/api/datasets/<dataset_id>/graphs/<report_id>', methods=['DELETE'])
def delete_graph_from_dataset(dataset_id, report_id):
    """
    从数据集中删除指定图 (样本)。
    This deletes the specific .pt file and updates graph_stats.json.
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    if not target_dir.exists():
         return jsonify({"error": "Dataset not found"}), 404
         
    stats_path = target_dir / "graph_stats.json"
    graphs_dir = target_dir / "graphs"
    
    # Check for new format
    if graphs_dir.exists() and graphs_dir.is_dir():
        graph_file = graphs_dir / f"{report_id}.pt"
        
        if not graph_file.exists():
             return jsonify({"error": "Graph file not found"}), 404
             
        try:
            # 1. Delete file
            os.remove(graph_file)
            
            # 2. Update stats if exists
            remaining_count = 0
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                new_stats = [s for s in stats if s.get('report_id') != report_id]
                remaining_count = len(new_stats)
                
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(new_stats, f, indent=2)
            else:
                remaining_count = len(list(graphs_dir.glob("*.pt")))
            
            return jsonify({"message": f"Graph {report_id} deleted successfully", "remaining": remaining_count})
            
        except Exception as e:
            logging.error(f"Failed to delete graph {report_id}: {e}")
            return jsonify({"error": str(e)}), 500

    # Fallback to old single-file format (Read-only or full rewrite)
    graphs_path = target_dir / "graphs.pt"
    if graphs_path.exists():
        return jsonify({"error": "Legacy dataset format (single file) does not support individual deletion via this API yet. Please re-process data."}), 400

    return jsonify({"error": "Dataset files missing"}), 404

@app.route('/api/datasets/<dataset_id>/split', methods=['POST'])
def split_dataset(dataset_id):
    """
    将数据集划分为训练集/验证集/测试集 (Logic placeholder)
    In a real GNN pipeline, we often do this dynamically or save indices.
    Here we can save a split.json file in the dataset directory.
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    if not target_dir.exists():
        return jsonify({"error": "Dataset not found"}), 404
        
    config = request.json
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    
    try:
        # Load all graphs (or just list them)
        graphs_dir = target_dir / "graphs"
        if graphs_dir.exists() and graphs_dir.is_dir():
             pt_files = list(graphs_dir.glob("*.pt"))
        else:
             pt_files = list(target_dir.rglob("*.pt"))
             
        if not pt_files:
             return jsonify({"error": "No graphs found"}), 400
             
        # Mock split logic - just shuffle and save filenames
        import random
        filenames = [f.name for f in pt_files]
        random.shuffle(filenames)
        
        n_total = len(filenames)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            "train": filenames[:n_train],
            "val": filenames[n_train:n_train+n_val],
            "test": filenames[n_train+n_val:],
            "config": config,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(target_dir / 'splits.json', 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)
            
        return jsonify({"message": "Dataset split successfully", "splits": {k: len(v) for k, v in splits.items() if isinstance(v, list)}})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets/<dataset_id>/stats', methods=['GET'])
def get_dataset_stats(dataset_id):
    """获取数据集的统计信息"""
    target_dir = PROCESSED_DATA_DIR / dataset_id
    stats_file = target_dir / "graph_stats.json"
    
    if stats_file.exists():
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Stats not found"}), 404

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    列出所有已训练的模型。
    """
    models = []
    if TRAINING_RUNS_DIR.exists():
        for run_dir in TRAINING_RUNS_DIR.iterdir():
            if run_dir.is_dir():
                # 查找 results.json
                results_file = run_dir / 'results.json'
                if results_file.exists():
                    try:
                        import json
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        model_name = results.get('custom_name')
                        if not model_name:
                             model_name = f"{results.get('config', {}).get('model_type', 'Unknown')}_{run_dir.name}"

                        models.append({
                            "id": run_dir.name,
                            "name": model_name,
                            "type": results.get('config', {}).get('model_type', 'GNN'),
                            "dataset_id": results.get('config', {}).get('dataset_id', 'Unknown'),
                            "dataset_name": results.get('config', {}).get('dataset_name', 'Unknown'),
                            "accuracy": results.get('test_accuracy', 0.0),
                            "f1_score": results.get('test_f1_weighted', 0.0),
                            "epochs": results.get('config', {}).get('epochs', 0),
                            "created": datetime.datetime.fromtimestamp(run_dir.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                            "status": "Completed",
                            "classification_report": results.get('classification_report', {}),
                            "path": str(run_dir)
                        })
                    except Exception as e:
                        logging.warning(f"Failed to parse results for {run_dir}: {e}")
    
    # 按创建时间倒序
    models.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(models)

@app.route('/api/models/<model_id>/rename', methods=['PUT'])
def rename_model(model_id):
    """
    重命名模型 (Update results.json).
    """
    target_dir = TRAINING_RUNS_DIR / model_id
    if not target_dir.exists():
        return jsonify({"error": "Model not found"}), 404
        
    data = request.json
    new_name = data.get('new_name')
    if not new_name:
        return jsonify({"error": "Missing new_name"}), 400
        
    results_file = target_dir / 'results.json'
    if not results_file.exists():
        return jsonify({"error": "Results file not found"}), 404
        
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        results['custom_name'] = new_name
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        return jsonify({"message": "Model renamed successfully", "name": new_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """
    删除指定的模型（整个训练运行目录）。
    """
    target_dir = TRAINING_RUNS_DIR / model_id
    
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Model not found"}), 404
        
    try:
        shutil.rmtree(target_dir)
        return jsonify({"message": "Model deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model_api():
    """
    启动模型训练（异步）。
    """
    config = request.json
    if not config:
        return jsonify({"error": "Request body must be a JSON with training configuration"}), 400

    processed_data_path = config.get('processed_data_path')
    if not processed_data_path or not Path(processed_data_path).exists():
        return jsonify({"error": f"Processed data path is missing or does not exist: {processed_data_path}"}), 400

    # 将主输出目录注入配置中
    config['base_output_dir'] = str(TRAINING_RUNS_DIR)

    # 创建异步任务
    task_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    TASK_QUEUE[task_id] = {
        'status': 'pending',
        'created_at': timestamp,
        'type': 'train',
        'config': config
    }
    save_tasks()

    # 启动线程
    thread = threading.Thread(target=async_task_wrapper, args=(task_id, run_training_pipeline, config))
    thread.start()

    return jsonify({
        "message": "Training task submitted",
        "task_id": task_id,
        "status_url": f"/api/tasks/{task_id}"
    }), 202

@app.route('/api/inference', methods=['POST'])
def run_inference_api():
    """
    Run inference task (asynchronous).
    """
    config = request.json
    if not config:
        return jsonify({"error": "Request body must be JSON"}), 400

    model_id = config.get('model_id')
    dataset_id = config.get('dataset_id')

    if not model_id or not dataset_id:
        return jsonify({"error": "Missing model_id or dataset_id"}), 400

    model_dir = TRAINING_RUNS_DIR / model_id
    dataset_dir = PROCESSED_DATA_DIR / dataset_id

    if not model_dir.exists():
        return jsonify({"error": "Model not found"}), 404
    if not dataset_dir.exists():
        return jsonify({"error": "Dataset not found"}), 404

    task_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = ATTRIBUTION_RESULTS_DIR / f"{timestamp}_{task_id}"

    TASK_QUEUE[task_id] = {
        'status': 'pending',
        'created_at': timestamp,
        'type': 'inference',
        'config': config
    }
    save_tasks()

    def inference_worker():
        return run_inference_pipeline(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            output_dir=output_dir
        )

    thread = threading.Thread(target=async_task_wrapper, args=(task_id, inference_worker))
    thread.start()

    return jsonify({
        "message": "Inference task submitted",
        "task_id": task_id,
        "status_url": f"/api/tasks/{task_id}"
    }), 202

@app.route('/api/attribution_results', methods=['GET'])
def list_attribution_results():
    """
    List past attribution results.
    """
    results = []
    if ATTRIBUTION_RESULTS_DIR.exists():
        for res_dir in ATTRIBUTION_RESULTS_DIR.iterdir():
            if res_dir.is_dir():
                res_file = res_dir / "inference_results.json"
                if res_file.exists():
                    try:
                        with open(res_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Add metadata from directory name
                            parts = res_dir.name.split('_')
                            ts = parts[0]
                            tid = parts[1] if len(parts) > 1 else "unknown"
                            
                            results.append({
                                "id": res_dir.name,
                                "task_id": tid,
                                "created": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}",
                                "total_samples": data.get("total_samples", 0),
                                "label_distribution": data.get("label_distribution", {}),
                                "path": str(res_dir)
                            })
                    except Exception as e:
                        logging.warning(f"Failed to load result {res_dir}: {e}")
    
    results.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(results)

@app.route('/api/attribution_results/<result_id>', methods=['GET'])
def get_attribution_result_detail(result_id):
    """
    Get detailed attribution result.
    """
    target_dir = ATTRIBUTION_RESULTS_DIR / result_id
    res_file = target_dir / "inference_results.json"
    
    if not res_file.exists():
        return jsonify({"error": "Result not found"}), 404
        
    try:
        with open(res_file, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/attribution_results/<result_id>', methods=['DELETE'])
def delete_attribution_result(result_id):
    """
    Delete an attribution result (directory).
    """
    target_dir = ATTRIBUTION_RESULTS_DIR / result_id
    
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Result not found"}), 404
        
    try:
        shutil.rmtree(target_dir)
        return jsonify({"message": "Attribution result deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """
    生成分析报告。
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Task ID could be from a training task or an attribution/inference task
    task_id = data.get('task_id', 'Unknown')
    
    # Try to load real results if not provided in payload
    analysis_results = data.get('analysis_results', {})
    
    if not analysis_results.get('attributions'):
        # Check if task_id corresponds to an attribution result
        # Attribution result IDs are timestamps + task_ids usually, or just task_ids if we map them.
        # But frontend might pass a task_id from TASK_QUEUE.
        
        task = TASK_QUEUE.get(task_id)
        if task and task.get('type') == 'inference' and task.get('status') == 'completed':
             # Load inference results
             # The result field in task might contain the path or summary
             # Inference pipeline returns dict with 'total_samples', 'label_distribution', 'results'
             # Or it saves to file.
             
             # Let's check run_inference_pipeline return value. It returns dict.
             inf_res = task.get('result', {})
             
             # Convert to report format
             # Top attribution is the one with highest count in label_distribution? 
             # Or we look at individual high confidence samples?
             # Usually distribution.
             
             dist = inf_res.get('label_distribution', {})
             if dist:
                 sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                 top_attr = sorted_dist[0][0]
                 
                 # Create attribution list
                 attrs = []
                 total = inf_res.get('total_samples', 1)
                 for name, count in sorted_dist:
                     attrs.append({
                         "name": name,
                         "score": count / total, # Proportion as score
                         "risk": "High" # Placeholder
                     })
                 
                 analysis_results['top_attribution'] = top_attr
                 analysis_results['attributions'] = attrs
                 analysis_results['total_samples'] = total
    
    # Enrich with default structure if still missing
    if 'attributions' not in analysis_results:
        analysis_results['attributions'] = [{'name': 'Unknown', 'score': 0.0, 'risk': 'Low'}]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"report_{task_id}_{timestamp}.pdf"
    
    try:
        # Pass output_dir explicitly
        generator = ReportGenerator(output_dir=REPORTS_DIR)
        report_path = generator.generate_report(task_id, analysis_results, output_filename)
        
        return jsonify({
            "message": "Report generated successfully",
            "report_url": f"/reports/{output_filename}", # Frontend should prepend API URL if needed, or use relative
            "report_path": str(report_path)
        })
    except Exception as e:
        logging.error(f"Report generation failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/reports/<filename>', methods=['GET'])
def download_report(filename):
    """
    下载报告文件。
    """
    from flask import send_from_directory
    return send_from_directory(REPORTS_DIR, filename)

@app.route('/api/gangs', methods=['GET'])
def get_gangs():
    """
    Get the list of known APT gangs.
    """
    try:
        import json
        gangs_file = BASE_DIR / 'backend' / 'data' / 'gangs.json'
        if not gangs_file.exists():
             # Fallback if file not found (or path issue)
             gangs_file = Path(__file__).resolve().parent / 'data' / 'gangs.json'
        
        if gangs_file.exists():
            with open(gangs_file, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        else:
             return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务，监听所有网络接口的 5001 端口
    # Disable debug mode to prevent auto-reloading on file changes
    app.run(host='0.0.0.0', port=5001, debug=False)
