# backend/api.py
from flask import Flask, request, jsonify, send_file, abort
from pathlib import Path
import os
import shutil
import datetime
import logging
import time
from typing import Any

from core.preprocess import run_preprocessing_pipeline
from core.inference import run_inference_pipeline
from core.report_generator import ReportGenerator
from runtime_config import (
    ENABLE_INFERENCE,
    ENABLE_PREPROCESSING,
    ENABLE_TRAINING,
    build_runtime_config,
)

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


def _feature_disabled_response(feature: str):
    return jsonify({
        "error": f"{feature} is disabled by server configuration",
        "feature": feature,
    }), 403

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


@app.route('/api/runtime_config', methods=['GET'])
def get_runtime_config():
    return jsonify(build_runtime_config())


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
TASK_QUEUE_LOCK = threading.Lock()
TASK_SAVE_MIN_INTERVAL_SEC = 0.5
_LAST_TASK_SAVE_TS = 0.0
ACTIVE_TASK_STATUSES = {'pending', 'running'}


def _snapshot_tasks_unlocked():
    return json.dumps(TASK_QUEUE, indent=2, ensure_ascii=False)


def _read_json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open('r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as exc:
        logging.warning("Failed to read JSON from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Lightweight read-through caches for hot GET endpoints.
#
# Dataset listing and per-graph stats are read from disk on every poll, which
# becomes the dominant cost (and the source of frontend stutter) once there
# are many / large datasets. We cache by a freshness signature so results stay
# correct automatically: a file cache keys on (path, mtime, size); a directory
# cache keys on a signature built from each child's name + mtime. When the
# signature is unchanged the cached payload is reused with no disk work.
# ---------------------------------------------------------------------------
_FILE_JSON_CACHE: dict[str, tuple[tuple[float, int], Any]] = {}
_FILE_JSON_CACHE_LOCK = threading.Lock()
_DIR_PAYLOAD_CACHE: dict[str, tuple[Any, Any]] = {}
_DIR_PAYLOAD_CACHE_LOCK = threading.Lock()


def _read_json_cached(path: Path) -> Any:
    """Read a JSON file, reusing the parsed result while mtime/size are stable."""
    try:
        stat = path.stat()
    except OSError:
        return None
    sig = (stat.st_mtime, stat.st_size)
    key = str(path)
    with _FILE_JSON_CACHE_LOCK:
        cached = _FILE_JSON_CACHE.get(key)
        if cached and cached[0] == sig:
            return cached[1]
    data = _read_json_file(path)
    with _FILE_JSON_CACHE_LOCK:
        _FILE_JSON_CACHE[key] = (sig, data)
    return data


def _dir_signature(root: Path) -> tuple:
    """A cheap signature that changes when a directory's children change."""
    if not root.exists():
        return ()
    entries = []
    try:
        for child in root.iterdir():
            try:
                st_ = child.stat()
                entries.append((child.name, st_.st_mtime, st_.st_size))
            except OSError:
                continue
    except OSError:
        return ()
    entries.sort()
    return tuple(entries)


def _dir_cache_get(key: str, signature: Any) -> Any | None:
    with _DIR_PAYLOAD_CACHE_LOCK:
        cached = _DIR_PAYLOAD_CACHE.get(key)
        if cached and cached[0] == signature:
            return cached[1]
    return None


def _dir_cache_set(key: str, signature: Any, payload: Any) -> None:
    with _DIR_PAYLOAD_CACHE_LOCK:
        _DIR_PAYLOAD_CACHE[key] = (signature, payload)


def _find_inference_result_dir(task_id: str) -> Path | None:
    if not task_id or not ATTRIBUTION_RESULTS_DIR.exists():
        return None
    matches = sorted(ATTRIBUTION_RESULTS_DIR.glob(f"*_{task_id}"))
    for candidate in reversed(matches):
        if candidate.is_dir() and (candidate / 'inference_results.json').exists():
            return candidate
    return None


def _build_result_ref(task_type: str, result: Any, task: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(result, dict):
        return None

    if task_type == 'train':
        model_path = result.get('model_path')
        if model_path:
            return {"kind": "train_run", "path": str(Path(model_path).resolve().parent)}

    if task_type == 'inference':
        output_dir = result.get('output_dir') or task.get('output_dir')
        if not output_dir:
            found_dir = _find_inference_result_dir(str(task.get('id', '')))
            if found_dir:
                output_dir = str(found_dir)
        if output_dir:
            return {"kind": "inference_run", "path": str(Path(output_dir).resolve())}

    if task_type == 'preprocess':
        output_path = result.get('output_path') or task.get('output_dir')
        if output_path:
            return {"kind": "preprocess_run", "path": str(Path(output_path).resolve())}

    return None


def _summarize_task_result(task_type: str, result: Any, task: dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return result

    if task_type == 'train':
        return {
            "test_accuracy": result.get("test_accuracy", 0.0),
            "test_f1_weighted": result.get("test_f1_weighted", 0.0),
            "temperature": result.get("temperature"),
            "model_path": result.get("model_path"),
            "history_plot": result.get("history_plot"),
            "confusion_matrix_plot": result.get("confusion_matrix_plot"),
        }

    if task_type == 'inference':
        return {
            "total_samples": result.get("total_samples", 0),
            "label_distribution": result.get("label_distribution", {}),
            "output_dir": result.get("output_dir") or task.get("output_dir"),
        }

    if task_type == 'preprocess':
        return {
            "output_path": result.get("output_path"),
            "label_mapping_path": result.get("label_mapping_path"),
            "graph_stats_path": result.get("graph_stats_path"),
        }

    return result


def _load_task_result(task: dict[str, Any]) -> Any:
    ref = task.get('result_ref') or {}
    if not ref:
        return task.get('result_summary')

    path_str = ref.get('path')
    kind = ref.get('kind')
    if not path_str:
        return task.get('result_summary')

    path = Path(path_str)
    if kind == 'train_run':
        results = _read_json_file(path / 'results.json')
        return results if isinstance(results, dict) else task.get('result_summary')

    if kind == 'inference_run':
        results = _read_json_file(path / 'inference_results.json')
        if isinstance(results, dict):
            results.setdefault('output_dir', str(path))
            return results
        return task.get('result_summary')

    if kind == 'preprocess_run':
        summary = dict(task.get('result_summary') or {})
        summary.setdefault('output_path', str(path))
        return summary

    return task.get('result_summary')


def _normalize_task_record(task_id: str, task: dict[str, Any]) -> bool:
    changed = False
    task['id'] = task_id

    legacy_result = task.pop('result', None)
    if legacy_result is not None:
        task_type = task.get('type', 'unknown')
        task['result_summary'] = _summarize_task_result(task_type, legacy_result, task)
        result_ref = _build_result_ref(task_type, legacy_result, task)
        if result_ref:
            task['result_ref'] = result_ref
        changed = True

    if task.get('type') == 'inference' and not task.get('result_ref'):
        result_dir = _find_inference_result_dir(task_id)
        if result_dir:
            task['result_ref'] = {"kind": "inference_run", "path": str(result_dir.resolve())}
            task.setdefault('output_dir', str(result_dir.resolve()))
            changed = True

    return changed


def _task_list_item(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    item_type = task.get('type', 'unknown')
    config = task.get('config', {}) if isinstance(task.get('config'), dict) else {}
    task_info = {
        "id": task_id,
        "status": task.get('status'),
        "type": item_type,
        "created": task.get('created_at', ''),
        "progress": task.get('progress', 0),
        "message": task.get('message', ''),
        "error": task.get('error'),
        "name": task.get('dataset_name') or task.get('output_name') or f"{item_type} task",
        "files": task.get('file_count', len(task.get('files', [])) if isinstance(task.get('files'), list) else 0),
    }

    if item_type == 'train':
        task_info['model'] = config.get('model_type')
        task_info['dataset'] = config.get('dataset_name')
        task_info['name'] = f"Train {task_info['model']}"

    if item_type == 'inference':
        task_info['name'] = "Attribution Inference"
        task_info['model'] = config.get('model_id')
        task_info['dataset'] = config.get('dataset_id')

    result_summary = task.get('result_summary')
    if isinstance(result_summary, dict):
        if item_type == 'train':
            task_info['accuracy'] = result_summary.get('test_accuracy')
            task_info['f1'] = result_summary.get('test_f1_weighted')
        if item_type == 'inference':
            task_info['total_samples'] = result_summary.get('total_samples', 0)

    return task_info


def _task_detail_payload(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = dict(task)
    payload['id'] = task_id
    payload['result'] = _load_task_result(task)
    return payload


def set_task_fields(task_id, *, force_save=False, **fields):
    changed = False
    with TASK_QUEUE_LOCK:
        task = TASK_QUEUE.get(task_id)
        if not task:
            return False
        for key, value in fields.items():
            if task.get(key) != value:
                task[key] = value
                changed = True
    if changed:
        save_tasks(force=force_save)
    return changed

def load_tasks():
    global TASK_QUEUE
    if TASK_QUEUE_FILE.exists():
        try:
            content = TASK_QUEUE_FILE.read_text(encoding='utf-8')
            if content:
                with TASK_QUEUE_LOCK:
                    TASK_QUEUE = json.loads(content)
            
            # Check for stale async tasks left by a previous server process.
            with TASK_QUEUE_LOCK:
                for tid, task in TASK_QUEUE.items():
                    if task.get('status') in ACTIVE_TASK_STATUSES:
                        task['status'] = 'failed'
                        task['error'] = 'Server restarted before the asynchronous task completed'
                    _normalize_task_record(tid, task)
            save_tasks(force=True)
            logging.info(f"Loaded {len(TASK_QUEUE)} tasks from persistence.")
            
        except Exception as e:
            logging.error(f"Failed to load tasks: {e}")
            with TASK_QUEUE_LOCK:
                TASK_QUEUE = {}

def save_tasks(force=False):
    global _LAST_TASK_SAVE_TS
    try:
        now = time.monotonic()
        with TASK_QUEUE_LOCK:
            if not force and now - _LAST_TASK_SAVE_TS < TASK_SAVE_MIN_INTERVAL_SEC:
                return
            payload = _snapshot_tasks_unlocked()
            _LAST_TASK_SAVE_TS = now
        temp_file = TASK_QUEUE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(payload)
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
        set_task_fields(task_id, status='running', message='Running', force_save=True)
        result = func(*args, **kwargs)
        with TASK_QUEUE_LOCK:
            task = TASK_QUEUE.get(task_id)
            task_type = task.get('type', 'unknown') if task else 'unknown'
            task_for_summary = dict(task or {})
            task_for_summary['id'] = task_id
        result_ref = _build_result_ref(task_type, result, task_for_summary)
        result_summary = _summarize_task_result(task_type, result, task_for_summary)
        fields = {
            'status': 'completed',
            'progress': 100,
            'message': 'Completed',
            'completed_at': datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            'result_summary': result_summary,
        }
        if result_ref:
            fields['result_ref'] = result_ref
        set_task_fields(task_id, force_save=True, **fields)
        logging.info(f"Task {task_id} completed successfully.")
    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
        set_task_fields(task_id, status='failed', error=str(e), message='Failed', force_save=True)
    finally:
        save_tasks(force=True)


def has_processed_graphs(dataset_path: Path) -> bool:
    if not dataset_path.exists():
        return False
    graphs_dir = dataset_path / 'graphs'
    if graphs_dir.exists() and graphs_dir.is_dir():
        return any(graphs_dir.glob("*.pt"))
    graphs_pt = dataset_path / 'graphs.pt'
    return graphs_pt.exists() and graphs_pt.is_file()

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """
    获取所有任务列表 (简略信息)
    """
    task_type = request.args.get('type')
    active_only = request.args.get('active_only', '').lower() in {'1', 'true', 'yes'}
    statuses = {s.strip() for s in request.args.get('status', '').split(',') if s.strip()}
    limit = request.args.get('limit', type=int)
    tasks_list = []
    # Convert dict to list and sort by creation time (descending)
    # We assume keys are UUIDs. 
    # We need to look at 'created_at' or 'created' field.
    
    with TASK_QUEUE_LOCK:
        task_items = list(TASK_QUEUE.items())

    for t_id, t_data in task_items:
        status = t_data.get('status')
        item_type = t_data.get('type', 'unknown')
        if task_type and item_type != task_type:
            continue
        if active_only and status not in ACTIVE_TASK_STATUSES:
            continue
        if statuses and status not in statuses:
            continue
        tasks_list.append(_task_list_item(t_id, t_data))
        
    # Sort by created time (if available strings)
    tasks_list.sort(key=lambda x: x['created'], reverse=True)
    if limit and limit > 0:
        tasks_list = tasks_list[:limit]
        
    return jsonify(tasks_list)

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    查询异步任务的状态。
    """
    with TASK_QUEUE_LOCK:
        task = TASK_QUEUE.get(task_id)
        task = dict(task) if task else None
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(_task_detail_payload(task_id, task))

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """
    删除任务记录。
    """
    with TASK_QUEUE_LOCK:
        existed = task_id in TASK_QUEUE
        if existed:
            del TASK_QUEUE[task_id]
    if existed:
        save_tasks(force=True)
        return jsonify({"message": "Task deleted successfully"})
    return jsonify({"error": "Task not found"}), 404

@app.route('/api/raw_datasets', methods=['GET'])
def list_raw_datasets():
    """
    列出所有未处理的原始数据集 (dataset_TXT 下的子目录).
    """
    # Primary root is dataset_TXT
    root = BASE_DIR / 'dataset_TXT'

    cache_key = "raw_datasets"
    signature = _dir_signature(root)
    cached = _dir_cache_get(cache_key, signature)
    if cached is not None:
        return jsonify(cached)

    datasets = []
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

    _dir_cache_set(cache_key, signature, datasets)
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

    cache_key = f"raw_files:{path}"
    signature = _dir_signature(path)
    cached = _dir_cache_get(cache_key, signature)
    if cached is not None:
        return jsonify(cached)

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

    payload = {"files": files, "total_shown": count}
    _dir_cache_set(cache_key, signature, payload)
    return jsonify(payload)

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
    if not ENABLE_PREPROCESSING:
        return _feature_disabled_response("preprocessing")

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
        
        with TASK_QUEUE_LOCK:
            TASK_QUEUE[task_id] = {
                'status': 'pending',
                'created_at': timestamp,
                'type': 'preprocess',
                'source': str(path),
                'output_name': output_name or dir_name,
                'output_dir': str(output_dir),
            }
        save_tasks(force=True)
        
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

                set_task_fields(task_id, progress=pct, message=msg)

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

        with TASK_QUEUE_LOCK:
            TASK_QUEUE[task_id] = {
                'status': 'pending',
                'created_at': timestamp,
                'type': 'preprocess',
                'source': 'upload',
                'output_name': output_name or dir_name,
                'output_dir': str(output_dir),
                'file_count': len(saved_files),
            }
        save_tasks(force=True)

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

                    set_task_fields(task_id, progress=pct, message=msg)

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

def _datasets_signature() -> tuple:
    """Signature covering the dataset roots and the stats files inside each."""
    sig = [_dir_signature(PROCESSED_DATA_DIR)]
    if PROCESSED_DATA_DIR.exists():
        for run_dir in sorted(PROCESSED_DATA_DIR.iterdir()):
            if run_dir.is_dir():
                for probe in (run_dir / "graph_stats.json", run_dir / "graphs", run_dir / "metadata.json"):
                    try:
                        st_ = probe.stat()
                        sig.append((probe.name, run_dir.name, st_.st_mtime))
                    except OSError:
                        continue
    return tuple(sig)


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """
    列出所有已处理的数据集 (Grouped by preprocessing run)。
    """
    include_raw_stats = request.args.get('include_raw_stats', '').lower() in {'1', 'true', 'yes'}

    cache_key = f"datasets:{include_raw_stats}"
    signature = _datasets_signature()
    cached = _dir_cache_get(cache_key, signature)
    if cached is not None:
        return jsonify(cached)

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
                                }
                                if include_raw_stats:
                                    dataset_info["stats"]["raw_stats"] = stats_data
                        except Exception as e:
                            logging.warning(f"Failed to load stats for {run_dir}: {e}")
                            
                    datasets.append(dataset_info)
    
    # Sort by created desc
    datasets.sort(key=lambda x: x['created'], reverse=True)
    _dir_cache_set(cache_key, signature, datasets)
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
        stats_list = _read_json_cached(stats_file)
        if isinstance(stats_list, list):
            # report_id -> stat (report_id is recorded per graph in preprocess.py)
            for s in stats_list:
                stats_map[s.get('report_id')] = s

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
    
    if stats_map:
        return jsonify(list(stats_map.values()))

    if stats_file.exists():
        data = _read_json_cached(stats_file)
        if isinstance(data, list):
            return jsonify(data)

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
    将数据集划分为训练集/验证集/测试集。
    当前接口基于已生成的图文件列表落盘 `splits.json`，供前端或离线流程复用。
    """
    target_dir = PROCESSED_DATA_DIR / dataset_id
    if not target_dir.exists():
        return jsonify({"error": "Dataset not found"}), 404
        
    if not ENABLE_TRAINING:
        return _feature_disabled_response("training")

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
             
        # Persist a reusable filename-level split for downstream tooling.
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
        data = _read_json_cached(stats_file)
        if data is None:
            return jsonify({"error": "Failed to read stats"}), 500
        return jsonify(data)

    return jsonify({"error": "Stats not found"}), 404

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    列出所有已训练的模型。
    """
    include_report = request.args.get('include_report', '').lower() in {'1', 'true', 'yes'}
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

                        model_info = {
                            "id": run_dir.name,
                            "name": model_name,
                            "type": results.get('config', {}).get('model_type', 'GNN'),
                            "dataset_id": results.get('config', {}).get('dataset_id', 'Unknown'),
                            "dataset_name": results.get('config', {}).get('dataset_name', 'Unknown'),
                            "accuracy": results.get('test_accuracy', 0.0),
                            "f1_score": results.get('test_f1_weighted', 0.0),
                            "epochs": results.get('config', {}).get('epochs', 0),
                            "batch_size": results.get('config', {}).get('batch_size', 32),
                            "created": datetime.datetime.fromtimestamp(run_dir.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                            "status": "Completed",
                            "path": str(run_dir)
                        }
                        if include_report:
                            model_info["classification_report"] = results.get('classification_report', {})
                        models.append(model_info)
                    except Exception as e:
                        logging.warning(f"Failed to parse results for {run_dir}: {e}")
    
    # 按创建时间倒序
    models.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(models)

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_detail(model_id):
    target_dir = TRAINING_RUNS_DIR / model_id
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Model not found"}), 404

    results_file = target_dir / 'results.json'
    if not results_file.exists():
        return jsonify({"error": "Results file not found"}), 404

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        model_name = results.get('custom_name')
        if not model_name:
            model_name = f"{results.get('config', {}).get('model_type', 'Unknown')}_{target_dir.name}"

        return jsonify({
            "id": target_dir.name,
            "name": model_name,
            "type": results.get('config', {}).get('model_type', 'GNN'),
            "dataset_id": results.get('config', {}).get('dataset_id', 'Unknown'),
            "dataset_name": results.get('config', {}).get('dataset_name', 'Unknown'),
            "accuracy": results.get('test_accuracy', 0.0),
            "f1_score": results.get('test_f1_weighted', 0.0),
            "epochs": results.get('config', {}).get('epochs', 0),
            "batch_size": results.get('config', {}).get('batch_size', 32),
            "created": datetime.datetime.fromtimestamp(target_dir.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            "status": "Completed",
            "classification_report": results.get('classification_report', {}),
            "path": str(target_dir)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    if not ENABLE_TRAINING:
        return _feature_disabled_response("training")

    config = request.json
    if not config:
        return jsonify({"error": "Request body must be a JSON with training configuration"}), 400

    processed_data_path = config.get('processed_data_path')
    if not processed_data_path or not Path(processed_data_path).exists():
        return jsonify({"error": f"Processed data path is missing or does not exist: {processed_data_path}"}), 400
    if not has_processed_graphs(Path(processed_data_path)):
        return jsonify({"error": f"Processed dataset is empty or graph construction has not completed yet: {processed_data_path}"}), 400

    try:
        batch_size = int(config.get('batch_size', 32))
    except (TypeError, ValueError):
        return jsonify({"error": "batch_size must be an integer"}), 400
    if batch_size < 1:
        return jsonify({"error": "batch_size must be greater than 0"}), 400
    config['batch_size'] = batch_size

    # 将主输出目录注入配置中
    config['base_output_dir'] = str(TRAINING_RUNS_DIR)

    # 创建异步任务
    task_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with TASK_QUEUE_LOCK:
        TASK_QUEUE[task_id] = {
            'status': 'pending',
            'created_at': timestamp,
            'type': 'train',
            'config': config,
            'output_name': f"train_{config.get('model_type', 'model')}_{timestamp}",
        }
    save_tasks(force=True)

    # 启动线程
    from core.train import run_training_pipeline
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
    if not ENABLE_INFERENCE:
        return _feature_disabled_response("inference")

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
    if not has_processed_graphs(dataset_dir):
        return jsonify({"error": "Dataset is empty or preprocessing has not completed yet"}), 400

    task_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = ATTRIBUTION_RESULTS_DIR / f"{timestamp}_{task_id}"

    with TASK_QUEUE_LOCK:
        TASK_QUEUE[task_id] = {
            'status': 'pending',
            'created_at': timestamp,
            'type': 'inference',
            'config': config,
            'output_dir': str(output_dir),
        }
    save_tasks(force=True)

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

# Per-sample heavy fields the UI never consumes in the history view. Dropping
# them shrinks an inference payload by well over half (a single results file can
# be hundreds of MB; graph/attention blobs dominate it).
_HEAVY_SAMPLE_FIELDS = ("graph_data", "attention_data")


def _result_created_from_dirname(name: str) -> tuple[str, str]:
    parts = name.split('_')
    ts = parts[0]
    tid = parts[1] if len(parts) > 1 else "unknown"
    try:
        created = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    except Exception:
        created = ts
    return tid, created


def _ensure_result_summary(res_dir: Path) -> dict | None:
    """Return a small summary for a result dir, building a sidecar once.

    The full ``inference_results.json`` can be hundreds of MB, so we never read
    it just to list results. We persist a tiny ``summary.json`` next to it and
    reuse that on every later call. The sidecar is rebuilt if it is older than
    the source file.
    """
    res_file = res_dir / "inference_results.json"
    if not res_file.exists():
        return None

    summary_file = res_dir / "summary.json"
    try:
        if summary_file.exists() and summary_file.stat().st_mtime >= res_file.stat().st_mtime:
            cached = _read_json_cached(summary_file)
            if isinstance(cached, dict):
                return cached
    except OSError:
        pass

    # Build the sidecar from the heavy source once.
    data = _read_json_file(res_file)
    if not isinstance(data, dict):
        return None
    tid, created = _result_created_from_dirname(res_dir.name)
    summary = {
        "id": res_dir.name,
        "task_id": tid,
        "created": created,
        "total_samples": data.get("total_samples", 0),
        "label_distribution": data.get("label_distribution", {}),
        "path": str(res_dir),
    }
    try:
        tmp = summary_file.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False)
        tmp.replace(summary_file)
    except Exception as exc:
        logging.warning("Failed to write summary sidecar for %s: %s", res_dir, exc)
    return summary


@app.route('/api/attribution_results', methods=['GET'])
def list_attribution_results():
    """List past attribution results (lightweight summaries only)."""
    if not ATTRIBUTION_RESULTS_DIR.exists():
        return jsonify([])

    signature = _dir_signature(ATTRIBUTION_RESULTS_DIR)
    cached = _dir_cache_get("attribution_results", signature)
    if cached is not None:
        return jsonify(cached)

    results = []
    for res_dir in ATTRIBUTION_RESULTS_DIR.iterdir():
        if res_dir.is_dir():
            summary = _ensure_result_summary(res_dir)
            if summary:
                results.append(summary)

    results.sort(key=lambda x: x['created'], reverse=True)
    _dir_cache_set("attribution_results", signature, results)
    return jsonify(results)


@app.route('/api/attribution_results/<result_id>', methods=['GET'])
def get_attribution_result_detail(result_id):
    """Detail without the heavy per-sample graph/attention blobs.

    Returns distribution + per-sample table fields + explanation, but strips
    ``graph_data``/``attention_data`` so the payload is small enough to cache
    and ship quickly. Pass ``?full=1`` to get the untouched file.
    """
    target_dir = ATTRIBUTION_RESULTS_DIR / result_id
    res_file = target_dir / "inference_results.json"
    if not res_file.exists():
        return jsonify({"error": "Result not found"}), 404

    want_full = request.args.get('full', '').lower() in {'1', 'true', 'yes'}
    if want_full:
        data = _read_json_file(res_file)
        if not isinstance(data, dict):
            return jsonify({"error": "Failed to read result"}), 500
        return jsonify(data)

    # Serve a cached slim sidecar (table fields + distribution, no explanation /
    # graph / attention) so repeat views never re-read the huge source file.
    slim_file = target_dir / "detail_slim.json"
    try:
        if slim_file.exists() and slim_file.stat().st_mtime >= res_file.stat().st_mtime:
            cached = _read_json_cached(slim_file)
            if isinstance(cached, dict):
                return jsonify(cached)
    except OSError:
        pass

    data = _read_json_file(res_file)
    if not isinstance(data, dict):
        return jsonify({"error": "Failed to read result"}), 500

    slim = {
        "total_samples": data.get("total_samples", 0),
        "label_distribution": data.get("label_distribution", {}),
        "output_dir": data.get("output_dir"),
        "results": [
            {
                "report_id": s.get("report_id"),
                "predicted_label": s.get("predicted_label"),
                "confidence": s.get("confidence"),
                "top3": s.get("top3"),
            }
            for s in data.get("results", [])
        ],
    }
    try:
        tmp = slim_file.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(slim, f, ensure_ascii=False)
        tmp.replace(slim_file)
    except Exception as exc:
        logging.warning("Failed to write slim detail for %s: %s", target_dir, exc)
    return jsonify(slim)


@app.route('/api/attribution_results/<result_id>/sample/<report_id>', methods=['GET'])
def get_attribution_sample(result_id, report_id):
    """Return a single sample's explanation, loaded on demand.

    Backed by a one-time ``explanations.json`` sidecar (all samples minus the
    heavy graph/attention blobs), so after the first build every sample lookup
    is a small cached read instead of re-parsing the multi-hundred-MB source.
    """
    target_dir = ATTRIBUTION_RESULTS_DIR / result_id
    res_file = target_dir / "inference_results.json"
    if not res_file.exists():
        return jsonify({"error": "Result not found"}), 404

    expl_file = target_dir / "explanations.json"
    index = None
    try:
        if expl_file.exists() and expl_file.stat().st_mtime >= res_file.stat().st_mtime:
            cached = _read_json_cached(expl_file)
            if isinstance(cached, dict):
                index = cached
    except OSError:
        pass

    if index is None:
        data = _read_json_file(res_file)
        if not isinstance(data, dict):
            return jsonify({"error": "Failed to read result"}), 500
        index = {
            str(s.get("report_id")): {k: v for k, v in s.items() if k not in _HEAVY_SAMPLE_FIELDS}
            for s in data.get("results", [])
        }
        try:
            tmp = expl_file.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False)
            tmp.replace(expl_file)
        except Exception as exc:
            logging.warning("Failed to write explanations sidecar for %s: %s", target_dir, exc)

    sample = index.get(str(report_id))
    if sample is None:
        return jsonify({"error": "Sample not found"}), 404
    return jsonify(sample)

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
        
        with TASK_QUEUE_LOCK:
            task = TASK_QUEUE.get(task_id)
        if task and task.get('type') == 'inference' and task.get('status') == 'completed':
             inf_res = _load_task_result(task) or {}
             
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

    result_dir = ATTRIBUTION_RESULTS_DIR / task_id
    if not result_dir.exists():
        found_dir = _find_inference_result_dir(str(task_id))
        if found_dir:
            result_dir = found_dir
    result_file = result_dir / "inference_results.json"
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            inf_res = json.load(f)
        dist = inf_res.get('label_distribution', {})
        total = inf_res.get('total_samples', 1)
        if dist and not analysis_results.get('attributions'):
            sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            analysis_results['top_attribution'] = sorted_dist[0][0]
            analysis_results['attributions'] = [
                {
                    "name": name,
                    "score": count / total,
                    "risk": "High" if idx == 0 else "Medium"
                }
                for idx, (name, count) in enumerate(sorted_dist)
            ]
            analysis_results['total_samples'] = total
        sample_results = sorted(
            inf_res.get('results', []),
            key=lambda item: item.get('confidence', 0),
            reverse=True
        )
        if sample_results:
            analysis_results['sample_explanations'] = sample_results[:3]
            analysis_results['explanation'] = analysis_results.get('explanation') or sample_results[0].get('explanation', {})
            analysis_results['graph_data'] = analysis_results.get('graph_data') or sample_results[0].get('graph_data')
            analysis_results['attention_data'] = analysis_results.get('attention_data') or sample_results[0].get('attention_data')
    
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
