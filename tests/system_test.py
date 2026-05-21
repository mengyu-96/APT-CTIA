import requests
import time
import os
import glob
import json

# Configuration
API_URL = "http://127.0.0.1:5001/api"
DATASET_DIR = r"d:\git\APT归因\dataset_TXT"
MAX_FILES = 20  # Limit files for testing speed, or remove limit for full test

def check_backend():
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            print("[+] Backend is running.")
            return True
    except requests.exceptions.ConnectionError:
        print("[-] Backend is NOT running. Please start it first.")
        return False
    return False

def upload_and_preprocess():
    print(f"[*] Scanning {DATASET_DIR}...")
    files = glob.glob(os.path.join(DATASET_DIR, "*.txt")) + \
            glob.glob(os.path.join(DATASET_DIR, "*.pdf")) + \
            glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    if not files:
        print("[-] No files found in dataset directory.")
        return None

    # Select a subset for testing
    selected_files = files[:MAX_FILES]
    print(f"[*] Selected {len(selected_files)} files for upload.")

    files_payload = []
    # We need to open files in binary mode
    open_files = []
    try:
        for file_path in selected_files:
            f = open(file_path, 'rb')
            open_files.append(f)
            files_payload.append(('files', (os.path.basename(file_path), f, 'application/octet-stream')))

        print("[*] Uploading files and starting preprocessing...")
        response = requests.post(f"{API_URL}/preprocess", files=files_payload, timeout=300)
        
        if response.status_code == 202:
            data = response.json()
            task_id = data['task_id']
            print(f"[+] Preprocessing task submitted. Task ID: {task_id}")
            return task_id
        else:
            print(f"[-] Upload failed: {response.text}")
            return None
    finally:
        for f in open_files:
            f.close()

def wait_for_task(task_id):
    print(f"[*] Waiting for task {task_id} to complete...")
    while True:
        try:
            response = requests.get(f"{API_URL}/tasks/{task_id}")
            if response.status_code == 200:
                status = response.json().get('status')
                if status == 'completed':
                    print("[+] Task completed successfully.")
                    return response.json().get('result')
                elif status == 'failed':
                    print(f"[-] Task failed: {response.json().get('error')}")
                    return None
                else:
                    print(f"[*] Status: {status}...", end='\r')
                    time.sleep(2)
            else:
                print(f"[-] Failed to check status: {response.text}")
                return None
        except Exception as e:
            print(f"[-] Error polling task: {e}")
            return None

def train_model(processed_data_path):
    print("[*] Submitting training task...")
    config = {
        "processed_data_path": processed_data_path,
        "model_type": "GAT",
        "epochs": 10,  # Small number for quick testing
        "lr": 0.005,
        "batch_size": 16,
        "hidden_dim": 64,
        "dropout": 0.5
    }
    
    response = requests.post(f"{API_URL}/train", json=config)
    if response.status_code == 202:
        task_id = response.json().get('task_id')
        print(f"[+] Training task submitted. Task ID: {task_id}")
        return task_id
    else:
        print(f"[-] Training submission failed: {response.text}")
        return None

def generate_report(task_id, analysis_results):
    print("[*] Generating report...")
    payload = {
        "task_id": task_id,
        "analysis_results": analysis_results
    }
    response = requests.post(f"{API_URL}/generate_report", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"[+] Report generated: {data['report_path']}")
        return data['report_path']
    else:
        print(f"[-] Report generation failed: {response.text}")
        return None

def main():
    if not check_backend():
        return

    # 1. Preprocess
    preprocess_task_id = upload_and_preprocess()
    if not preprocess_task_id:
        return

    preprocess_result = wait_for_task(preprocess_task_id)
    if not preprocess_result:
        return
    
    processed_path = preprocess_result.get('output_path')
    print(f"[*] Processed data path: {processed_path}")

    # 2. Train
    train_task_id = train_model(processed_path)
    if not train_task_id:
        return

    train_result = wait_for_task(train_task_id)
    if not train_result:
        return

    # 3. Report
    # Construct a mock analysis result based on training output for the report
    # In a real scenario, we might want to run inference on new samples, 
    # but here we use the training/test evaluation as the "result" to report on.
    
    # We need to construct the input for generate_report based on what train_result returns
    # train_result contains 'test_accuracy', 'test_f1_weighted', 'classification_report', etc.
    
    analysis_results = {
        "top_attribution": "TEST_APT", # Mock
        "attributions": [],
        "confusion_matrix_plot": train_result.get('confusion_matrix_plot'),
        "history_plot": train_result.get('history_plot'),
        "iocs": {"IP": ["1.1.1.1"], "Hash": ["deadbeef"]}, # Mock IOCs
        "graph_data": {"nodes": [], "edges": []} # Mock Graph
    }
    
    # Populate attributions from classification report if available
    if 'classification_report' in train_result:
        cls_report = train_result['classification_report']
        # Find best f1
        best_f1 = -1
        top_name = "Unknown"
        
        for name, metrics in cls_report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                score = metrics['f1-score']
                analysis_results['attributions'].append({
                    "name": name,
                    "score": metrics.get('precision', 0),
                    "risk": "High" if score > 0.8 else "Medium"
                })
                if score > best_f1:
                    best_f1 = score
                    top_name = name
        
        analysis_results['top_attribution'] = top_name

    generate_report(train_task_id, analysis_results)
    print("\n[=] Full system test completed.")

if __name__ == "__main__":
    main()
