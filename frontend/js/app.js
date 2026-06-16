document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    initFileUpload();
    initFeatureTabs();
    initClustering();
    initAttribution();
    initGangStorage();
    initReport();
    initModelManagement();
});

window.uploadedSamples = [];
window.uploadedDatasets = [];

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');

    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const pageName = this.getAttribute('data-page');

            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            pages.forEach(page => {
                page.classList.remove('active');
                if (page.id === 'page-' + pageName) {
                    page.classList.add('active');
                }
            });

            if (pageName === 'case-studies') {
                updateGangList();
            }
        });
    });

    const homeFeatureItems = document.querySelectorAll('.feature-module');
    homeFeatureItems.forEach((item, index) => {
        item.addEventListener('click', function() {
            const pageMap = [
                'upload',
                'models',
                'features',
                'clustering',
                'attribution',
                'case-studies',
                'dataset',
                'report'
            ];
            const targetPage = pageMap[index];
            if (targetPage) {
                navItems.forEach(nav => {
                    if (nav.getAttribute('data-page') === targetPage) {
                        nav.click();
                    }
                });
            }
        });
    });
}

function updateDatasetSelect() {
    const datasetSelect = document.getElementById('datasetSelect');
    const featuresDatasetSelect = document.getElementById('featuresDatasetSelect');

    // 更新模型训练模块的数据集选择
    if (datasetSelect) {
        const currentValue = datasetSelect.value;
        datasetSelect.innerHTML = '<option value="all">所有数据集</option>';

        window.uploadedDatasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name} (${dataset.sampleCount} 样本)`;
            datasetSelect.appendChild(option);
        });

        if (currentValue !== 'all' && currentValue !== 'recent') {
            datasetSelect.value = currentValue;
        }
    }

    // 更新特征提取模块的数据集选择
    if (featuresDatasetSelect) {
        const currentValue = featuresDatasetSelect.value;
        featuresDatasetSelect.innerHTML = '<option value="all">所有数据集</option>';

        window.uploadedDatasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name} (${dataset.sampleCount} 样本)`;
            featuresDatasetSelect.appendChild(option);
        });

        if (currentValue !== 'all') {
            featuresDatasetSelect.value = currentValue;
        }
    }
}

function initFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');
        const clearButton = document.querySelector('.upload-list button');

        if (!uploadArea || !fileInput || !fileList) return;

        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });

        if (clearButton) {
            clearButton.addEventListener('click', function() {
                fileList.innerHTML = `
                    <div style="text-align: center; padding: 3rem; color: #aaa;">
                        <i class="fas fa-box" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                        <p>暂无上传的样本</p>
                    </div>
                `;
                fileInput.value = '';
                window.uploadedSamples = [];
                updateSampleSelect();
            });
        }

        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#764ba2';
            this.style.background = 'rgba(118, 75, 162, 0.1)';
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = 'rgba(102, 126, 234, 0.05)';
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = 'rgba(102, 126, 234, 0.05)';
            
            const files = e.dataTransfer.files;
            handleFiles(files);
        });

        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        function handleFiles(files) {
            // 清除默认消息
            if (fileList.querySelector('div[style*="text-align: center"]')) {
                fileList.innerHTML = '';
            }
            
            Array.from(files).forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.dataset.fileName = file.name;
                
                const icon = getFileIcon(file.name);
                const fileSize = formatFileSize(file.size);
                
                fileItem.innerHTML = `
                    <i class="${icon}"></i>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${fileSize}</div>
                        <div class="file-progress" style="display: none;">
                            <div class="progress-bar">
                                <div class="progress-fill"></div>
                            </div>
                            <div class="progress-text">预处理中...</div>
                        </div>
                    </div>
                    <div class="file-actions">
                        <button class="btn-preprocess" style="background: rgba(244, 67, 54, 0.2); color: #F44336; border: 1px solid #F44336; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; transition: all 0.3s ease;">预处理</button>
                    </div>
                    <div class="file-status uploading">上传中...</div>
                `;
                
                fileList.appendChild(fileItem);
                
                const preprocessBtn = fileItem.querySelector('.btn-preprocess');
                preprocessBtn.addEventListener('click', function() {
                    startPreprocessing(fileItem, file);
                });
                
                setTimeout(() => {
                    const status = fileItem.querySelector('.file-status');
                    status.textContent = '待预处理';
                    status.className = 'file-status pending';
                }, 2000);
            });
        }

        function startPreprocessing(fileItem, file) {
            const preprocessBtn = fileItem.querySelector('.btn-preprocess');
            const progressContainer = fileItem.querySelector('.file-progress');
            const progressFill = fileItem.querySelector('.progress-fill');
            const progressText = fileItem.querySelector('.progress-text');
            const status = fileItem.querySelector('.file-status');
            
            if (preprocessBtn.disabled) return;
            
            preprocessBtn.disabled = true;
            preprocessBtn.textContent = '预处理中...';
            preprocessBtn.style.opacity = '0.5';
            
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            progressText.textContent = '预处理中...';
            
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    
                    setTimeout(() => {
                        progressFill.style.width = '100%';
                        progressText.textContent = '预处理成功';
                        progressText.style.color = '#4CAF50';
                        
                        status.textContent = '预处理完成';
                        status.className = 'file-status completed';
                        
                        preprocessBtn.textContent = '已完成';
                        preprocessBtn.style.background = 'rgba(76, 175, 80, 0.2)';
                        preprocessBtn.style.color = '#4CAF50';
                        preprocessBtn.style.borderColor = '#4CAF50';
                        
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                        }, 500);
                        
                        window.uploadedSamples.push({
                            name: file.name,
                            size: formatFileSize(file.size),
                            type: file.type || getFileType(file.name)
                        });
                        
                        updateSampleSelect();
                    }, 300);
                } else {
                    progressFill.style.width = progress + '%';
                }
            }, 200);
        }

        function getFileType(filename) {
            const ext = filename.split('.').pop().toLowerCase();
            const types = {
                'exe': 'application/x-msdownload',
                'dll': 'application/x-msdownload',
                'doc': 'application/msword',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'xls': 'application/vnd.ms-excel',
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'pdf': 'application/pdf',
                'zip': 'application/zip',
                'rar': 'application/x-rar-compressed'
            };
            return types[ext] || 'application/octet-stream';
        }

        function getFileIcon(filename) {
            const ext = filename.split('.').pop().toLowerCase();
            const icons = {
                'exe': 'fas fa-file-code',
                'dll': 'fas fa-file-code',
                'doc': 'fas fa-file-word',
                'docx': 'fas fa-file-word',
                'xls': 'fas fa-file-excel',
                'xlsx': 'fas fa-file-excel',
                'pdf': 'fas fa-file-pdf',
                'zip': 'fas fa-file-archive',
                'rar': 'fas fa-file-archive'
            };
            return icons[ext] || 'fas fa-file';
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    }

function initFeatureTabs() {
    const tabBtns = document.querySelectorAll('.feature-tabs .feature-tab');
    const panels = document.querySelectorAll('.feature-panel');
    const extractAllFeaturesBtn = document.getElementById('extractAllFeatures');
    const featuresDatasetSelect = document.getElementById('featuresDatasetSelect');

    if (tabBtns.length === 0) return;

    initCharts();
    updateSampleSelect();
    updateDatasetSelect(); // 初始化数据集选择

    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            panels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === tabName + '-features') {
                    panel.classList.add('active');
                }
            });

            if (tabName === 'all') {
                const overallPieChart = Chart.getChart('overallPieChart');
                if (overallPieChart) overallPieChart.update('active');
            } else if (tabName === 'static') {
                const staticBarChart = Chart.getChart('staticBarChart');
                if (staticBarChart) staticBarChart.update('active');
            } else if (tabName === 'dynamic') {
                const dynamicBarChart = Chart.getChart('dynamicBarChart');
                if (dynamicBarChart) dynamicBarChart.update('active');
            } else if (tabName === 'network') {
                const networkBarChart = Chart.getChart('networkBarChart');
                if (networkBarChart) networkBarChart.update('active');
            }
        });
    });

    // 添加提取所有特征按钮的事件监听器
    if (extractAllFeaturesBtn && featuresDatasetSelect) {
        extractAllFeaturesBtn.addEventListener('click', function() {
            const selectedDataset = featuresDatasetSelect.value;
            
            if (selectedDataset === 'all') {
                if (window.uploadedDatasets.length === 0) {
                    alert('请先上传数据集');
                    return;
                }
                alert('正在从所有数据集提取特征...');
            } else {
                const dataset = window.uploadedDatasets.find(d => d.id == selectedDataset);
                if (dataset) {
                    alert(`正在从数据集 "${dataset.name}" 提取所有特征...`);
                } else {
                    alert('请选择有效的数据集');
                    return;
                }
            }

            // 模拟特征提取过程
            const activePanel = document.querySelector('.feature-panel.active');
            if (activePanel) {
                const panelId = activePanel.id;
                const visualizationContainer = activePanel.querySelector('.visualization-container');
                
                if (visualizationContainer) {
                    visualizationContainer.innerHTML = '<div style="text-align: center; padding: 3rem;"><i class="fas fa-spinner fa-spin" style="font-size: 3rem; color: #667eea;"></i><p style="margin-top: 1rem;">正在提取特征...</p></div>';
                }
            }

            // 2秒后恢复图表显示
            setTimeout(() => {
                initCharts();
                const activePanel = document.querySelector('.feature-panel.active');
                if (activePanel) {
                    const tabName = activePanel.id.replace('-features', '');
                    if (tabName === 'all') {
                        showAllFeaturesCharts();
                        const overallPieChart = Chart.getChart('overallPieChart');
                        const staticPieChart = Chart.getChart('staticPieChart');
                        const dynamicPieChart = Chart.getChart('dynamicPieChart');
                        const networkPieChart = Chart.getChart('networkPieChart');
                        
                        if (overallPieChart) overallPieChart.update('active');
                        if (staticPieChart) staticPieChart.update('active');
                        if (dynamicPieChart) dynamicPieChart.update('active');
                        if (networkPieChart) networkPieChart.update('active');
                    } else if (tabName === 'static') {
                        showStaticFeaturesChart();
                        const staticBarChart = Chart.getChart('staticBarChart');
                        if (staticBarChart) staticBarChart.update('active');
                    } else if (tabName === 'dynamic') {
                        showDynamicFeaturesChart();
                        const dynamicBarChart = Chart.getChart('dynamicBarChart');
                        if (dynamicBarChart) dynamicBarChart.update('active');
                    } else if (tabName === 'network') {
                        showNetworkFeaturesChart();
                        const networkBarChart = Chart.getChart('networkBarChart');
                        if (networkBarChart) networkBarChart.update('active');
                    }
                }
                alert('特征提取完成！');
            }, 2000);
        });
    }
}

function updateSampleSelect() {
    const sampleSelect = document.getElementById('sampleSelect');
    if (!sampleSelect) return;

    sampleSelect.innerHTML = '<option value="all">所有上传样本</option>';

    if (window.uploadedSamples.length === 0) {
        sampleSelect.innerHTML += '<option value="recent">暂无上传样本</option>';
    } else {
        window.uploadedSamples.forEach((sample, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${sample.name} (${sample.size})`;
            sampleSelect.appendChild(option);
        });
    }

    sampleSelect.onchange = function() {
        updateChartsForSample(this.value);
    };
}

function updateChartsForSample(sampleValue) {
    const activePanel = document.querySelector('.feature-panel.active');
    if (!activePanel) return;

    if (activePanel.id === 'all-features') {
        updateAllFeaturesCharts(sampleValue);
    } else if (activePanel.id === 'static-features') {
        updateStaticFeaturesChart(sampleValue);
    } else if (activePanel.id === 'dynamic-features') {
        updateDynamicFeaturesChart(sampleValue);
    } else if (activePanel.id === 'network-features') {
        updateNetworkFeaturesChart(sampleValue);
    }
}

function updateAllFeaturesCharts(sampleValue) {
    const overallPieChart = Chart.getChart('overallPieChart');
    const staticPieChart = Chart.getChart('staticPieChart');
    const dynamicPieChart = Chart.getChart('dynamicPieChart');
    const networkPieChart = Chart.getChart('networkPieChart');

    if (overallPieChart) {
        if (sampleValue === 'all') {
            overallPieChart.data.datasets[0].data = [450, 535, 355];
        } else {
            const sampleIndex = parseInt(sampleValue);
            const randomFactor = (sampleIndex + 1) * 0.1;
            overallPieChart.data.datasets[0].data = [
                Math.round(450 * (1 + randomFactor)),
                Math.round(535 * (1 + randomFactor * 0.8)),
                Math.round(355 * (1 + randomFactor * 0.6))
            ];
        }
        overallPieChart.update('active');
    }

    if (staticPieChart) {
        if (sampleValue === 'all') {
            staticPieChart.data.datasets[0].data = [85, 120, 200, 45];
        } else {
            const sampleIndex = parseInt(sampleValue);
            const randomFactor = (sampleIndex + 1) * 0.1;
            staticPieChart.data.datasets[0].data = [
                Math.round(85 * (1 + randomFactor)),
                Math.round(120 * (1 + randomFactor * 0.8)),
                Math.round(200 * (1 + randomFactor * 0.6)),
                Math.round(45 * (1 + randomFactor * 0.5))
            ];
        }
        staticPieChart.update('active');
    }

    if (dynamicPieChart) {
        if (sampleValue === 'all') {
            dynamicPieChart.data.datasets[0].data = [150, 95, 180, 110];
        } else {
            const sampleIndex = parseInt(sampleValue);
            const randomFactor = (sampleIndex + 1) * 0.1;
            dynamicPieChart.data.datasets[0].data = [
                Math.round(150 * (1 + randomFactor)),
                Math.round(95 * (1 + randomFactor * 0.8)),
                Math.round(180 * (1 + randomFactor * 0.6)),
                Math.round(110 * (1 + randomFactor * 0.5))
            ];
        }
        dynamicPieChart.update('active');
    }

    if (networkPieChart) {
        if (sampleValue === 'all') {
            networkPieChart.data.datasets[0].data = [75, 130, 90, 60];
        } else {
            const sampleIndex = parseInt(sampleValue);
            const randomFactor = (sampleIndex + 1) * 0.1;
            networkPieChart.data.datasets[0].data = [
                Math.round(75 * (1 + randomFactor)),
                Math.round(130 * (1 + randomFactor * 0.8)),
                Math.round(90 * (1 + randomFactor * 0.6)),
                Math.round(60 * (1 + randomFactor * 0.5))
            ];
        }
        networkPieChart.update('active');
    }
}

function updateStaticFeaturesChart(sampleValue) {
    const staticBarChart = Chart.getChart('staticBarChart');
    if (!staticBarChart) return;

    if (sampleValue === 'all') {
        staticBarChart.data.datasets[0].data = [85, 120, 200, 45];
    } else {
        const sampleIndex = parseInt(sampleValue);
        const randomFactor = (sampleIndex + 1) * 0.1;
        staticBarChart.data.datasets[0].data = [
            Math.round(85 * (1 + randomFactor)),
            Math.round(120 * (1 + randomFactor * 0.8)),
            Math.round(200 * (1 + randomFactor * 0.6)),
            Math.round(45 * (1 + randomFactor * 0.5))
        ];
    }
    staticBarChart.update('active');
}

function updateDynamicFeaturesChart(sampleValue) {
    const dynamicBarChart = Chart.getChart('dynamicBarChart');
    if (!dynamicBarChart) return;

    if (sampleValue === 'all') {
        dynamicBarChart.data.datasets[0].data = [150, 95, 180, 110];
    } else {
        const sampleIndex = parseInt(sampleValue);
        const randomFactor = (sampleIndex + 1) * 0.1;
        dynamicBarChart.data.datasets[0].data = [
            Math.round(150 * (1 + randomFactor)),
            Math.round(95 * (1 + randomFactor * 0.8)),
            Math.round(180 * (1 + randomFactor * 0.6)),
            Math.round(110 * (1 + randomFactor * 0.5))
        ];
    }
    dynamicBarChart.update('active');
}

function updateNetworkFeaturesChart(sampleValue) {
    const networkBarChart = Chart.getChart('networkBarChart');
    if (!networkBarChart) return;

    if (sampleValue === 'all') {
        networkBarChart.data.datasets[0].data = [75, 130, 90, 60];
    } else {
        const sampleIndex = parseInt(sampleValue);
        const randomFactor = (sampleIndex + 1) * 0.1;
        networkBarChart.data.datasets[0].data = [
            Math.round(75 * (1 + randomFactor)),
            Math.round(130 * (1 + randomFactor * 0.8)),
            Math.round(90 * (1 + randomFactor * 0.6)),
            Math.round(60 * (1 + randomFactor * 0.5))
        ];
    }
    networkBarChart.update('active');
}

function initCharts() {
    createStaticBarChart();
    createDynamicBarChart();
    createNetworkBarChart();
    createOverallPieChart();
}

function createStaticBarChart() {
    const ctx = document.getElementById('staticBarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['文件信息', '导入/导出表', '字符串提取', '数字签名'],
            datasets: [{
                label: '特征数量',
                data: [85, 120, 200, 45],
                backgroundColor: [
                    'rgba(0, 230, 118, 0.7)',
                    'rgba(0, 200, 83, 0.7)',
                    'rgba(0, 150, 136, 0.7)',
                    'rgba(0, 121, 107, 0.7)'
                ],
                borderColor: [
                    'rgba(0, 230, 118, 1)',
                    'rgba(0, 200, 83, 1)',
                    'rgba(0, 150, 136, 1)',
                    'rgba(0, 121, 107, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: '静态特征统计',
                    color: '#00e676',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(0, 230, 118, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(0, 230, 118, 0.1)'
                    }
                }
            }
        }
    });
}

function createDynamicBarChart() {
    const ctx = document.getElementById('dynamicBarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['行为分析', '注册表操作', '文件操作', '网络连接'],
            datasets: [{
                label: '特征数量',
                data: [150, 95, 180, 110],
                backgroundColor: [
                    'rgba(255, 152, 0, 0.7)',
                    'rgba(255, 112, 67, 0.7)',
                    'rgba(255, 87, 34, 0.7)',
                    'rgba(255, 112, 67, 0.7)'
                ],
                borderColor: [
                    'rgba(255, 152, 0, 1)',
                    'rgba(255, 112, 67, 1)',
                    'rgba(255, 87, 34, 1)',
                    'rgba(255, 112, 67, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: '动态特征统计',
                    color: '#ff9800',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(255, 152, 0, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(255, 152, 0, 0.1)'
                    }
                }
            }
        }
    });
}

function createNetworkBarChart() {
    const ctx = document.getElementById('networkBarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['域名解析', '流量分析', '协议分析', '时间序列'],
            datasets: [{
                label: '特征数量',
                data: [75, 130, 90, 60],
                backgroundColor: [
                    'rgba(156, 39, 176, 0.7)',
                    'rgba(142, 36, 170, 0.7)',
                    'rgba(123, 31, 162, 0.7)',
                    'rgba(106, 27, 154, 0.7)'
                ],
                borderColor: [
                    'rgba(156, 39, 176, 1)',
                    'rgba(142, 36, 170, 1)',
                    'rgba(123, 31, 162, 1)',
                    'rgba(106, 27, 154, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: '网络特征统计',
                    color: '#9c27b0',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(156, 39, 176, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#aaa'
                    },
                    grid: {
                        color: 'rgba(156, 39, 176, 0.1)'
                    }
                }
            }
        }
    });
}

function createOverallPieChart() {
    const ctx = document.getElementById('overallPieChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['静态特征', '动态特征', '网络特征'],
            datasets: [{
                data: [450, 535, 355],
                backgroundColor: [
                    'rgba(0, 230, 118, 0.8)',
                    'rgba(255, 152, 0, 0.8)',
                    'rgba(156, 39, 176, 0.8)'
                ],
                borderColor: [
                    'rgba(0, 230, 118, 1)',
                    'rgba(255, 152, 0, 1)',
                    'rgba(156, 39, 176, 1)'
                ],
                borderWidth: 2,
                offset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#fff',
                        font: {
                            size: 14
                        }
                    }
                },
                title: {
                    display: true,
                    text: '总体特征分布',
                    color: '#00d4ff',
                    font: {
                        size: 16
                    }
                }
            }
        }
    });
}

function initClustering() {
    const runBtn = document.getElementById('runClustering');
    const scatterPlotCanvas = document.getElementById('scatterPlot');
    const datasetSelect = document.getElementById('clusteringDatasetSelect');
    const algorithmSelect = document.getElementById('clusteringAlgorithm');
    const clusterCountInput = document.getElementById('clusterCount');
    const clusterSummaryCards = document.getElementById('clusterCards');
    const clusterDetailContent = document.getElementById('clusterDetailContent');
    const noSelectionMessage = document.getElementById('noSelectionMessage');
    const selectedClusterDetailsDiv = document.getElementById('selectedClusterDetails');
    const resetZoomBtn = document.getElementById('resetZoom');
    const clearSelectionBtn = document.getElementById('clearSelection');
    const hoverTooltip = document.getElementById('hoverTooltip');
    const selectionBox = document.getElementById('selectionBox');

    if (!runBtn || !scatterPlotCanvas || !datasetSelect || !algorithmSelect || !clusterCountInput || !clusterSummaryCards || !clusterDetailContent || !noSelectionMessage || !selectedClusterDetailsDiv || !resetZoomBtn || !clearSelectionBtn || !hoverTooltip || !selectionBox) return;

    let canvas = scatterPlotCanvas;
    let ctx = canvas.getContext('2d');
    window.clusteringPoints = [];
    window.clusteringClusters = [];
    let selectedPoints = [];
    let selectedCluster = null;
    let hoveredPoint = null;
    let zoom = 1;
    let pan = { x: 0, y: 0 };
    let isPanning = false;
    let panStart = { x: 0, y: 0 };
    let isSelecting = false;
    let selectionStart = null;
    let selectionEnd = null;

    function resizeCanvas() {
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = 500;
        drawScatterPlot();
    }

    function getClusterColor(clusterId, confidence) {
        const hue = (clusterId * 60) % 360;
        const saturation = 70;
        const lightness = confidence > 0.7 ? 50 : 65;
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    function generateMockData() {
        window.clusteringPoints = [];
        window.clusteringClusters = [];
        points = [];
        clusters = [];
        const numClusters = parseInt(clusterCountInput.value) || 5;
        const numPoints = 150;
        const featureDimension = 10;

        for (let i = 0; i < numClusters; i++) {
            const centerX = Math.random() * canvas.width * 0.6 + canvas.width * 0.2;
            const centerY = Math.random() * canvas.height * 0.6 + canvas.height * 0.2;
            const spread = 50 + Math.random() * 50;
            const confidence = 0.5 + Math.random() * 0.5;
            const c2Domains = ['evil.com', 'malicious.net', 'c2-server.org', 'command.cn'][Math.floor(Math.random() * 4)];
            const firstSeen = new Date(2024 - Math.floor(Math.random() * 3), Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toLocaleDateString();
            const md5 = Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
            const reports = ['APT-2024-' + (1000 + Math.floor(Math.random() * 1000)), 'THREAT-' + (2024000 + Math.floor(Math.random() * 1000))];

            const clusterPoints = Math.floor(numPoints / numClusters) + (i === 0 ? numPoints % numClusters : 0);
            let clusterXSum = 0;
            let clusterYSum = 0;
            let validPoints = 0;
            let clusterFeatureSum = new Array(featureDimension).fill(0);
            
            for (let j = 0; j < clusterPoints; j++) {
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * spread;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
                const isOutlier = Math.random() < 0.05;

                const featureVector = [];
                for (let k = 0; k < featureDimension; k++) {
                    featureVector.push(Math.random());
                }

                if (!isOutlier) {
                    clusterXSum += x;
                    clusterYSum += y;
                    validPoints++;
                    for (let k = 0; k < featureDimension; k++) {
                        clusterFeatureSum[k] += featureVector[k];
                    }
                }

                window.clusteringPoints.push({
                    id: window.clusteringPoints.length,
                    x: x,
                    y: y,
                    clusterId: isOutlier ? -1 : i,
                    confidence: isOutlier ? 0.3 : confidence,
                    name: ['sample.exe', 'malware.dll', 'trojan.doc', 'backdoor.pdf'][Math.floor(Math.random() * 4)] + '_' + window.clusteringPoints.length,
                    c2Domains: isOutlier ? [] : [c2Domains],
                    firstSeen: firstSeen,
                    md5: md5,
                    reports: reports,
                    featureVector: featureVector,
                    isOutlier: isOutlier
                });
            }

            const centroidX = validPoints > 0 ? (clusterXSum / validPoints).toFixed(2) : 0;
            const centroidY = validPoints > 0 ? (clusterYSum / validPoints).toFixed(2) : 0;
            const featureVector = validPoints > 0 ? clusterFeatureSum.map(val => val / validPoints) : new Array(featureDimension).fill(0);

            window.clusteringClusters.push({
                id: i,
                name: `簇 ${i + 1}`,
                count: clusterPoints,
                confidence: confidence,
                firstSeen: firstSeen,
                c2Domains: c2Domains,
                md5: md5,
                reports: reports.join(', '),
                centroid: { x: centroidX, y: centroidY },
                featureVector: featureVector
            });
        }

        window.clusteringResults = window.clusteringClusters.map(c => ({
            id: c.id + 1,
            name: c.name,
            count: c.count,
            description: `包含 ${c.count} 个样本的聚类`,
            date: '今天',
            isCustom: false,
            centroid: c.centroid,
            featureVector: c.featureVector,
            version: 'v1.0.0'
        }));
    }

    function drawScatterPlot() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.save();
        ctx.translate(pan.x, pan.y);
        ctx.scale(zoom, zoom);

        window.clusteringPoints.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, point.isOutlier ? 8 : 6, 0, Math.PI * 2);
            
            if (point.isOutlier) {
                ctx.fillStyle = 'rgba(244, 67, 54, 0.8)';
            } else {
                ctx.fillStyle = getClusterColor(point.clusterId, point.confidence);
            }
            
            if (selectedPoints.includes(point)) {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
            } else {
                ctx.strokeStyle = 'transparent';
            }
            
            ctx.fill();
            ctx.stroke();
        });

        ctx.restore();
    }

    function updateClusterSummary() {
        console.log('updateClusterSummary called');
        console.log('clusterSummaryCards element:', clusterSummaryCards);
        console.log('window.clusteringClusters:', window.clusteringClusters);
        
        if (!clusterSummaryCards) {
            console.error('clusterSummaryCards element not found!');
            return;
        }
        
        if (!window.clusteringClusters || window.clusteringClusters.length === 0) {
            console.error('window.clusteringClusters is empty!');
            return;
        }
        
        clusterSummaryCards.innerHTML = window.clusteringClusters.map(cluster => `
            <div class="cluster-summary-card" data-cluster-id="${cluster.id}" style="background: rgba(0, 23, 46, 0.9); padding: 0.75rem; border-radius: 6px; border-left: 4px solid ${getClusterColor(cluster.id, cluster.confidence)}; cursor: pointer; transition: all 0.2s ease; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #ffffff; font-weight: 600; font-size: 0.95rem;">${cluster.name}</span>
                <span style="font-size: 0.8rem; padding: 0.2rem 0.4rem; border-radius: 4px; background: ${cluster.confidence > 0.7 ? 'rgba(76, 175, 80, 0.2); color: #4caf50;' : 'rgba(255, 193, 7, 0.2); color: #ffc107;'}">置信度${cluster.confidence > 0.7 ? '高' : '低'}</span>
            </div>
        `).join('');

        document.querySelectorAll('.cluster-summary-card').forEach(card => {
            card.addEventListener('click', function() {
                const clusterId = parseInt(this.getAttribute('data-cluster-id'));
                selectCluster(clusterId);
            });
        });

        updateAttributionClusterSelect();
    }

    function updateAttributionClusterSelect() {
        console.log('updateAttributionClusterSelect called');
        console.log('Current clusters:', window.clusteringClusters);
        
        const attributionClusterSelect = document.getElementById('attributionClusterSelect');
        console.log('attributionClusterSelect element:', attributionClusterSelect);
        
        if (attributionClusterSelect) {
            attributionClusterSelect.innerHTML = `
                <option value="">选择模型训练结果...</option>
                ${window.clusteringClusters.map(cluster => `
                    <option value="${cluster.id}">${cluster.name} (${cluster.count}个样本)</option>
                `).join('')}
            `;
            console.log('Attribution cluster select updated');
        } else {
            console.error('attributionClusterSelect element not found!');
        }
    }

    function selectCluster(clusterId) {
        selectedCluster = window.clusteringClusters.find(c => c.id === clusterId);
        selectedPoints = window.clusteringPoints.filter(p => p.clusterId === clusterId);

        noSelectionMessage.style.display = 'none';
        selectedClusterDetailsDiv.style.display = 'block';

        document.getElementById('detailClusterId').textContent = selectedCluster.name;
        document.getElementById('detailSampleCount').textContent = selectedCluster.count;
        document.getElementById('detailConfidence').textContent = (selectedCluster.confidence * 100).toFixed(0) + '%';
        document.getElementById('detailFirstSeen').textContent = selectedCluster.firstSeen;
        document.getElementById('detailC2Domains').textContent = selectedCluster.c2Domains;
        document.getElementById('detailMD5').textContent = selectedCluster.md5;
        document.getElementById('detailReports').textContent = selectedCluster.reports;

        drawScatterPlot();
    }

    function calculateCosineSimilarity(vector1, vector2) {
        if (!vector1 || !vector2 || vector1.length !== vector2.length) {
            return 0;
        }
        
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 === 0 || norm2 === 0) {
            return 0;
        }
        
        return dotProduct / (norm1 * norm2);
    }

    function showTooltip(point, x, y) {
        hoverTooltip.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">${point.name}</div>
            <div style="font-size: 0.85rem; color: #aaa; margin-bottom: 0.25rem;">簇: ${point.isOutlier ? '离群点' : '簇 ' + (point.clusterId + 1)}</div>
            <div style="font-size: 0.85rem; color: #aaa; margin-bottom: 0.25rem;">置信度: ${(point.confidence * 100).toFixed(0)}%</div>
            <div style="font-size: 0.85rem; color: #aaa; margin-bottom: 0.25rem;">首次出现: ${point.firstSeen}</div>
            ${!point.isOutlier ? `<div style="font-size: 0.85rem; color: #aaa; margin-bottom: 0.25rem;">C2域名: ${point.c2Domains.join(', ')}</div>` : ''}
            <div style="font-size: 0.85rem; color: #aaa;">MD5: ${point.md5.substring(0, 16)}...</div>
        `;
        hoverTooltip.style.display = 'block';
        hoverTooltip.style.left = (x + 15) + 'px';
        hoverTooltip.style.top = (y + 15) + 'px';
    }

    function hideTooltip() {
        hoverTooltip.style.display = 'none';
    }

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left - pan.x) / zoom,
            y: (e.clientY - rect.top - pan.y) / zoom
        };
    }

    canvas.addEventListener('mousedown', function(e) {
        if (e.button === 2) {
            isPanning = true;
            panStart = { x: e.clientX - pan.x, y: e.clientY - pan.y };
        } else {
            const coords = getCanvasCoordinates(e);
            isSelecting = true;
            selectionStart = coords;
            selectionEnd = coords;
        }
    });

    canvas.addEventListener('mousemove', function(e) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (isPanning) {
            pan.x = e.clientX - panStart.x;
            pan.y = e.clientY - panStart.y;
            drawScatterPlot();
            return;
        }

        if (isSelecting) {
            const coords = getCanvasCoordinates(e);
            selectionEnd = coords;
            
            selectionBox.style.display = 'block';
            selectionBox.style.left = Math.min(selectionStart.x * zoom + pan.x, selectionEnd.x * zoom + pan.x) + 'px';
            selectionBox.style.top = Math.min(selectionStart.y * zoom + pan.y, selectionEnd.y * zoom + pan.y) + 'px';
            selectionBox.style.width = Math.abs((selectionEnd.x - selectionStart.x) * zoom) + 'px';
            selectionBox.style.height = Math.abs((selectionEnd.y - selectionStart.y) * zoom) + 'px';
        } else {
            const coords = getCanvasCoordinates(e);
            let found = false;
            
            for (let i = window.clusteringPoints.length - 1; i >= 0; i--) {
                const point = window.clusteringPoints[i];
                const dx = coords.x - point.x;
                const dy = coords.y - point.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance <= 8) {
                    hoveredPoint = point;
                    showTooltip(point, x, y);
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                hoveredPoint = null;
                hideTooltip();
            }
        }
    });

    canvas.addEventListener('mouseup', function(e) {
        if (isPanning) {
            isPanning = false;
            return;
        }

        if (isSelecting) {
            isSelecting = false;
            selectionBox.style.display = 'none';

            const minX = Math.min(selectionStart.x, selectionEnd.x);
            const maxX = Math.max(selectionStart.x, selectionEnd.x);
            const minY = Math.min(selectionStart.y, selectionEnd.y);
            const maxY = Math.max(selectionStart.y, selectionEnd.y);

            selectedPoints = window.clusteringPoints.filter(p => 
                p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY
            );

            if (selectedPoints.length > 0) {
                const clusterIds = [...new Set(selectedPoints.filter(p => !p.isOutlier).map(p => p.clusterId))];
                
                if (clusterIds.length === 1) {
                    selectCluster(clusterIds[0]);
                } else {
                    noSelectionMessage.style.display = 'none';
                    selectedClusterDetailsDiv.style.display = 'block';
                    
                    document.getElementById('detailClusterId').textContent = '多选样本';
                    document.getElementById('detailSampleCount').textContent = selectedPoints.length;
                    document.getElementById('detailConfidence').textContent = '-';
                    document.getElementById('detailFirstSeen').textContent = '-';
                    document.getElementById('detailC2Domains').textContent = '-';
                    document.getElementById('detailMD5').textContent = '-';
                    document.getElementById('detailReports').textContent = '-';
                }
                
                drawScatterPlot();
            }
        }
    });

    canvas.addEventListener('wheel', function(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        zoom *= delta;
        zoom = Math.max(0.5, Math.min(3, zoom));
        drawScatterPlot();
    });

    canvas.addEventListener('contextmenu', function(e) {
        e.preventDefault();
    });

    resetZoomBtn.addEventListener('click', function() {
        zoom = 1;
        pan = { x: 0, y: 0 };
        drawScatterPlot();
    });

    clearSelectionBtn.addEventListener('click', function() {
        selectedPoints = [];
        selectedCluster = null;
        noSelectionMessage.style.display = 'block';
        selectedClusterDetailsDiv.style.display = 'none';
        drawScatterPlot();
    });

    runBtn.addEventListener('click', function() {
        const selectedDataset = datasetSelect.value;
        const algorithm = algorithmSelect.value;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 23, 46, 0.9)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#00d4ff';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('正在运行模型训练...', canvas.width / 2, canvas.height / 2);
        
        setTimeout(() => {
            resizeCanvas();
            generateMockData();
            drawScatterPlot();
            updateClusterSummary();
            
            if (typeof updateGangList === 'function') {
                updateGangList();
            }
        }, 2000);
    });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
}

function renderAttributionDetails(cluster) {
    console.log('renderAttributionDetails called with cluster:', cluster);
    console.log('Cluster featureVector:', cluster.featureVector);
    console.log('Known gangs:', window.knownGangs);
    
    const featureDimensions = ['PDB路径特征', 'DLL:API组合特征', 'C2域名特征', 'OpCode序列特征', '互斥量特征', '注册表特征', '网络行为特征', '文件操作特征', '进程注入特征', '反调试特征'];
    
    let similarityResults = [];
    if (window.knownGangs && window.knownGangs.length > 0 && cluster.featureVector) {
        similarityResults = window.knownGangs.map(gang => {
            if (!gang.featureVector) {
                return { name: gang.name, similarity: Math.random() * 0.4 + 0.1 };
            }
            const similarity = calculateCosineSimilarity(cluster.featureVector, gang.featureVector);
            return { name: gang.name, similarity: similarity };
        }).sort((a, b) => b.similarity - a.similarity);
    } else {
        const aptGroups = ['APT29', 'Lazarus', 'Sofacy', 'APT28', 'Fancy Bear', 'Cozy Bear'];
        similarityResults = aptGroups.map(name => ({
            name: name,
            similarity: Math.random() * 0.6 + 0.2
        })).sort((a, b) => b.similarity - a.similarity);
    }

    const topMatches = similarityResults.slice(0, 3);
    const topMatch = topMatches[0];
    const topGang = window.knownGangs ? window.knownGangs.find(g => g.name === topMatch.name) : null;

    console.log('Top matches:', topMatches);
    console.log('Top gang:', topGang);

    const radarChartId = 'threatRadarChart_' + Date.now();
    
    const attributionContent = document.getElementById('attributionContent');
    console.log('Attribution content element:', attributionContent);
    
    if (attributionContent) {
        console.log('Updating attribution content HTML');
        attributionContent.innerHTML = `
            <div style="background: rgba(0, 23, 46, 0.9); padding: 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2); margin-bottom: 1rem;">
                <h3 style="color: #00d4ff; margin-bottom: 1rem; font-size: 1.1rem;">${cluster.name} · ${cluster.count}个样本</h3>
                
                <div style="background: rgba(0, 23, 46, 0.9); padding: 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2); margin-bottom: 1rem;">
                    <div style="color: #00d4ff; font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem;">特征画像</div>
                    <canvas id="${radarChartId}" style="max-height: 250px;"></canvas>
                </div>
                
                <div style="background: rgba(0, 23, 46, 0.9); padding: 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2); margin-bottom: 1rem;">
                    <div style="color: #00d4ff; font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem;">归因分析</div>
                        <select id="gangAttributionSelect" style="width: 100%; padding: 0.5rem; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 6px; font-size: 0.9rem; background: rgba(0, 23, 46, 0.9); color: #ffffff; margin-bottom: 0.75rem;">
                            <option value="">选择组织...</option>
                            ${topMatches.map(match => `
                                <option value="${match.name}" ${match === topMatch ? 'selected' : ''}>
                                    ${match.name} (匹配度 ${(match.similarity * 100).toFixed(0)}%) ${match === topMatch ? '← 系统推荐' : ''}
                                </option>
                            `).join('')}
                            <option value="custom">新组织... (自定义命名)</option>
                        </select>
                        <button id="confirmAttribution" style="width: 100%; padding: 0.6rem; border: none; border-radius: 6px; font-size: 0.9rem; font-weight: 600; cursor: pointer; background: linear-gradient(135deg, #0099cc 0%, #006699 100%); color: #ffffff; transition: all 0.3s ease;">确认归因并入库</button>
                    </div>

                    ${topGang ? `
                    <div style="background: rgba(0, 23, 46, 0.9); padding: 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2);">
                        <div style="color: #00d4ff; font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem;">${cluster.name} 特征向量详情</div>
                        <div style="border-bottom: 1px solid rgba(0, 212, 255, 0.3); margin-bottom: 0.75rem; padding-bottom: 0.5rem; font-size: 0.8rem; color: #aaa;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                            <thead>
                                <tr style="background: rgba(0, 212, 255, 0.1);">
                                    <th style="padding: 0.5rem; text-align: left; color: #00d4ff; font-weight: 600;">特征维度</th>
                                    <th style="padding: 0.5rem; text-align: center; color: #00d4ff; font-weight: 600;">本簇值</th>
                                    <th style="padding: 0.5rem; text-align: center; color: #00d4ff; font-weight: 600;">${topGang.name}质心</th>
                                    <th style="padding: 0.5rem; text-align: center; color: #00d4ff; font-weight: 600;">差异</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${featureDimensions.map((dim, index) => {
                                    const clusterValue = cluster.featureVector[index] || 0;
                                    const gangValue = topGang.featureVector[index] || 0;
                                    const diff = clusterValue - gangValue;
                                    const diffColor = diff > 0 ? '#4caf50' : (diff < 0 ? '#f44336' : '#aaa');
                                    const diffSign = diff > 0 ? '+' : '';
                                    return `
                                        <tr style="border-bottom: 1px solid rgba(0, 212, 255, 0.1);">
                                            <td style="padding: 0.5rem; color: #ffffff;">${dim}</td>
                                            <td style="padding: 0.5rem; text-align: center; color: #00d4ff; font-weight: 600;">${clusterValue.toFixed(2)}</td>
                                            <td style="padding: 0.5rem; text-align: center; color: #aaa;">${gangValue.toFixed(2)}</td>
                                            <td style="padding: 0.5rem; text-align: center; color: ${diffColor}; font-weight: 600;">${diffSign}${diff.toFixed(2)}</td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                    ` : ''}
            </div>
        `;

        console.log('HTML updated, now creating chart...');
        setTimeout(() => {
            const ctx = document.getElementById(radarChartId);
            console.log('Canvas element:', ctx);
            if (ctx && cluster.featureVector) {
                console.log('Creating radar chart with data:', cluster.featureVector);
                new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: featureDimensions,
                        datasets: [
                            {
                                label: cluster.name,
                                data: cluster.featureVector,
                                backgroundColor: 'rgba(0, 212, 255, 0.2)',
                                borderColor: 'rgba(0, 212, 255, 1)',
                                borderWidth: 2,
                                pointBackgroundColor: 'rgba(0, 212, 255, 1)',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: 'rgba(0, 212, 255, 1)'
                            },
                            topGang ? {
                                label: topGang.name,
                                data: topGang.featureVector,
                                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                                borderColor: 'rgba(255, 193, 7, 1)',
                                borderWidth: 2,
                                pointBackgroundColor: 'rgba(255, 193, 7, 1)',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: 'rgba(255, 193, 7, 1)'
                            } : null
                        ].filter(d => d !== null)
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {
                            r: {
                                beginAtZero: true,
                                max: 1,
                                ticks: {
                                    stepSize: 0.2,
                                    color: '#ffffff',
                                    font: { size: 11 },
                                    backdropColor: 'transparent'
                                },
                                pointLabels: {
                                    color: '#ffffff',
                                    font: { size: 10 }
                                },
                                grid: {
                                    color: 'rgba(0, 212, 255, 0.2)'
                                },
                                angleLines: {
                                    color: 'rgba(0, 212, 255, 0.2)'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'bottom',
                                labels: {
                                    color: '#ffffff',
                                    font: { size: 11 },
                                    boxWidth: 14,
                                    padding: 10
                                }
                            }
                        }
                    }
                });
                console.log('Chart created successfully');
            } else {
                console.error('Canvas element not found or cluster has no featureVector');
            }
        }, 100);

        const confirmBtn = document.getElementById('confirmAttribution');
        if (confirmBtn) {
            confirmBtn.addEventListener('click', function() {
                const select = document.getElementById('gangAttributionSelect');
                const selectedValue = select.value;
                
                if (selectedValue === 'custom') {
                    const customName = prompt('请输入新组织名称:');
                    if (customName && customName.trim()) {
                        const organizationName = customName.trim();
                        
                        if (!window.knownGangs) {
                            window.knownGangs = [];
                        }
                        
                        const newGang = {
                            id: Date.now(),
                            name: organizationName,
                            count: cluster.count,
                            featureVector: [...cluster.featureVector],
                            version: 'v1.0.0',
                            date: '今天',
                            isKnown: false
                        };
                        
                        window.knownGangs.push(newGang);
                        
                        alert(`已归因为新组织: ${organizationName}，并已入库到团伙库`);
                        
                        renderAttributionDetails(cluster);
                        updateGangList();
                    } else {
                        alert('请输入有效的组织名称');
                    }
                } else if (selectedValue) {
                    const selectedMatch = topMatches.find(m => m.name === selectedValue);
                    if (selectedMatch) {
                        const existingGang = window.knownGangs ? window.knownGangs.find(g => g.name === selectedValue) : null;
                        
                        if (existingGang) {
                            const oldCount = existingGang.count;
                            const newCount = oldCount + cluster.count;
                            
                            const newFeatureVector = existingGang.featureVector.map((oldValue, index) => {
                                const newValue = cluster.featureVector[index] || 0;
                                return (oldValue * oldCount + newValue * cluster.count) / newCount;
                            });
                            
                            existingGang.count = newCount;
                            existingGang.featureVector = newFeatureVector;
                            existingGang.version = 'v' + (parseFloat(existingGang.version.replace('v', '')) + 0.1).toFixed(1);
                            existingGang.date = '今天';
                        } else {
                            if (!window.knownGangs) {
                                window.knownGangs = [];
                            }
                            
                            const newGang = {
                                id: Date.now(),
                                name: selectedValue,
                                count: cluster.count,
                                featureVector: [...cluster.featureVector],
                                version: 'v1.0.0',
                                date: '今天',
                                isKnown: true
                            };
                            
                            window.knownGangs.push(newGang);
                        }
                        
                        alert(`已确认为 ${selectedValue} (匹配度 ${(selectedMatch.similarity * 100).toFixed(0)}%)，并已入库到团伙库`);
                        
                        renderAttributionDetails(cluster);
                        updateGangList();
                    }
                } else {
                    alert('请先选择一个组织');
                }
            });
        }
    } else {
        console.error('Attribution content element not found');
    }
}

function calculateCosineSimilarity(vector1, vector2) {
    if (!vector1 || !vector2 || vector1.length !== vector2.length) {
        return 0;
    }
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vector1.length; i++) {
        dotProduct += vector1[i] * vector2[i];
        norm1 += vector1[i] * vector1[i];
        norm2 += vector2[i] * vector2[i];
    }
    
    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);
    
    if (norm1 === 0 || norm2 === 0) {
        return 0;
    }
    
    return dotProduct / (norm1 * norm2);
}

function initAttribution() {
    console.log('initAttribution called');
    const attributionClusterSelect = document.getElementById('attributionClusterSelect');
    const analyzeAttributionBtn = document.getElementById('analyzeAttribution');

    console.log('attributionClusterSelect:', attributionClusterSelect);
    console.log('analyzeAttributionBtn:', analyzeAttributionBtn);

    if (!attributionClusterSelect || !analyzeAttributionBtn) {
        console.error('Attribution elements not found!');
        return;
    }

    analyzeAttributionBtn.addEventListener('click', function() {
        console.log('Analyze attribution button clicked');
        const clusterId = parseInt(attributionClusterSelect.value);
        console.log('Selected clusterId:', clusterId);
        
        if (isNaN(clusterId)) {
            alert('请先选择模型训练结果');
            return;
        }

        console.log('Current clusters:', window.clusteringClusters);
        const cluster = window.clusteringClusters.find(c => c.id === clusterId);
        console.log('Found cluster:', cluster);
        
        if (!cluster) {
            alert('未找到选中的聚类结果');
            return;
        }

        console.log('Calling renderAttributionDetails...');
        renderAttributionDetails(cluster);
    });
}

function initGangStorage() {
    const gangFilter = document.getElementById('gangFilter');
    const gangList = document.getElementById('gangList');
    
    if (!gangFilter || !gangList) return;

    // 初始化团伙列表
    if (!window.knownGangs) {
        window.knownGangs = [
            {
                id: 100,
                name: 'APT29',
                description: '俄罗斯APT组织，主要针对政府机构',
                date: '今天',
                isCustom: false,
                isKnown: true,
                count: 1262,
                centroid: { x: 0, y: 0 },
                featureVector: [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.5, 0.8, 0.7],
                version: 'v2.3.1'
            },
            {
                id: 101,
                name: 'Lazarus',
                description: '朝鲜APT组织，主要进行金融攻击',
                date: '昨天',
                isCustom: false,
                isKnown: true,
                count: 904,
                centroid: { x: 0, y: 0 },
                featureVector: [0.6, 0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.6, 0.8],
                version: 'v2.3.1'
            },
            {
                id: 102,
                name: 'Sofacy',
                description: '俄罗斯APT组织，主要针对军事和外交',
                date: '3天前',
                isCustom: false,
                isKnown: true,
                count: 758,
                centroid: { x: 0, y: 0 },
                featureVector: [0.7, 0.6, 0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.6],
                version: 'v2.3.1'
            },
            {
                id: 103,
                name: 'APT28',
                description: '俄罗斯APT组织，主要进行网络间谍活动',
                date: '1周前',
                isCustom: false,
                isKnown: true,
                count: 632,
                centroid: { x: 0, y: 0 },
                featureVector: [0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.6, 0.8, 0.7, 0.9],
                version: 'v2.3.1'
            },
            {
                id: 104,
                name: 'APT1',
                description: '中国APT组织，主要针对商业和政府',
                date: '2周前',
                isCustom: false,
                isKnown: true,
                count: 521,
                centroid: { x: 0, y: 0 },
                featureVector: [0.75, 0.65, 0.85, 0.7, 0.75, 0.8, 0.65, 0.75, 0.7, 0.8],
                version: 'v2.3.1'
            },
            {
                id: 105,
                name: 'APT41',
                description: '中国APT组织，同时进行网络间谍和金融攻击',
                date: '2周前',
                isCustom: false,
                isKnown: true,
                count: 489,
                centroid: { x: 0, y: 0 },
                featureVector: [0.7, 0.75, 0.8, 0.75, 0.7, 0.75, 0.8, 0.7, 0.75, 0.8],
                version: 'v2.3.1'
            },
            {
                id: 106,
                name: 'Sandworm',
                description: '俄罗斯APT组织，主要针对关键基础设施',
                date: '3周前',
                isCustom: false,
                isKnown: true,
                count: 432,
                centroid: { x: 0, y: 0 },
                featureVector: [0.85, 0.7, 0.75, 0.8, 0.85, 0.7, 0.75, 0.85, 0.8, 0.75],
                version: 'v2.3.1'
            },
            {
                id: 107,
                name: 'Fancy Bear',
                description: '俄罗斯APT组织，主要进行政治攻击',
                date: '1个月前',
                isCustom: false,
                isKnown: true,
                count: 378,
                centroid: { x: 0, y: 0 },
                featureVector: [0.8, 0.75, 0.7, 0.85, 0.8, 0.75, 0.7, 0.8, 0.85, 0.7],
                version: 'v2.3.1'
            },
            {
                id: 108,
                name: 'Cozy Bear',
                description: '俄罗斯APT组织，主要进行情报收集',
                date: '1个月前',
                isCustom: false,
                isKnown: true,
                count: 345,
                centroid: { x: 0, y: 0 },
                featureVector: [0.75, 0.8, 0.75, 0.7, 0.75, 0.8, 0.75, 0.7, 0.8, 0.75],
                version: 'v2.3.1'
            },
            {
                id: 109,
                name: 'OilRig',
                description: '伊朗APT组织，主要针对中东地区',
                date: '1个月前',
                isCustom: false,
                isKnown: true,
                count: 298,
                centroid: { x: 0, y: 0 },
                featureVector: [0.7, 0.7, 0.8, 0.75, 0.7, 0.7, 0.8, 0.75, 0.7, 0.8],
                version: 'v2.3.1'
            },
            {
                id: 110,
                name: 'Kimsuky',
                description: '朝鲜APT组织，主要进行情报收集',
                date: '2个月前',
                isCustom: false,
                isKnown: true,
                count: 256,
                centroid: { x: 0, y: 0 },
                featureVector: [0.65, 0.75, 0.7, 0.8, 0.65, 0.75, 0.7, 0.8, 0.65, 0.75],
                version: 'v2.3.1'
            },
            {
                id: 111,
                name: 'OceanLotus',
                description: '越南APT组织，主要针对东南亚地区',
                date: '2个月前',
                isCustom: false,
                isKnown: true,
                count: 234,
                centroid: { x: 0, y: 0 },
                featureVector: [0.6, 0.7, 0.75, 0.7, 0.6, 0.7, 0.75, 0.7, 0.6, 0.75],
                version: 'v2.3.1'
            },
            {
                id: 112,
                name: 'MuddyWater',
                description: '伊朗APT组织，主要针对中东和南亚',
                date: '3个月前',
                isCustom: false,
                isKnown: true,
                count: 198,
                centroid: { x: 0, y: 0 },
                featureVector: [0.68, 0.72, 0.78, 0.74, 0.68, 0.72, 0.78, 0.74, 0.68, 0.78],
                version: 'v2.3.1'
            }
        ];
    }

    updateGangList();

    gangFilter.addEventListener('change', function() {
        const filterValue = this.value;
        
        gangList.innerHTML = '<div style="text-align: center;"><i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #667eea;"></i><p style="margin-top: 1rem;">正在过滤团伙...</p></div>';
        
        updateGangList(filterValue);
    });
}

// 更新团伙列表
function updateGangList(filterValue = 'all') {
    const gangList = document.getElementById('gangList');
    if (!gangList) return;

    // 只显示已知团伙（从APT归因结果中入库的）
    let allGangs = [];
    if (window.knownGangs) {
        allGangs = allGangs.concat(window.knownGangs);
    }

    if (allGangs.length === 0) {
        gangList.innerHTML = '<div style="text-align: center; padding: 3rem; color: #aaa;"><i class="fas fa-users" style="font-size: 3rem; margin-bottom: 1rem;"></i><p>暂无团伙数据，请先在APT归因结果中确认归因</p></div>';
        return;
    }

    // 过滤团伙
    let filteredGangs = allGangs;
    if (filterValue !== 'all') {
        filteredGangs = allGangs.filter(gang => {
            if (filterValue === 'known' && gang.isKnown) return true;
            if (filterValue === 'custom' && !gang.isKnown) return true;
            return false;
        });
    }

    // 显示过滤后的团伙
    if (filteredGangs.length === 0) {
        gangList.innerHTML = '<div style="text-align: center; padding: 3rem; color: #aaa;"><i class="fas fa-users" style="font-size: 3rem; margin-bottom: 1rem;"></i><p>没有符合条件的团伙</p></div>';
        return;
    }

    gangList.innerHTML = `
        <table style="width: 100%; border-collapse: collapse; background: rgba(0, 23, 46, 0.9); border-radius: 12px; overflow: hidden; border: 1px solid rgba(0, 212, 255, 0.2);">
            <thead>
                <tr style="background: rgba(0, 212, 255, 0.1); border-bottom: 2px solid rgba(0, 212, 255, 0.3);">
                    <th style="padding: 1rem; text-align: left; color: #00d4ff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.2);">组织名</th>
                    <th style="padding: 1rem; text-align: center; color: #00d4ff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.2);">样本数</th>
                    <th style="padding: 1rem; text-align: center; color: #00d4ff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.2);">质心版本</th>
                    <th style="padding: 1rem; text-align: center; color: #00d4ff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.2);">最后更新</th>
                    <th style="padding: 1rem; text-align: center; color: #00d4ff; font-weight: 600; font-size: 0.9rem;">操作</th>
                </tr>
            </thead>
            <tbody>
                ${filteredGangs.map(gang => `
                    <tr style="border-bottom: 1px solid rgba(0, 212, 255, 0.2); transition: background 0.3s ease;" onmouseover="this.style.background='rgba(0, 212, 255, 0.05)'" onmouseout="this.style.background='transparent'">
                        <td style="padding: 1rem; color: #ffffff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.1);">
                            ${gang.name}
                            ${gang.isKnown ? '<span style="background: rgba(0, 212, 255, 0.2); color: #00d4ff; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem; border: 1px solid rgba(0, 212, 255, 0.3);">已知</span>' : '<span style="background: rgba(76, 175, 80, 0.2); color: #4caf50; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem; border: 1px solid rgba(76, 175, 80, 0.3);">自定义</span>'}
                        </td>
                        <td style="padding: 1rem; text-align: center; color: #00d4ff; font-weight: 600; font-size: 0.9rem; border-right: 1px solid rgba(0, 212, 255, 0.1);">${gang.count.toLocaleString()}</td>
                        <td style="padding: 1rem; text-align: center; color: #ffffff; font-size: 0.85rem; border-right: 1px solid rgba(0, 212, 255, 0.1);">${gang.version || 'v1.0.0'}</td>
                        <td style="padding: 1rem; text-align: center; color: #ffffff; font-size: 0.85rem; border-right: 1px solid rgba(0, 212, 255, 0.1);">${gang.date}</td>
                        <td style="padding: 1rem; text-align: center;">
                            <button onclick="showGangDetails(${gang.id})" style="background: rgba(0, 212, 255, 0.2); color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.4); padding: 0.3rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(0, 212, 255, 0.3)'" onmouseout="this.style.background='rgba(0, 212, 255, 0.2)'">详情</button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// 显示团伙详情
function showGangDetails(gangId) {
    let allGangs = [];
    if (window.knownGangs) {
        allGangs = allGangs.concat(window.knownGangs);
    }
    if (window.clusteringResults && window.clusteringResults.length > 0) {
        allGangs = allGangs.concat(window.clusteringResults);
    }
    
    const gang = allGangs.find(g => g.id === gangId);
    if (!gang) return;

    const featureVectorPreview = gang.featureVector ? gang.featureVector.slice(0, 5).map(v => v.toFixed(2)).join(', ') + '...' : 'N/A';

    alert(`组织详情\n\n组织名称: ${gang.name}\n样本数量: ${gang.count.toLocaleString()}\n质心版本: ${gang.version || 'v1.0.0'}\n最后更新: ${gang.date}\n描述: ${gang.description}\n\n特征向量预览: [${featureVectorPreview}]`);
}

// 显示团伙归因
function showGangAttribution(gangId) {
    let allGangs = [];
    if (window.knownGangs) {
        allGangs = allGangs.concat(window.knownGangs);
    }
    if (window.clusteringResults && window.clusteringResults.length > 0) {
        allGangs = allGangs.concat(window.clusteringResults);
    }
    
    const gang = allGangs.find(g => g.id === gangId);
    if (!gang) return;

    if (!gang.featureVector) {
        alert('该组织没有特征向量数据，无法进行归因分析');
        return;
    }

    const knownGangs = allGangs.filter(g => !g.name.includes('未知') && g.id !== gangId);
    
    if (knownGangs.length === 0) {
        alert('没有已知的组织可供归因比较');
        return;
    }

    let similarityResults = knownGangs.map(g => {
        if (!g.featureVector) {
            return { name: g.name, similarity: Math.random() * 0.4 + 0.1 };
        }
        const similarity = calculateCosineSimilarity(gang.featureVector, g.featureVector);
        return { name: g.name, similarity: similarity };
    }).sort((a, b) => b.similarity - a.similarity);

    const topMatches = similarityResults.slice(0, 3);
    const topMatch = topMatches[0];

    const attributionText = topMatches.map((match, index) => {
        const prefix = match === topMatch ? '★ ' : '  ';
        return `${prefix}${match.name} (匹配度 ${(match.similarity * 100).toFixed(0)}%)`;
    }).join('\n');

    alert(`归因分析结果\n\n当前组织: ${gang.name}\n\n推荐归因:\n${attributionText}\n\n系统推荐: ${topMatch.name} (匹配度 ${(topMatch.similarity * 100).toFixed(0)}%)`);
}

// 编辑团伙名称
function editGangName(gangId) {
    const gang = window.clusteringResults.find(g => g.id === gangId);
    if (!gang) return;

    const newName = prompt('请输入新的团伙名称:', gang.name);
    if (newName && newName.trim() !== '') {
        gang.name = newName.trim();
        gang.isCustom = true;
        updateGangList(document.getElementById('gangFilter').value);
        alert('团伙名称已更新！');
    }
}

function initReport() {
    const generateBtn = document.getElementById('generateReport');
    const reportContent = document.getElementById('reportContent');
    const reportType = document.getElementById('reportType');
    const reportFormat = document.getElementById('reportFormat');

    if (!generateBtn || !reportContent || !reportType || !reportFormat) return;

    generateBtn.addEventListener('click', function() {
        const type = reportType.options[reportType.selectedIndex].text;
        const format = reportFormat.options[reportFormat.selectedIndex].text;
        
        reportContent.innerHTML = `
            <div style="text-align: center;">
                <i class="fas fa-spinner fa-spin" style="font-size: 3rem; color: #667eea;"></i>
                <p style="margin-top: 1rem;">正在生成${type} (${format})...</p>
            </div>
        `;

        setTimeout(() => {
            reportContent.innerHTML = `
                <div style="text-align: left;">
                    <h3 style="color: #333; margin-bottom: 1rem;">APT归因分析报告</h3>
                    <p style="margin-bottom: 0.5rem;"><strong>报告类型:</strong> ${type}</p>
                    <p style="margin-bottom: 0.5rem;"><strong>导出格式:</strong> ${format}</p>
                    <p style="margin-bottom: 0.5rem;"><strong>生成时间:</strong> ${new Date().toLocaleString()}</p>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #eee;">
                    <h4 style="color: #333; margin-bottom: 0.5rem;">摘要</h4>
                    <p style="color: #666; line-height: 1.6;">本报告基于系统对上传样本的分析结果，包含特征提取、模型训练和APT归因结果等关键信息。报告详细记录了样本的静态和动态特征、网络行为模式以及与已知APT组织的关联性分析。</p>
                    <h4 style="color: #333; margin: 1rem 0 0.5rem;">主要发现</h4>
                    <ul style="color: #666; line-height: 1.8; padding-left: 1.5rem;">
                        <li>检测到15个与APT29相关的样本</li>
                        <li>发现12个Lazarus组织样本</li>
                        <li>识别出8个Sofacy相关样本</li>
                        <li>发现15个待归因样本，需要进一步分析</li>
                    </ul>
                    <div style="margin-top: 2rem; text-align: center;">
                        <button class="btn btn-primary" onclick="alert('报告已导出为${format}格式')">
                            <i class="fas fa-download"></i> 下载报告
                        </button>
                    </div>
                </div>
            `;
        }, 2000);
    });
}

function initModelManagement() {
    const modelUploadArea = document.getElementById('modelUploadArea');
    const modelInput = document.getElementById('modelInput');
    const uploadModelBtn = document.getElementById('uploadModel');
    const modelsGrid = document.getElementById('modelsGrid');

    if (!modelUploadArea || !modelInput || !uploadModelBtn || !modelsGrid) return;

    modelUploadArea.addEventListener('click', function() {
        modelInput.click();
    });

    modelUploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = '#00d4ff';
        this.style.background = 'rgba(0, 212, 255, 0.1)';
    });

    modelUploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = '#00d4ff';
        this.style.background = 'rgba(0, 23, 46, 0.9)';
    });

    modelUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = '#00d4ff';
        this.style.background = 'rgba(0, 23, 46, 0.9)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            modelInput.files = files;
            updateModelUploadArea(files[0].name);
        }
    });

    modelInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            updateModelUploadArea(this.files[0].name);
        }
    });

    function updateModelUploadArea(filename) {
        modelUploadArea.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <p>已选择: ${filename}</p>
        `;
    }

    uploadModelBtn.addEventListener('click', function() {
        const modelName = document.getElementById('modelName').value.trim();
        const modelType = document.getElementById('modelType').value;
        const modelDescription = document.getElementById('modelDescription').value.trim();
        const modelFile = modelInput.files[0];

        if (!modelName || !modelFile) {
            alert('请填写模型名称并上传模型文件');
            return;
        }

        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        
        const typeLabels = {
            'classification': '分类模型',
            'clustering': '聚类模型',
            'detection': '检测模型'
        };

        modelCard.innerHTML = `
            <div class="model-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h4>${modelName}</h4>
            <p class="model-type">${typeLabels[modelType]}</p>
            <p class="model-desc">${modelDescription || '暂无描述'}</p>
            <div class="model-actions">
                <button class="btn btn-sm btn-select">选择</button>
                <button class="btn btn-sm btn-delete">删除</button>
            </div>
        `;

        modelsGrid.insertBefore(modelCard, modelsGrid.firstChild);

        document.getElementById('modelName').value = '';
        document.getElementById('modelDescription').value = '';
        modelInput.value = '';
        modelUploadArea.innerHTML = `
            <i class="fas fa-upload"></i>
            <p>拖拽模型文件到此处或点击上传</p>
        `;

        alert('模型上传成功！');
    });

    modelsGrid.addEventListener('click', function(e) {
        if (e.target.classList.contains('btn-delete')) {
            const card = e.target.closest('.model-card');
            if (confirm('确定要删除这个模型吗？')) {
                card.remove();
            }
        } else if (e.target.classList.contains('btn-select')) {
            const card = e.target.closest('.model-card');
            const modelName = card.querySelector('h4').textContent;
            alert('已选择模型: ' + modelName);
        }
    });
}