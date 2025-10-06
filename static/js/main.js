// Global variables
let currentDataPath = null;
let modelTrained = false;
let adversarialFilePath = null;

// Tab navigation
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');

    // Load model stats when opening visualize or about tabs
    if (tabName === 'visualize' || tabName === 'about') {
        loadModelStats();
    }
}

// Upload data
async function uploadData() {
    const fileInput = document.getElementById('dataFile');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('uploadStatus', 'Please select a file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showStatus('uploadStatus', 'Uploading and analyzing data...', 'info');

    try {
        const response = await fetch('/api/upload_data', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentDataPath = data.filepath;
            displayDataStats(data);
            showStatus('uploadStatus', 'Data uploaded successfully!', 'success');
        } else {
            showStatus('uploadStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `Error: ${error.message}`, 'error');
    }
}

// Display data statistics
function displayDataStats(stats) {
    const statsContent = document.getElementById('statsContent');
    const statsContainer = document.getElementById('dataStats');

    const html = `
        <div class="stat-item">
            <strong>Total Rows:</strong> ${stats.rows.toLocaleString()}
        </div>
        <div class="stat-item">
            <strong>Total Columns:</strong> ${stats.columns}
        </div>
        <div class="stat-item">
            <strong>Columns:</strong> ${stats.column_names.slice(0, 10).join(', ')}${stats.column_names.length > 10 ? '...' : ''}
        </div>
    `;

    statsContent.innerHTML = html;
    statsContainer.classList.remove('hidden');
}

// Update hyperparameters based on model type
function updateHyperparameters() {
    const modelType = document.getElementById('modelType').value;
    const hyperparamsDiv = document.getElementById('hyperparameters');

    let html = '';

    if (modelType === 'random_forest') {
        html = `
            <div class="form-group">
                <label>Number of Estimators:</label>
                <input type="number" id="param_n_estimators" value="200" min="50" max="500">
            </div>
            <div class="form-group">
                <label>Max Depth:</label>
                <input type="number" id="param_max_depth" value="20" min="5" max="50">
            </div>
            <div class="form-group">
                <label>Min Samples Split:</label>
                <input type="number" id="param_min_samples_split" value="5" min="2" max="20">
            </div>
        `;
    } else if (modelType === 'xgboost' || modelType === 'lightgbm') {
        html = `
            <div class="form-group">
                <label>Number of Estimators:</label>
                <input type="number" id="param_n_estimators" value="200" min="50" max="500">
            </div>
            <div class="form-group">
                <label>Max Depth:</label>
                <input type="number" id="param_max_depth" value="10" min="3" max="20">
            </div>
            <div class="form-group">
                <label>Learning Rate:</label>
                <input type="number" id="param_learning_rate" value="0.1" min="0.01" max="0.5" step="0.01">
            </div>
        `;
    } else if (modelType === 'neural_net') {
        html = `
            <div class="form-group">
                <label>Epochs:</label>
                <input type="number" id="param_epochs" value="50" min="10" max="200">
            </div>
            <div class="form-group">
                <label>Batch Size:</label>
                <input type="number" id="param_batch_size" value="32" min="16" max="128">
            </div>
            <div class="form-group">
                <label>Learning Rate:</label>
                <input type="number" id="param_learning_rate" value="0.001" min="0.0001" max="0.01" step="0.0001">
            </div>
            <div class="form-group">
                <label>Dropout Rate:</label>
                <input type="number" id="param_dropout_rate" value="0.3" min="0.1" max="0.5" step="0.1">
            </div>
        `;
    }

    hyperparamsDiv.innerHTML = html;
}

// Get hyperparameters from form
function getHyperparameters() {
    const params = {};
    const hyperparamsDiv = document.getElementById('hyperparameters');
    const inputs = hyperparamsDiv.querySelectorAll('input');

    inputs.forEach(input => {
        const paramName = input.id.replace('param_', '');
        params[paramName] = parseFloat(input.value);
    });

    return params;
}

// Train model
async function trainModel() {
    if (!currentDataPath) {
        showStatus('trainStatus', 'Please upload data first', 'error');
        return;
    }

    const modelType = document.getElementById('modelType').value;
    const targetColumn = document.getElementById('targetColumn').value;
    const testSize = parseFloat(document.getElementById('testSize').value);
    const useSmote = document.getElementById('useSmote').checked;
    const hyperparams = getHyperparameters();

    showStatus('trainStatus', 'Training model... This may take a few minutes.', 'info');

    try {
        const response = await fetch('/api/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentDataPath,
                model_type: modelType,
                target_column: targetColumn,
                test_size: testSize,
                use_smote: useSmote,
                hyperparams: hyperparams
            })
        });

        const data = await response.json();

        if (response.ok) {
            modelTrained = true;
            displayTrainingResults(data);
            showStatus('trainStatus', 'Model trained successfully!', 'success');

            // Show download button
            const downloadBtn = document.getElementById('downloadModelBtn');
            if (downloadBtn) {
                downloadBtn.style.display = 'inline-block';
            }
        } else {
            showStatus('trainStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('trainStatus', `Error: ${error.message}`, 'error');
    }
}

// Display training results
function displayTrainingResults(data) {
    const metricsDisplay = document.getElementById('metricsDisplay');
    const resultsContainer = document.getElementById('trainResults');

    const metrics = data.metrics;

    // Safety check for metrics object
    if (!metrics || Object.keys(metrics).length === 0) {
        metricsDisplay.innerHTML = '<div style="color: #f44336;">No metrics available</div>';
        return;
    }

    // Build HTML with safety checks for each metric
    let html = '<div class="metrics-grid">';

    if (metrics.accuracy !== undefined) {
        html += `
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</div>
            </div>`;
    }

    if (metrics.precision !== undefined) {
        html += `
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">${(metrics.precision * 100).toFixed(2)}%</div>
            </div>`;
    }

    if (metrics.recall !== undefined) {
        html += `
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">${(metrics.recall * 100).toFixed(2)}%</div>
            </div>`;
    }

    if (metrics.f1_score !== undefined) {
        html += `
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</div>
            </div>`;
    }

    if (metrics.roc_auc !== undefined) {
        html += `
            <div class="metric-card">
                <div class="metric-label">ROC AUC</div>
                <div class="metric-value">${(metrics.roc_auc * 100).toFixed(2)}%</div>
            </div>`;
    }

    html += '</div>';

    metricsDisplay.innerHTML = html;
    resultsContainer.classList.remove('hidden');
}

// Prediction options
function showPredictOption(option) {
    document.getElementById('predictFile').classList.add('hidden');
    document.getElementById('predictManual').classList.add('hidden');
    document.getElementById('predictTess').classList.add('hidden');
    document.getElementById('predictAdversarial').classList.add('hidden');

    if (option === 'file') {
        document.getElementById('predictFile').classList.remove('hidden');
    } else if (option === 'manual') {
        document.getElementById('predictManual').classList.remove('hidden');
    } else if (option === 'tess') {
        document.getElementById('predictTess').classList.remove('hidden');
    } else if (option === 'adversarial') {
        document.getElementById('predictAdversarial').classList.remove('hidden');
    }
}

// Download trained model
async function downloadModel() {
    try {
        showStatus('trainStatus', 'Preparing model download...', 'info');

        const response = await fetch('/api/download_model');

        if (!response.ok) {
            const data = await response.json();
            showStatus('trainStatus', `Error: ${data.error}`, 'error');
            return;
        }

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'exoplanet_model.zip';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Download the file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showStatus('trainStatus', 'Model downloaded successfully!', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showStatus('trainStatus', `Error: ${error.message}`, 'error');
    }
}

// Upload pretrained model
async function uploadModel() {
    const fileInput = document.getElementById('modelFile');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('trainStatus', 'Please select a model file', 'error');
        return;
    }

    if (!file.name.endsWith('.zip')) {
        showStatus('trainStatus', 'Please select a .zip file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showStatus('trainStatus', 'Loading model...', 'info');

    try {
        const response = await fetch('/api/upload_model', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('trainStatus', `Model loaded successfully! Type: ${data.model_type}`, 'success');

            // Display metrics if available and valid
            if (data.metrics && Object.keys(data.metrics).length > 0 && data.metrics.accuracy !== undefined) {
                displayTrainingResults(data.metrics);
            } else if (data.metrics && Object.keys(data.metrics).length > 0) {
                // Show basic metrics info if available
                const metricsDiv = document.getElementById('metricsDisplay');
                if (metricsDiv) {
                    let html = '<div style="background: #1e3a5f; padding: 15px; border-radius: 8px; margin-top: 15px;">';
                    html += '<h4 style="color: #64b5f6;">Model Metrics</h4>';
                    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">';
                    for (const [key, value] of Object.entries(data.metrics)) {
                        html += `<div><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</div>`;
                    }
                    html += '</div></div>';
                    metricsDiv.innerHTML = html;
                }
            }

            // Show download button
            const downloadBtn = document.getElementById('downloadModelBtn');
            if (downloadBtn) {
                downloadBtn.style.display = 'inline-block';
            }

            // Reload model stats
            loadModelStats();

            // Clear file input
            fileInput.value = '';
        } else {
            showStatus('trainStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showStatus('trainStatus', `Error: ${error.message}`, 'error');
    }
}

// Predict from file
async function predictFromFile() {
    const fileInput = document.getElementById('predictDataFile');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('predictStatus', 'Please select a file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showStatus('predictStatus', 'Uploading data...', 'info');

    try {
        // First upload the file
        const uploadResponse = await fetch('/api/upload_data', {
            method: 'POST',
            body: formData
        });

        const uploadData = await uploadResponse.json();

        if (!uploadResponse.ok) {
            showStatus('predictStatus', `Error: ${uploadData.error}`, 'error');
            return;
        }

        // Then predict
        showStatus('predictStatus', 'Making predictions...', 'info');

        const predictResponse = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: uploadData.filepath
            })
        });

        const predictData = await predictResponse.json();

        if (predictResponse.ok) {
            displayPredictions(predictData.predictions, predictData.summary);
            showStatus('predictStatus', 'Predictions completed!', 'success');
        } else {
            showStatus('predictStatus', `Error: ${predictData.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    }
}

// Predict from manual input
async function predictManual() {
    const inputs = {
        koi_period: document.getElementById('input_period').value,
        koi_duration: document.getElementById('input_duration').value,
        koi_depth: document.getElementById('input_depth').value,
        koi_prad: document.getElementById('input_prad').value,
        koi_teq: document.getElementById('input_teq').value,
        koi_insol: document.getElementById('input_insol').value,
        koi_model_snr: document.getElementById('input_snr').value,
        koi_steff: document.getElementById('input_steff').value
    };

    // Check if at least some values are provided
    const hasValues = Object.values(inputs).some(v => v !== '');
    if (!hasValues) {
        showStatus('predictStatus', 'Please enter at least some values', 'error');
        return;
    }

    showStatus('predictStatus', 'Making prediction...', 'info');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: inputs
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayPredictions(data.predictions);
            showStatus('predictStatus', 'Prediction completed!', 'success');
        } else {
            showStatus('predictStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    }
}

// Display predictions
function displayPredictions(predictions, summary) {
    const predictionsDisplay = document.getElementById('predictionsDisplay');
    const resultsContainer = document.getElementById('predictResults');

    let html = '';

    // Export buttons at the TOP (always visible)
    html += `
        <div style="background: #1e3a5f; padding: 15px; border-radius: 8px; margin-bottom: 20px; position: sticky; top: 0; z-index: 100;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <h4 style="color: #64b5f6; margin: 0; display: flex; align-items: center; gap: 8px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Export & Analytics
                </h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button onclick="exportToCSV()" class="btn" style="background: #2ecc71; padding: 10px 20px; border: none; border-radius: 6px; color: white; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        Download CSV
                    </button>
                    <button onclick="showStatistics()" class="btn" style="background: #3498db; padding: 10px 20px; border: none; border-radius: 6px; color: white; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="3" y1="9" x2="21" y2="9"></line>
                            <line x1="9" y1="21" x2="9" y2="9"></line>
                        </svg>
                        View Statistics
                    </button>
                </div>
            </div>
        </div>
    `;

    // Show summary for large batches
    if (summary) {
        html += `
            <div style="background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #64b5f6; margin-bottom: 10px;">Batch Prediction Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>Total Samples:</strong> ${summary.total_samples}</div>
                    <div><strong>Exoplanets Found:</strong> ${summary.total_exoplanets}</div>
                    <div><strong>Non-Exoplanets:</strong> ${summary.total_non_exoplanets}</div>
                    <div><strong>Avg Confidence:</strong> ${(summary.avg_confidence * 100).toFixed(1)}%</div>
                </div>
                <p style="margin-top: 10px; color: #b0b0b0;">Showing first ${summary.samples_returned} of ${summary.total_samples} predictions</p>
                <p style="margin-top: 5px; color: #64b5f6; display: flex; align-items: center; gap: 6px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                    Click "View Details" on any exoplanet to see visualizations and analysis
                </p>
            </div>
        `;
    }

    predictions.forEach((pred, idx) => {
        const isExoplanet = pred.prediction === 'Exoplanet';
        const confidence = (pred.confidence * 100).toFixed(2);

        html += `
            <div class="prediction-card">
                <div class="prediction-result ${isExoplanet ? 'exoplanet' : 'not-exoplanet'}">
                    ${idx > 0 ? `Sample ${idx + 1}: ` : ''}${pred.prediction}
                </div>
                <div class="confidence-bar" style="position: relative;">
                    <div class="confidence-fill" style="width: ${confidence}%">
                        ${confidence >= 15 ? `${confidence}% Confidence` : ''}
                    </div>
                    ${confidence < 15 ? `<span style="position: absolute; left: ${confidence}%; margin-left: 8px; top: 50%; transform: translateY(-50%); white-space: nowrap; color: #64b5f6;">${confidence}% Confidence</span>` : ''}
                </div>
                <div style="margin-top: 10px; color: #b0b0b0;">
                    Exoplanet Probability: ${(pred.probability_exoplanet * 100).toFixed(2)}%<br>
                    Not Exoplanet Probability: ${(pred.probability_not_exoplanet * 100).toFixed(2)}%
                </div>
        `;

        // Add TESS metadata if available
        if (pred.tess_metadata) {
            const tess = pred.tess_metadata;
            html += `
                <div style="margin-top: 15px; padding: 15px; background: #1e3a5f; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h4 style="color: #64b5f6; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <circle cx="12" cy="12" r="6"></circle>
                            <circle cx="12" cy="12" r="2"></circle>
                        </svg>
                        TESS Object of Interest (TOI)
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 13px;">
                        ${tess.toi ? `<div><strong style="color: #64b5f6;">TOI:</strong> ${tess.toi}</div>` : ''}
                        ${tess.tid ? `<div><strong style="color: #64b5f6;">TIC ID:</strong> ${tess.tid}</div>` : ''}
                        ${tess.ra ? `<div><strong style="color: #64b5f6;">RA:</strong> ${typeof tess.ra === 'number' ? tess.ra.toFixed(4) : tess.ra}°</div>` : ''}
                        ${tess.dec ? `<div><strong style="color: #64b5f6;">Dec:</strong> ${typeof tess.dec === 'number' ? tess.dec.toFixed(4) : tess.dec}°</div>` : ''}
                        ${tess.pl_orbper ? `<div><strong style="color: #64b5f6;">Period:</strong> ${typeof tess.pl_orbper === 'number' ? tess.pl_orbper.toFixed(2) : tess.pl_orbper} days</div>` : ''}
                        ${tess.pl_rade ? `<div><strong style="color: #64b5f6;">Radius:</strong> ${typeof tess.pl_rade === 'number' ? tess.pl_rade.toFixed(2) : tess.pl_rade} R⊕</div>` : ''}
                        ${tess.tfopwg_disp ? `<div><strong style="color: #64b5f6;">Status:</strong> ${tess.tfopwg_disp}</div>` : ''}
                        ${tess.toi_created ? `<div><strong style="color: #64b5f6;">Created:</strong> ${new Date(tess.toi_created).toLocaleDateString()}</div>` : ''}
                        ${tess.rowupdate ? `<div><strong style="color: #64b5f6;">Updated:</strong> ${new Date(tess.rowupdate).toLocaleDateString()}</div>` : ''}
                    </div>
                </div>
            `;
        }

        // Add Kepler metadata if available
        if (pred.kepler_metadata) {
            const kepler = pred.kepler_metadata;
            html += `
                <div style="margin-top: 15px; padding: 15px; background: #2c1a4d; border-radius: 8px; border-left: 4px solid #9b59b6;">
                    <h4 style="color: #bb86fc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 6v6l4 2"></path>
                        </svg>
                        Kepler Object of Interest (KOI)
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 13px;">
                        ${kepler.kepoi_name ? `<div><strong style="color: #bb86fc;">KOI:</strong> ${kepler.kepoi_name}</div>` : ''}
                        ${kepler.kepid ? `<div><strong style="color: #bb86fc;">Kepler ID:</strong> ${kepler.kepid}</div>` : ''}
                        ${kepler.kepler_name ? `<div><strong style="color: #bb86fc;">Planet Name:</strong> ${kepler.kepler_name}</div>` : ''}
                        ${kepler.ra ? `<div><strong style="color: #bb86fc;">RA:</strong> ${typeof kepler.ra === 'number' ? kepler.ra.toFixed(4) : kepler.ra}°</div>` : ''}
                        ${kepler.dec ? `<div><strong style="color: #bb86fc;">Dec:</strong> ${typeof kepler.dec === 'number' ? kepler.dec.toFixed(4) : kepler.dec}°</div>` : ''}
                        ${kepler.koi_period ? `<div><strong style="color: #bb86fc;">Period:</strong> ${typeof kepler.koi_period === 'number' ? kepler.koi_period.toFixed(2) : kepler.koi_period} days</div>` : ''}
                        ${kepler.koi_prad ? `<div><strong style="color: #bb86fc;">Radius:</strong> ${typeof kepler.koi_prad === 'number' ? kepler.koi_prad.toFixed(2) : kepler.koi_prad} R⊕</div>` : ''}
                        ${kepler.koi_pdisposition ? `<div><strong style="color: #bb86fc;">Disposition:</strong> ${kepler.koi_pdisposition}</div>` : ''}
                        ${kepler.koi_score !== undefined ? `<div><strong style="color: #bb86fc;">KOI Score:</strong> ${typeof kepler.koi_score === 'number' ? (kepler.koi_score * 100).toFixed(1) : kepler.koi_score}%</div>` : ''}
                    </div>
                </div>
            `;
        }

        // Add note about data quality for visualizations
        if (isExoplanet && !pred.light_curve_plot && pred.characteristics) {
            html += `
                <div style="margin-top: 15px; padding: 10px; background: rgba(255, 165, 0, 0.1); border-left: 3px solid #ff9800; border-radius: 4px; display: flex; align-items: start; gap: 8px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffb74d" stroke-width="2" style="flex-shrink: 0; margin-top: 2px;">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                    <small style="color: #ffb74d;">Light curve visualization not available - requires detailed planet metadata (radius, mass, temperature). Real Kepler data contains only transit signal features.</small>
                </div>
            `;
        }

        // Add planet characteristics
        if (isExoplanet && pred.characteristics) {
            const char = pred.characteristics;
            html += `
                <div style="margin-top: 15px; padding: 15px; background: #1a1a2e; border-radius: 8px;">
                    <h4 style="color: #64b5f6; margin-bottom: 10px;">Planet Characteristics</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                        <div style="padding: 10px; background: #16213e; border-radius: 6px;">
                            <div style="font-size: 24px; margin-bottom: 5px;">${char.size.emoji}</div>
                            <div style="font-weight: bold; color: ${char.size.color};">${char.size.category}</div>
                            <div style="font-size: 12px; color: #999;">${char.size.description}</div>
                        </div>
                        <div style="padding: 10px; background: #16213e; border-radius: 6px;">
                            <div style="font-size: 24px; margin-bottom: 5px;">${char.temperature.emoji}</div>
                            <div style="font-weight: bold; color: ${char.temperature.color};">${char.temperature.category}</div>
                            <div style="font-size: 12px; color: #999;">${char.temperature.description}</div>
                        </div>
                        <div style="padding: 10px; background: #16213e; border-radius: 6px;">
                            <div style="font-size: 24px; margin-bottom: 5px;">${char.star.emoji}</div>
                            <div style="font-weight: bold; color: ${char.star.color};">${char.star.category}</div>
                            <div style="font-size: 12px; color: #999;">${char.star.description}</div>
                        </div>
                        <div style="padding: 10px; background: #16213e; border-radius: 6px;">
                            <div style="font-size: 24px; margin-bottom: 5px;">${char.composition.emoji}</div>
                            <div style="font-weight: bold;">${char.composition.composition}</div>
                            <div style="font-size: 12px; color: #999;">${char.composition.description}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: #0f3460; border-radius: 6px; font-style: italic;">
                        ${char.summary}
                    </div>
                </div>
            `;
        }

        // Add enhanced features for detected exoplanets
        if (isExoplanet && pred.discovery_story) {
            html += `
                <div style="margin-top: 20px; padding: 15px; background: #1e3a5f; border-radius: 8px;">
                    <h3 style="color: #64b5f6; margin-bottom: 10px;">Discovery Report</h3>
                    <div style="white-space: pre-line; line-height: 1.6;">${pred.discovery_story}</div>
                </div>
            `;
        }

        // Add habitability score
        if (pred.habitability) {
            const hab = pred.habitability;
            html += `
                <div style="margin-top: 15px; padding: 15px; background: #2d1810; border-left: 4px solid #ff9800; border-radius: 4px;">
                    <h4 style="color: #ff9800; margin-bottom: 8px;">Habitability Assessment</h4>
                    <div style="margin-bottom: 10px;">
                        <strong>Score:</strong> ${(hab.overall_score * 100).toFixed(1)}/100
                        <div style="background: #444; height: 8px; border-radius: 4px; margin-top: 5px;">
                            <div style="background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
                                        width: ${(hab.overall_score * 100)}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    <div><strong>Classification:</strong> ${hab.classification}</div>
                    <div><strong>Assessment:</strong> ${hab.description}</div>
                </div>
            `;
        }

        // Add "View Details" button for exoplanets without visualizations loaded
        if (isExoplanet && !pred.light_curve_plot && pred.row_index !== undefined) {
            html += `
                <div style="margin-top: 15px; text-align: center;">
                    <button onclick="loadPlanetDetails(${pred.row_index}, ${idx}, '${summary?.filepath || ''}', ${pred.confidence})"
                            class="btn"
                            id="details_btn_${idx}"
                            style="background: #3498db; padding: 12px 24px; border: none; border-radius: 6px; color: white; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 8px; justify-content: center;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"></circle>
                            <path d="m21 21-4.35-4.35"></path>
                        </svg>
                        View Detailed Analysis
                    </button>
                    <div id="details_loading_${idx}" style="display: none; margin-top: 10px; color: #64b5f6;">
                        Loading visualizations...
                    </div>
                    <div id="details_container_${idx}"></div>
                </div>
            `;
        }

        // Add light curve visualization (only available with rich metadata)
        if (pred.light_curve_plot) {
            html += `
                <div style="margin-top: 15px;">
                    <h4 style="color: #64b5f6; margin-bottom: 8px;">Transit Light Curve</h4>
                    <div id="lightcurve_${idx}"></div>
                </div>
            `;
        }

        // Add comparison chart
        if (pred.comparison_chart) {
            html += `
                <div style="margin-top: 15px;">
                    <div id="comparison_${idx}"></div>
                </div>
            `;
        }

        html += `</div>`;
    });

    predictionsDisplay.innerHTML = html;

    // Render Plotly charts after DOM update
    predictions.forEach((pred, idx) => {
        if (pred.light_curve_plot) {
            const plotData = JSON.parse(pred.light_curve_plot);
            Plotly.newPlot(`lightcurve_${idx}`, plotData.data, plotData.layout);
        }
        if (pred.comparison_chart) {
            const plotData = JSON.parse(pred.comparison_chart);
            Plotly.newPlot(`comparison_${idx}`, plotData.data, plotData.layout);
        }
    });

    resultsContainer.classList.remove('hidden');

    // Store predictions globally for export
    window.currentPredictions = predictions;
    window.currentSummary = summary;

    // Show leaderboard for large batches
    if (summary && !summary.visualizations_included) {
        loadLeaderboard(predictions);
    }
}

async function exportToCSV() {
    if (!window.currentPredictions) {
        alert('No predictions to export');
        return;
    }

    try {
        const response = await fetch('/api/export_csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predictions: window.currentPredictions })
        });

        const data = await response.json();

        if (data.success) {
            // Create download link
            const blob = new Blob([data.csv_data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            showStatus('predictStatus', 'CSV exported successfully!', 'success');
        } else {
            showStatus('predictStatus', `Export error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Export error: ${error.message}`, 'error');
    }
}

async function loadLeaderboard(predictions) {
    try {
        const response = await fetch('/api/leaderboard', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predictions: predictions })
        });

        const data = await response.json();

        if (data.success && data.leaderboard.length > 0) {
            displayLeaderboard(data.leaderboard, data.total_habitable);
        }
    } catch (error) {
        console.error('Leaderboard error:', error);
    }
}

function displayLeaderboard(leaderboard, total) {
    const predictionsDisplay = document.getElementById('predictionsDisplay');

    let html = `
        <div style="margin-top: 20px; padding: 20px; background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%); border-radius: 8px; border: 2px solid #64b5f6;">
            <h3 style="color: #64b5f6; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>
                    <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>
                    <path d="M4 22h16"></path>
                    <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path>
                    <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path>
                    <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path>
                </svg>
                Top Habitable Planets Discovered
                <span style="font-size: 14px; color: #b0b0b0; margin-left: auto;">${total} total candidates</span>
            </h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #16213e; color: #64b5f6;">
                            <th style="padding: 10px; text-align: left;">Rank</th>
                            <th style="padding: 10px; text-align: left;">Planet</th>
                            <th style="padding: 10px; text-align: center;">Score</th>
                            <th style="padding: 10px; text-align: left;">Type</th>
                            <th style="padding: 10px; text-align: left;">Temperature</th>
                            <th style="padding: 10px; text-align: center;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    leaderboard.forEach((planet, idx) => {
        let medalSvg = '';
        if (idx === 0) {
            medalSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#FFD700" stroke="#B8860B" stroke-width="1.5"><circle cx="12" cy="8" r="6"/><path d="M15 13l-3 8-3-8"/></svg>';
        } else if (idx === 1) {
            medalSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#C0C0C0" stroke="#808080" stroke-width="1.5"><circle cx="12" cy="8" r="6"/><path d="M15 13l-3 8-3-8"/></svg>';
        } else if (idx === 2) {
            medalSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#CD7F32" stroke="#8B4513" stroke-width="1.5"><circle cx="12" cy="8" r="6"/><path d="M15 13l-3 8-3-8"/></svg>';
        } else {
            medalSvg = `<span style="font-weight: bold; color: #64b5f6;">#${idx + 1}</span>`;
        }
        const scoreColor = planet.score >= 0.7 ? '#2ecc71' : planet.score >= 0.5 ? '#f39c12' : '#e74c3c';

        html += `
            <tr style="border-bottom: 1px solid #2d5986; ${idx % 2 === 0 ? 'background: #1a2942;' : ''}">
                <td style="padding: 10px; font-size: 18px;">${medalSvg}</td>
                <td style="padding: 10px; font-weight: bold;">${planet.name}</td>
                <td style="padding: 10px; text-align: center;">
                    <span style="color: ${scoreColor}; font-weight: bold;">${(planet.score * 100).toFixed(1)}</span>
                    <div style="background: #0f1419; height: 6px; border-radius: 3px; margin-top: 4px;">
                        <div style="background: ${scoreColor}; width: ${planet.score * 100}%; height: 100%; border-radius: 3px;"></div>
                    </div>
                </td>
                <td style="padding: 10px;">${planet.size}</td>
                <td style="padding: 10px;">${planet.temperature}</td>
                <td style="padding: 10px; text-align: center;">${(planet.confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    predictionsDisplay.innerHTML += html;
}

function showStatistics() {
    if (!window.currentPredictions || !window.currentSummary) {
        alert('No statistics available');
        return;
    }

    const summary = window.currentSummary;
    const predictions = window.currentPredictions;

    // Calculate additional stats
    let habitableCount = 0;
    let avgHabScore = 0;
    let sizeBreakdown = {};
    let tempBreakdown = {};

    predictions.forEach(pred => {
        if (pred.habitability) {
            habitableCount++;
            avgHabScore += pred.habitability.overall_score;
        }
        if (pred.characteristics) {
            const size = pred.characteristics.size.category;
            const temp = pred.characteristics.temperature.category;
            sizeBreakdown[size] = (sizeBreakdown[size] || 0) + 1;
            tempBreakdown[temp] = (tempBreakdown[temp] || 0) + 1;
        }
    });

    avgHabScore = habitableCount > 0 ? avgHabScore / habitableCount : 0;

    // Create statistics modal
    const statsHtml = `
        <div id="statsModal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; display: flex; align-items: center; justify-content: center;" onclick="this.remove()">
            <div style="background: #1a1a2e; padding: 30px; border-radius: 12px; max-width: 800px; max-height: 90vh; overflow-y: auto;" onclick="event.stopPropagation()">
                <h2 style="color: #64b5f6; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="20" x2="12" y2="10"></line>
                        <line x1="18" y1="20" x2="18" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="16"></line>
                    </svg>
                    Discovery Statistics
                </h2>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; border-left: 4px solid #2ecc71;">
                        <div style="color: #b0b0b0; font-size: 12px;">Total Analyzed</div>
                        <div style="color: #fff; font-size: 24px; font-weight: bold;">${summary.total_samples}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                        <div style="color: #b0b0b0; font-size: 12px;">Exoplanets Found</div>
                        <div style="color: #fff; font-size: 24px; font-weight: bold;">${summary.total_exoplanets}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; border-left: 4px solid #f39c12;">
                        <div style="color: #b0b0b0; font-size: 12px;">Habitable Candidates</div>
                        <div style="color: #fff; font-size: 24px; font-weight: bold;">${habitableCount}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; border-left: 4px solid #e74c3c;">
                        <div style="color: #b0b0b0; font-size: 12px;">Avg Confidence</div>
                        <div style="color: #fff; font-size: 24px; font-weight: bold;">${(summary.avg_confidence * 100).toFixed(1)}%</div>
                    </div>
                </div>

                <h3 style="color: #64b5f6; margin: 20px 0 10px 0;">Planet Size Distribution</h3>
                <div style="background: #16213e; padding: 15px; border-radius: 8px;">
                    ${Object.entries(sizeBreakdown).map(([type, count]) => `
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span>${type}</span>
                                <span>${count}</span>
                            </div>
                            <div style="background: #0f1419; height: 8px; border-radius: 4px;">
                                <div style="background: #3498db; width: ${(count / predictions.length * 100)}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>

                <button onclick="document.getElementById('statsModal').remove()" style="margin-top: 20px; padding: 10px 20px; background: #e74c3c; color: white; border: none; border-radius: 6px; cursor: pointer;">
                    Close
                </button>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', statsHtml);
}

// Show visualization
async function showVisualization(vizType) {
    const vizContainer = document.getElementById('vizContainer');
    showStatus('vizStatus', 'Loading visualization...', 'info');

    try {
        let endpoint = '';
        if (vizType === 'metrics') {
            endpoint = '/api/visualization/metrics';
        } else if (vizType === 'confusion') {
            endpoint = '/api/visualization/confusion_matrix';
        } else if (vizType === 'importance') {
            endpoint = '/api/visualization/feature_importance';
        } else if (vizType === 'roc') {
            endpoint = '/api/visualization/roc';
        } else if (vizType === 'precision_recall') {
            endpoint = '/api/visualization/precision_recall';
        } else if (vizType === 'learning') {
            endpoint = '/api/visualization/learning_curves';
        }

        const response = await fetch(endpoint);

        if (!response.ok) {
            const data = await response.json();
            showStatus('vizStatus', `Error: ${data.error || 'Failed to load visualization'}`, 'error');
            return;
        }

        const data = await response.json();

        if (data.plot) {
            const plotData = JSON.parse(data.plot);
            Plotly.newPlot('vizContainer', plotData.data, plotData.layout, {responsive: true});
            document.getElementById('vizStatus').classList.add('hidden');
        } else {
            showStatus('vizStatus', 'Error: No plot data returned', 'error');
        }
    } catch (error) {
        console.error('Visualization error:', error);
        showStatus('vizStatus', `Error: ${error.message}`, 'error');
    }
}

// Load model statistics
async function loadModelStats() {
    try {
        const response = await fetch('/api/model_stats');
        const data = await response.json();

        if (data.trained) {
            const statsDiv = document.getElementById('currentModelStats');
            const metrics = data.metrics;

            let html = '<div class="metrics-grid">';

            // Always show model type
            html += `
                <div class="metric-card">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value" style="font-size: 1.2em;">${data.model_type}</div>
                </div>`;

            // Only show metrics that exist
            if (metrics && metrics.accuracy !== undefined) {
                html += `
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</div>
                    </div>`;
            }

            if (metrics && metrics.f1_score !== undefined) {
                html += `
                    <div class="metric-card">
                        <div class="metric-label">F1 Score</div>
                        <div class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</div>
                    </div>`;
            }

            if (metrics && metrics.roc_auc !== undefined) {
                html += `
                    <div class="metric-card">
                        <div class="metric-label">ROC AUC</div>
                        <div class="metric-value">${(metrics.roc_auc * 100).toFixed(2)}%</div>
                    </div>`;
            }

            html += '</div>';

            statsDiv.innerHTML = html;

            // Hide Feature Importance button for neural networks
            const featureImportanceBtn = document.getElementById('featureImportanceBtn');
            if (featureImportanceBtn) {
                if (data.model_type === 'neural_net') {
                    featureImportanceBtn.style.display = 'none';
                } else {
                    featureImportanceBtn.style.display = 'inline-block';
                }
            }
        }
    } catch (error) {
        console.error('Error loading model stats:', error);
    }
}

// Load planet details on-demand
async function loadPlanetDetails(rowIndex, displayIndex, filepath, confidence) {
    const btn = document.getElementById(`details_btn_${displayIndex}`);
    const loading = document.getElementById(`details_loading_${displayIndex}`);
    const container = document.getElementById(`details_container_${displayIndex}`);

    // Hide button, show loading
    btn.style.display = 'none';
    loading.style.display = 'block';

    try {
        const response = await fetch(`/api/visualize_planet/${rowIndex}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: filepath,
                confidence: confidence
            })
        });

        const data = await response.json();

        if (response.ok) {
            let html = '';

            // Add discovery story
            if (data.discovery_story) {
                html += `
                    <div style="margin-top: 20px; padding: 15px; background: #1e3a5f; border-radius: 8px;">
                        <h3 style="color: #64b5f6; margin-bottom: 10px;">Discovery Report</h3>
                        <div style="white-space: pre-line; line-height: 1.6;">${data.discovery_story}</div>
                    </div>
                `;
            }

            // Add light curve
            if (data.light_curve_plot) {
                html += `
                    <div style="margin-top: 15px;">
                        <h4 style="color: #64b5f6; margin-bottom: 8px;">Transit Light Curve</h4>
                        <div id="lightcurve_detail_${displayIndex}"></div>
                    </div>
                `;
            }

            // Add phase folded plot
            if (data.phase_folded_plot) {
                html += `
                    <div style="margin-top: 15px;">
                        <h4 style="color: #64b5f6; margin-bottom: 8px;">Phase-Folded Light Curve</h4>
                        <div id="phasefolded_detail_${displayIndex}"></div>
                    </div>
                `;
            }

            // Add comparison chart
            if (data.comparison_chart) {
                html += `
                    <div style="margin-top: 15px;">
                        <h4 style="color: #64b5f6; margin-bottom: 8px;">Comparison with Solar System</h4>
                        <div id="comparison_detail_${displayIndex}"></div>
                    </div>
                `;
            }

            container.innerHTML = html;
            loading.style.display = 'none';

            // Render Plotly charts
            if (data.light_curve_plot) {
                const plotData = JSON.parse(data.light_curve_plot);
                Plotly.newPlot(`lightcurve_detail_${displayIndex}`, plotData.data, plotData.layout, {responsive: true});
            }
            if (data.phase_folded_plot) {
                const plotData = JSON.parse(data.phase_folded_plot);
                Plotly.newPlot(`phasefolded_detail_${displayIndex}`, plotData.data, plotData.layout, {responsive: true});
            }
            if (data.comparison_chart) {
                const plotData = JSON.parse(data.comparison_chart);
                Plotly.newPlot(`comparison_detail_${displayIndex}`, plotData.data, plotData.layout, {responsive: true});
            }
        } else {
            loading.innerHTML = `<span style="color: #e74c3c;">Error: ${data.error}</span>`;
            btn.style.display = 'block';
        }
    } catch (error) {
        loading.innerHTML = `<span style="color: #e74c3c;">Error loading details: ${error.message}</span>`;
        btn.style.display = 'block';
    }
}

// Show status message
function showStatus(elementId, message, type) {
    const statusBox = document.getElementById(elementId);
    statusBox.textContent = message;
    statusBox.className = `status-box ${type}`;
    statusBox.classList.remove('hidden');
}

// Advanced Feature 1: Fetch Recent TESS Data
async function fetchRecentTESS() {
    const lookbackDays = document.getElementById('tessLookbackDays').value;

    showStatus('predictStatus', `Fetching TESS discoveries from last ${lookbackDays} days...`, 'info');

    try {
        const response = await fetch('/api/fetch_recent_tess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                lookback_days: parseInt(lookbackDays)
            })
        });

        const data = await response.json();

        if (response.ok) {
            if (data.count > 0) {
                if (data.predictions && data.predictions.length > 0) {
                    displayPredictions(data.predictions, data.summary);
                    showStatus('predictStatus',
                        `Fetched ${data.count} recent TESS TOIs and made predictions! (Saved to ${data.filepath})`,
                        'success');
                } else {
                    showStatus('predictStatus',
                        `Fetched ${data.count} recent TESS TOIs but no model is trained yet. Please train a model first on Kepler data.`,
                        'info');
                }
            } else {
                showStatus('predictStatus',
                    `No new TESS TOIs found in the last ${lookbackDays} days.`,
                    'info');
            }
        } else {
            showStatus('predictStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    }
}

// Advanced Feature 2: Generate Adversarial Test Cases
async function generateAdversarial() {
    const count = document.getElementById('adversarialCount').value;

    showStatus('predictStatus', 'Generating adversarial test cases...', 'info');

    try {
        const response = await fetch('/api/generate_adversarial', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                n_samples: parseInt(count)
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Store filepath for testing
            adversarialFilePath = data.filepath;

            showStatus('predictStatus',
                `Generated ${data.count} adversarial test cases (${data.types.join(', ')})! Saved to ${data.filepath}`,
                'success');

            // Enable the test button
            document.getElementById('testAdversarialBtn').disabled = false;
        } else {
            showStatus('predictStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    }
}

// Advanced Feature 3: Test Adversarial Robustness
async function testAdversarial() {
    if (!adversarialFilePath) {
        showStatus('predictStatus', 'Please generate adversarial test cases first', 'error');
        return;
    }

    showStatus('predictStatus', 'Testing model robustness on adversarial cases...', 'info');

    try {
        const response = await fetch('/api/test_adversarial', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: adversarialFilePath
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayAdversarialResults(data);
            showStatus('predictStatus',
                `Adversarial testing complete! Rejection rate: ${(data.rejection_rate * 100).toFixed(1)}%`,
                'success');
        } else {
            showStatus('predictStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    }
}

// Display adversarial test results
function displayAdversarialResults(data) {
    const resultsDisplay = document.getElementById('predictionsDisplay');
    const resultsContainer = document.getElementById('predictResults');

    let html = '<div style="padding: 20px;">';
    html += '<h3 style="color: #64b5f6; margin-bottom: 15px;">Adversarial Robustness Results</h3>';
    html += `<div style="background: #1e3a5f; padding: 15px; border-radius: 8px; margin-bottom: 20px;">`;
    html += `<div style="font-size: 16px; margin-bottom: 8px;"><strong>Total Test Cases:</strong> ${data.total_cases}</div>`;
    html += `<div style="font-size: 16px; margin-bottom: 8px; color: #4caf50;"><strong>Correctly Rejected (as fake):</strong> ${data.rejected_count} (${(data.rejection_rate * 100).toFixed(1)}%)</div>`;
    html += `<div style="font-size: 16px; color: #f44336;"><strong>Incorrectly Accepted (as real):</strong> ${data.accepted_count} (${(data.acceptance_rate * 100).toFixed(1)}%)</div>`;
    html += `</div>`;

    html += '<h4 style="color: #64b5f6; margin: 20px 0 10px 0;">Breakdown by Type:</h4>';
    html += '<table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">';
    html += '<thead><tr style="background: #1e3a5f;"><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Type</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Count</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Rejected</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Rejection Rate</th></tr></thead>';
    html += '<tbody>';

    for (const [type, stats] of Object.entries(data.by_type)) {
        html += `<tr style="background: #16213e;">`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${type.replace(/_/g, ' ')}</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${stats.total}</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${stats.rejected}</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${(stats.rejection_rate * 100).toFixed(1)}%</td>`;
        html += `</tr>`;
    }

    html += '</tbody></table>';

    html += '<h4 style="color: #64b5f6; margin: 20px 0 10px 0;">Sample Results (First 20):</h4>';
    html += '<table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr style="background: #1e3a5f;"><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Type</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Prediction</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Confidence</th><th style="padding: 12px; text-align: left; border: 1px solid #2c5282;">Result</th></tr></thead>';
    html += '<tbody>';

    data.sample_results.slice(0, 20).forEach(result => {
        const bgColor = result.prediction === 0 ? '#1b5e20' : '#7f0000';
        const resultText = result.prediction === 0 ? 'CORRECTLY REJECTED ✓' : 'INCORRECTLY ACCEPTED ✗';
        html += `<tr style="background: ${bgColor};">`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${result.type.replace(/_/g, ' ')}</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${result.prediction === 1 ? 'Planet' : 'Not Planet'}</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;">${(result.confidence * 100).toFixed(1)}%</td>`;
        html += `<td style="padding: 10px; border: 1px solid #2c5282;"><strong>${resultText}</strong></td>`;
        html += `</tr>`;
    });

    html += '</tbody></table>';
    html += '</div>';

    resultsDisplay.innerHTML = html;
    resultsContainer.classList.remove('hidden');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateHyperparameters();
    loadModelStats();
});
