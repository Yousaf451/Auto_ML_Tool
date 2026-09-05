import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';

// Register Chart.js components for scatter plot (used for heatmap alternative)
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const CLASSIFICATION_METRICS = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1 score'],
  ['roc_auc', 'ROC-AUC'], ['log_loss', 'Log loss'], ['balanced_accuracy', 'Balanced accuracy'],
  ['specificity', 'Specificity'], ['mcc', 'MCC']
];
const REGRESSION_METRICS = [
  ['mae', 'MAE'], ['mse', 'MSE'], ['rmse', 'RMSE'], ['r2', 'R²'], ['mape', 'MAPE'],
  ['median_absolute_error', 'Median absolute error']
];
const METRIC_LABELS = Object.fromEntries([...CLASSIFICATION_METRICS, ...REGRESSION_METRICS]);
const MODEL_DESCRIPTIONS = {
  logistic_regression: 'Fast linear baseline for classification.',
  random_forest_classifier: 'Strong general-purpose ensemble classifier.',
  svc: 'Flexible classifier for complex boundaries.',
  decision_tree_classifier: 'Interpretable tree-based classifier.',
  knn_classifier: 'Instance-based classifier for local patterns.',
  gradient_boosting_classifier: 'High-performing sequential ensemble.',
  linear_regression: 'Simple, transparent regression baseline.',
  random_forest_regressor: 'Robust ensemble for nonlinear regression.',
  svr: 'Flexible support-vector regression model.',
  decision_tree_regressor: 'Interpretable tree-based regressor.',
  knn_regressor: 'Instance-based regressor for local patterns.',
  gradient_boosting_regressor: 'High-performing sequential ensemble.'
};

const getErrorMessage = (err, fallback = 'Unknown error') => {
  if (typeof err?.response?.data === 'string') {
    return err.response.data;
  }

  const detail = err?.response?.data?.detail ?? err?.response?.data?.message;
  if (detail) {
    return typeof detail === 'string' ? detail : JSON.stringify(detail);
  }

  if (err?.code === 'ERR_NETWORK') {
    return 'The server could not be reached. Check that the backend is running.';
  }

  return err?.message || fallback;
};

// -------- Helper Components --------

// Simple Spinner
const Spinner = () => (
  <div style={{ textAlign: 'center', padding: '20px' }}>
    <div className="spinner" style={{
      border: '4px solid #f3f3f3',
      borderTop: '4px solid #3498db',
      borderRadius: '50%',
      width: '40px',
      height: '40px',
      animation: 'spin 1s linear infinite',
      margin: 'auto'
    }} />
    <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    <p>Loading...</p>
  </div>
);

// Correlation Heatmap using Chart.js (as a custom chart)
const CorrelationHeatmap = ({ columns, matrix }) => {
  if (!columns.length || !matrix.length) {
    return <p>No numeric columns found for correlation.</p>;
  }

  // Prepare data for a heatmap-like chart. We'll use a bubble chart as a simple heatmap.
  // Better: use a plugin like 'chartjs-chart-matrix', but to keep dependencies minimal, we'll use a colored table instead.
  // For simplicity, we'll render an HTML table with background colors based on correlation values.

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', margin: '10px 0' }}>
        <thead>
          <tr>
            <th></th>
            {columns.map(col => <th key={col} style={{ padding: '5px', textAlign: 'center' }}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {columns.map((rowCol, i) => (
            <tr key={rowCol}>
              <th style={{ padding: '5px', textAlign: 'right' }}>{rowCol}</th>
              {columns.map((col, j) => {
                const value = matrix[i][j];
                const color = value > 0 ? `rgba(0, 128, 0, ${Math.min(value, 1)})` :
                              value < 0 ? `rgba(255, 0, 0, ${Math.min(-value, 1)})` : 'rgba(0,0,0,0.1)';
                return (
                  <td key={col} style={{
                    padding: '5px',
                    textAlign: 'center',
                    backgroundColor: color,
                    color: '#fff',
                    fontSize: '0.8em'
                  }}>
                    {value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ---------- Main App Component ----------
function App() {
  // State for dataset and EDA
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null); // {dataset_id, preview, columns, shape}
  const [correlation, setCorrelation] = useState(null); // {columns, matrix}
  const [targetColumn, setTargetColumn] = useState('');
  const [suggestedTask, setSuggestedTask] = useState('');
  const [selectedTask, setSelectedTask] = useState('');

  // State for search configuration
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [searchStrategy, setSearchStrategy] = useState('random');
  const [nIter, setNIter] = useState(10);
  const [evaluationMetrics, setEvaluationMetrics] = useState(['accuracy']);
  const [primaryMetric, setPrimaryMetric] = useState('accuracy');
  const [customParams, setCustomParams] = useState({}); // store overridden hyperparameters

  // State for job
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('');
  const [jobProgress, setJobProgress] = useState(0);
  const [jobError, setJobError] = useState(null);
  const [results, setResults] = useState([]);
  const [modelMetadata, setModelMetadata] = useState(null);
  const previewRows = Array.isArray(datasetInfo?.preview) ? datasetInfo.preview.slice(0, 5) : [];

  const downloadArtifact = async (path, fallbackName) => {
    try {
      const response = await axios.get(`${API_BASE_URL}${path}`, { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = url;
      link.download = fallbackName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Download failed: ' + getErrorMessage(err));
    }
  };

  // Polling effect
  useEffect(() => {
    let interval;
    if (jobId && (jobStatus === 'pending' || jobStatus === 'running')) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE_URL}/api/jobs/${jobId}`);
          const job = res.data;
          setJobStatus(job.status);
          setJobProgress(job.progress);
          if (job.error) setJobError(job.error);
          if (job.status === 'completed') {
            clearInterval(interval);
            // Fetch results
            const resultsRes = await axios.get(`${API_BASE_URL}/api/jobs/${jobId}/results`);
            setResults(resultsRes.data.results);
            setEvaluationMetrics(resultsRes.data.evaluation_metrics || evaluationMetrics);
            setPrimaryMetric(resultsRes.data.primary_metric || primaryMetric);
            setModelMetadata(resultsRes.data.model_metadata || null);
          }
        } catch (err) {
          console.error('Polling error', err);
          clearInterval(interval);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  // Update available models based on selected task
  useEffect(() => {
    let models = [];
    if (selectedTask === 'classification') {
      models = [
        'logistic_regression',
        'random_forest_classifier',
        'svc',
        'decision_tree_classifier',
        'knn_classifier',
        'gradient_boosting_classifier'
      ];
    } else if (selectedTask === 'regression') {
      models = [
        'linear_regression',
        'random_forest_regressor',
        'svr',
        'decision_tree_regressor',
        'knn_regressor',
        'gradient_boosting_regressor'
      ];
    }
    setAvailableModels(models);
    setSelectedModels([]); // reset selection when task changes
    const defaultMetric = selectedTask === 'classification' ? 'accuracy' : 'r2';
    setEvaluationMetrics([defaultMetric]);
    setPrimaryMetric(defaultMetric);
  }, [selectedTask]);

  // Handle file upload
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setDatasetInfo(res.data);
      // Reset other states
      setCorrelation(null);
      setTargetColumn('');
      setSuggestedTask('');
      setSelectedTask('');
      setResults([]);
      setModelMetadata(null);
      setJobId(null);
      setJobStatus('');
      setJobProgress(0);
      setJobError(null);
      setSelectedModels([]);
      setCustomParams({});
    } catch (err) {
      const message = getErrorMessage(err, 'Upload failed');
      setUploadError(message);
      alert('Upload failed: ' + message);
    } finally {
      setUploading(false);
    }
  };

  // Fetch correlation when dataset uploaded
  const fetchCorrelation = async () => {
    if (!datasetInfo?.dataset_id) return;
    try {
      const res = await axios.get(`${API_BASE_URL}/api/datasets/${datasetInfo.dataset_id}/correlation`);
      setCorrelation(res.data);
    } catch (err) {
      console.error('Correlation fetch failed', err);
    }
  };

  useEffect(() => {
    if (datasetInfo?.dataset_id) {
      fetchCorrelation();
    }
  }, [datasetInfo?.dataset_id]);

  // Select target column
  const handleTargetSelect = async (col) => {
    setTargetColumn(col);
    if (!datasetInfo?.dataset_id) return;
    try {
      const res = await axios.post(`${API_BASE_URL}/api/datasets/${datasetInfo.dataset_id}/target`, {
        target_column: col
      });
      setSuggestedTask(res.data.suggested_task);
      setSelectedTask(res.data.suggested_task);
    } catch (err) {
      console.error('Target selection failed', err);
    }
  };

  // Toggle model selection
  const toggleModel = (model) => {
    setSelectedModels(prev => prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]);
  };

  // Run search
  const handleRunSearch = async () => {
    if (!datasetInfo?.dataset_id || !targetColumn || !selectedTask || selectedModels.length === 0 || evaluationMetrics.length === 0 || !evaluationMetrics.includes(primaryMetric)) {
      alert('Please select a dataset, target column, task, model, and at least one evaluation metric.');
      return;
    }
    try {
      const payload = {
        dataset_id: datasetInfo.dataset_id,
        target_column: targetColumn,
        task: selectedTask,
        models: selectedModels,
        search_strategy: searchStrategy,
        evaluation_metrics: evaluationMetrics,
        primary_metric: primaryMetric,
        n_iter: nIter,
        cv: 5
      };
      const res = await axios.post(`${API_BASE_URL}/api/search`, payload);
      setJobId(res.data.job_id);
      setJobStatus(res.data.status);
      setJobProgress(0);
      setJobError(null);
      setResults([]);
    } catch (err) {
      alert('Search failed: ' + getErrorMessage(err));
    }
  };

  const progressSteps = ['Upload', 'Explore', 'Configure', 'Train', 'Evaluate', 'Deploy'];
  const currentStep = results.length ? 5 : jobId ? 4 : targetColumn ? 3 : datasetInfo ? 2 : 1;
  const metricOptions = selectedTask === 'regression' ? REGRESSION_METRICS : CLASSIFICATION_METRICS;
  const featureColumns = modelMetadata?.feature_columns || [];
  const formatNumber = (value) => typeof value === 'number' ? value.toLocaleString() : value;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">M</span><span>Mini <b>AutoML</b></span></div>
        <nav className="nav-links" aria-label="Primary navigation"><a className="active" href="#workspace">Workspace</a><a href="#dataset">Dataset</a><a href="#automl">AutoML</a><a href="#results">Models</a></nav>
        <button className="icon-button" title="Help" aria-label="Help">?</button>
      </header>

      <main id="workspace" className="workspace">
        <section className="hero-section">
          <div><span className="eyebrow">AUTOMATED MODEL LAB</span><h1>From raw data to a model you can trust.</h1><p>Upload a dataset, discover its shape, and let a focused model search find your strongest baseline.</p></div>
          <div className="hero-note"><span className="status-dot" /> <span>{jobStatus === 'running' ? 'Training in progress' : results.length ? 'Model ready to deploy' : 'Workspace ready'}</span><small>One clear workflow, every step visible.</small></div>
        </section>

        <div className="stepper" aria-label="Workflow progress">
          {progressSteps.map((step, index) => <div className={`step ${index + 1 < currentStep ? 'complete' : ''} ${index + 1 === currentStep ? 'current' : ''}`} key={step}><span className="step-number">{index + 1 < currentStep ? '✓' : index + 1}</span><span>{step}</span></div>)}
        </div>

        <section id="dataset" className="section-block">
          <div className="section-heading"><div><span className="section-index">01</span><div><h2>Bring your dataset</h2><p>Start with a CSV or Excel file. We will inspect it before training.</p></div></div><span className="section-status">{datasetInfo ? 'Ready' : 'Next step'}</span></div>
          <div className={`upload-panel ${datasetInfo ? 'uploaded' : ''}`}>
            <div className="upload-icon">↥</div><div><h3>{datasetInfo ? 'Dataset uploaded' : 'Drop your dataset here'}</h3><p>{file ? file.name : 'CSV, XLSX, or XLS · up to 10 MB'}</p></div>
            <label className="button secondary-button">Browse files<input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange} /></label>
            <button className="button primary-button" onClick={handleUpload} disabled={!file || uploading}>{uploading ? 'Uploading...' : 'Upload dataset'}</button>
          </div>
          {uploadError && <div className="error-state" style={{ marginTop: '12px' }}><strong>Upload failed</strong><p>{uploadError}</p></div>}
          {!datasetInfo && <div className="empty-state"><span>◌</span><div><strong>No dataset connected</strong><p>Upload a file to unlock preview, analysis, and target selection.</p></div></div>}
          {datasetInfo && <div className="dataset-summary"><div><span>Dataset</span><strong>{file?.name || 'Uploaded dataset'}</strong></div><div><span>Rows</span><strong>{formatNumber(datasetInfo.shape[0])}</strong></div><div><span>Columns</span><strong>{datasetInfo.shape[1]}</strong></div><div><span>Status</span><strong className="text-success">● Ready</strong></div></div>}
        </section>

        {datasetInfo && <section className="section-block" id="explore">
          <div className="section-heading"><div><span className="section-index">02</span><div><h2>Understand your data</h2><p>A quick read of the dataset before we build anything.</p></div></div><span className="section-status">Complete</span></div>
          <div className="stat-grid"><div className="stat-card"><span>Rows</span><strong>{formatNumber(datasetInfo.shape[0])}</strong><small>observations</small></div><div className="stat-card"><span>Columns</span><strong>{datasetInfo.shape[1]}</strong><small>fields detected</small></div><div className="stat-card"><span>Numeric</span><strong>{datasetInfo.columns.filter(col => /int|float/.test(col.dtype)).length}</strong><small>quantitative fields</small></div><div className="stat-card"><span>Missing</span><strong>{formatNumber(datasetInfo.columns.reduce((sum, col) => sum + col.missing, 0))}</strong><small>values to handle</small></div></div>
          <div className="data-grid"><div className="panel"><div className="panel-heading"><div><h3>Dataset preview</h3><p>First five rows from your uploaded file.</p></div><span className="badge">Preview</span></div><div className="table-wrap"><table className="data-table"><thead><tr>{datasetInfo.columns.map(col => <th key={col.column}>{col.column}</th>)}</tr></thead><tbody>{previewRows.map((row, i) => <tr key={i}>{datasetInfo.columns.map(col => <td key={col.column}>{String(row[col.column] ?? '')}</td>)}</tr>)}</tbody></table></div></div><div className="panel"><div className="panel-heading"><div><h3>Column information</h3><p>Types and data quality at a glance.</p></div></div><div className="table-wrap"><table className="data-table compact"><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>{datasetInfo.columns.map(col => <tr key={col.column}><td><strong>{col.column}</strong></td><td><span className="type-pill">{col.dtype}</span></td><td>{col.missing}</td><td>{formatNumber(col.unique)}</td></tr>)}</tbody></table></div></div></div>
          <div className="eda-panel"><div className="panel-heading"><div><h3>Correlation analysis</h3><p>Relationships between numerical features. Values near +1 or -1 indicate stronger relationships.</p></div></div>{correlation ? <CorrelationHeatmap columns={correlation.columns} matrix={correlation.matrix} /> : <Spinner />}</div>
        </section>}

        {datasetInfo && <section className="section-block" id="automl">
          <div className="section-heading"><div><span className="section-index">03</span><div><h2>Choose what to predict</h2><p>Tell the platform which column contains the outcome you care about.</p></div></div><span className="section-status">{targetColumn ? 'Complete' : 'In progress'}</span></div>
          <div className="target-layout"><div className="target-card"><label htmlFor="target-column">Target column <span className="help-tip" title="The value your model learns to predict.">?</span></label><select id="target-column" value={targetColumn} onChange={(e) => handleTargetSelect(e.target.value)}><option value="">Select a target column</option>{datasetInfo.columns.map(col => <option key={col.column} value={col.column}>{col.column}</option>)}</select>{targetColumn && <div className="confirmation">✓ Target column selected</div>}</div>{suggestedTask && <div className="task-card"><span className="label">Task type</span><div className="segmented"><button className={selectedTask === 'classification' ? 'selected' : ''} onClick={() => setSelectedTask('classification')}>Classification</button><button className={selectedTask === 'regression' ? 'selected' : ''} onClick={() => setSelectedTask('regression')}>Regression</button></div><small>Auto-detected as <b>{suggestedTask}</b></small></div>}</div>
        </section>}

        {datasetInfo && targetColumn && selectedTask && <section className="section-block" id="configure">
          <div className="section-heading"><div><span className="section-index">04</span><div><h2>Configure AutoML</h2><p>Pick the models and metrics that matter for this experiment.</p></div></div><span className="section-status">{selectedModels.length ? 'Configured' : 'Choose models'}</span></div>
          <div className="config-grid"><div className="config-column"><div className="subheading"><h3>Model selection</h3><button className="text-button" onClick={() => setSelectedModels(selectedModels.length === availableModels.length ? [] : availableModels)}>{selectedModels.length === availableModels.length ? 'Clear all' : 'Select all'}</button></div><div className="model-grid">{availableModels.map(model => <label className={`model-option ${selectedModels.includes(model) ? 'selected' : ''}`} key={model}><input type="checkbox" checked={selectedModels.includes(model)} onChange={() => toggleModel(model)} /><span className="checkmark">{selectedModels.includes(model) ? '✓' : ''}</span><span><strong>{model.replace(/_/g, ' ')}</strong><small>{MODEL_DESCRIPTIONS[model]}</small></span></label>)}</div></div><div className="config-column settings-column"><div className="subheading"><h3>Search strategy</h3><span className="help-tip" title="More iterations can improve the search but take longer.">?</span></div><div className="strategy-row"><label><input type="radio" value="random" checked={searchStrategy === 'random'} onChange={(e) => setSearchStrategy(e.target.value)} /> Random search</label><label><input type="radio" value="grid" checked={searchStrategy === 'grid'} onChange={(e) => setSearchStrategy(e.target.value)} /> Grid search</label></div>{searchStrategy === 'random' && <label className="field-label">Iterations<input type="number" value={nIter} min={1} max={500} onChange={(e) => setNIter(parseInt(e.target.value) || 1)} /></label>}<div className="subheading metric-heading"><h3>Evaluation metrics</h3><button className="text-button" onClick={() => setEvaluationMetrics(evaluationMetrics.length === metricOptions.length ? [] : metricOptions.map(([value]) => value))}>{evaluationMetrics.length === metricOptions.length ? 'Clear all' : 'Select all'}</button></div><div className="metric-list">{metricOptions.map(([value, label]) => <label key={value}><input type="checkbox" checked={evaluationMetrics.includes(value)} onChange={() => { const next = evaluationMetrics.includes(value) ? evaluationMetrics.filter(name => name !== value) : [...evaluationMetrics, value]; setEvaluationMetrics(next); if (!next.includes(primaryMetric)) setPrimaryMetric(next[0] || ''); }} /><span>{label}</span></label>)}</div><label className="field-label">Primary metric <span className="help-tip" title="Used to rank models and choose the best result.">?</span><select value={primaryMetric} onChange={(e) => setPrimaryMetric(e.target.value)} disabled={!evaluationMetrics.length}>{evaluationMetrics.map(value => <option key={value} value={value}>{METRIC_LABELS[value]}</option>)}</select></label></div></div>
          <div className="run-bar"><div><strong>Ready to search</strong><span>{selectedModels.length} models · {evaluationMetrics.length} metrics · primary: {METRIC_LABELS[primaryMetric] || 'none'}</span></div><button className="button primary-button run-button" onClick={handleRunSearch} disabled={!selectedModels.length || !evaluationMetrics.length}>Run AutoML <span>→</span></button></div>
        </section>}

        {jobId && <section className={`section-block progress-section ${jobStatus === 'completed' ? 'success-section' : ''}`}><div className="section-heading"><div><span className="section-index">05</span><div><h2>{jobStatus === 'completed' ? 'Training complete' : 'Training your models'}</h2><p>{jobStatus === 'completed' ? 'Your results are ready to review.' : 'You can keep this page open while the search runs.'}</p></div></div><span className="section-status">{jobStatus}</span></div>{jobStatus === 'pending' || jobStatus === 'running' ? <><div className="progress-label"><strong>{Math.round(jobProgress * 100)}%</strong><span>Overall progress</span></div><div className="progress-track"><div style={{ width: `${Math.round(jobProgress * 100)}%` }} /></div><div className="progress-meta"><span>Job is evaluating {selectedModels.length} selected models</span><span>{Math.round(jobProgress * 100)}% complete</span></div></> : jobStatus === 'failed' ? <div className="error-state"><strong>Training could not complete</strong><p>{jobError}</p></div> : <div className="success-banner">✓ AutoML training completed successfully</div>}</section>}

        {results.length > 0 && <section className="section-block results-section" id="results"><div className="section-heading"><div><span className="section-index">06</span><div><h2>Compare and deploy</h2><p>The strongest candidate is ranked by your primary metric.</p></div></div><span className="section-status success">Best result found</span></div><div className="best-card"><div className="best-copy"><span className="eyebrow">RECOMMENDED MODEL</span><h2>{results[0].model_name.replace(/_/g, ' ')}</h2><p>Best fit for your selected task and evaluation criteria.</p></div><div className="best-score"><span>{METRIC_LABELS[primaryMetric] || primaryMetric}</span><strong>{results[0].mean_score.toFixed(4)}</strong><small>primary score</small></div></div>{modelMetadata && <div className="download-card"><div><h3>Your model is ready</h3><p>The download includes preprocessing, encoding, and the fitted model.</p></div><div className="button-row"><button className="button primary-button" onClick={() => downloadArtifact(`/api/jobs/${jobId}/model/download`, modelMetadata.model_file)}>Download model <small>.joblib</small></button><button className="button secondary-button" onClick={() => downloadArtifact(`/api/jobs/${jobId}/input-template`, modelMetadata.input_template_file)}>Input template <small>.csv</small></button></div></div>}<div className="panel comparison-panel"><div className="panel-heading"><div><h3>Model comparison</h3><p>All selected metrics, ordered by {METRIC_LABELS[primaryMetric] || primaryMetric}.</p></div></div><div className="table-wrap"><table className="data-table comparison-table"><thead><tr><th>Rank</th><th>Model</th>{evaluationMetrics.map(metric => <th key={metric}>{METRIC_LABELS[metric]}</th>)}<th>Time</th></tr></thead><tbody>{results.map((result, index) => <tr className={index === 0 ? 'best-row' : ''} key={index}><td>{index === 0 ? <span className="rank-badge">✓ 1</span> : index + 1}</td><td><strong>{result.model_name.replace(/_/g, ' ')}</strong></td>{evaluationMetrics.map(metric => <td key={metric}>{result.metric_scores?.[metric]?.mean?.toFixed(4) ?? '-'}</td>)}<td>{result.training_time.toFixed(2)}s</td></tr>)}</tbody></table></div></div>{modelMetadata && <div className="details-grid"><div className="panel"><div className="panel-heading"><div><h3>What data does my model expect?</h3><p>{modelMetadata.training_rows.toLocaleString()} training rows · {featureColumns.length} input features · target excluded from prediction data.</p></div></div><div className="table-wrap"><table className="data-table compact"><thead><tr><th>Feature</th><th>Type</th><th>Example</th></tr></thead><tbody>{featureColumns.map(feature => <tr key={feature}><td><strong>{feature}</strong></td><td><span className="type-pill">{modelMetadata.feature_dtypes[feature]}</span></td><td>{String(previewRows[0]?.[feature] ?? '')}</td></tr>)}</tbody></table></div></div><div className="panel usage-panel"><div className="panel-heading"><div><h3>Use your downloaded model</h3><p>Load the pipeline and pass a CSV with the same feature columns.</p></div></div><pre>{`import joblib\nimport pandas as pd\n\nmodel = joblib.load("${modelMetadata.model_file}")\ndata = pd.read_csv("new_data.csv")\npredictions = model.predict(data)\nprint(predictions)${modelMetadata.supports_predict_proba ? '\nprobabilities = model.predict_proba(data)' : ''}`}</pre><button className="text-button" onClick={() => downloadArtifact(`/api/jobs/${jobId}/input-template`, modelMetadata.input_template_file)}>Download input template →</button></div></div>}</section>}
      </main>
    </div>
  );
}

export default App;