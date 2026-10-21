import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Activity,
  Eye,
  UploadCloud,
  FileImage,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  RefreshCw,
  Printer,
  Download,
  Info,
  Layers,
  ShieldCheck,
  Cpu,
  Sparkles,
  User,
  Sliders,
  ChevronRight,
  X
} from 'lucide-react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  // --------------------------------------------------------------------------
  // State: Patient Metadata
  // --------------------------------------------------------------------------
  const [patientId, setPatientId] = useState('OCT-PT-84210');
  const [eyeSelection, setEyeSelection] = useState('OD'); // OD: Right, OS: Left, OU: Both

  // --------------------------------------------------------------------------
  // State: File Upload & Preview
  // --------------------------------------------------------------------------
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // --------------------------------------------------------------------------
  // State: Analysis & Explainability Results
  // --------------------------------------------------------------------------
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // --------------------------------------------------------------------------
  // State: Visualization Settings
  // --------------------------------------------------------------------------
  const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side', 'blend', 'heatmap-only', 'raw-only'
  const [blendOpacity, setBlendOpacity] = useState(0.7);

  // --------------------------------------------------------------------------
  // State: Backend Connectivity / Health
  // --------------------------------------------------------------------------
  const [systemHealth, setSystemHealth] = useState({
    status: 'checking',
    device: 'CPU',
    isCuda: false,
    model: 'convnext_tiny'
  });

  // Check backend health on mount and periodically
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`, { timeout: 3000 });
      if (res.data && res.data.status === 'ok') {
        setSystemHealth({
          status: 'online',
          device: res.data.device.toUpperCase(),
          isCuda: res.data.is_cuda,
          model: res.data.model
        });
      }
    } catch (err) {
      setSystemHealth(prev => ({
        ...prev,
        status: 'offline'
      }));
    }
  };

  // Generate a random patient ID
  const handleRandomizePatient = () => {
    const randomNum = Math.floor(10000 + Math.random() * 90000);
    setPatientId(`OCT-PT-${randomNum}`);
  };

  // --------------------------------------------------------------------------
  // File Upload Handlers
  // --------------------------------------------------------------------------
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const processSelectedFile = (file) => {
    setError(null);
    setResults(null);
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // --------------------------------------------------------------------------
  // Quick Preset Sample Loader (For Instant Demo / Verification)
  // --------------------------------------------------------------------------
  const handleLoadSample = async (sampleFileName, gradeLabel) => {
    try {
      setError(null);
      setResults(null);
      const res = await fetch(`/samples/${sampleFileName}`);
      if (!res.ok) {
        throw new Error(`Failed to load ${sampleFileName}`);
      }
      const blob = await res.blob();
      const file = new File([blob], sampleFileName, { type: 'image/jpeg' });
      processSelectedFile(file);
    } catch (err) {
      setError(`Could not load preset sample scan: ${err.message}`);
    }
  };

  // --------------------------------------------------------------------------
  // Send Image to FastAPI Inference Endpoint
  // --------------------------------------------------------------------------
  const handleAnalyzeScan = async () => {
    if (!selectedFile) {
      setError('Please select or drop an OCT B-scan image first.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000
      });

      setResults(response.data);
    } catch (err) {
      console.error('Inference error:', err);
      const msg = err.response?.data?.detail || err.message || 'Error communicating with the diagnostic backend.';
      setError(msg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // --------------------------------------------------------------------------
  // Helper: Visual Diagnosis Badges & Interpretations
  // --------------------------------------------------------------------------
  const getDiagnosisDetails = (classId) => {
    switch (classId) {
      case 0:
        return {
          title: 'No DME (Normal)',
          gradeCode: 'Grade 0',
          color: '#10b981',
          bannerClass: 'grade-0',
          badgeText: 'Normal Foveal Architecture',
          icon: CheckCircle2,
          pathology: 'Normal foveal depression with preserved retinal neurosensory layers. Intact Inner Limiting Membrane (ILM) and Retinal Pigment Epithelium (RPE). Zero intraretinal cystoid spaces or subretinal fluid observed.',
          recommendation: [
            'Maintain routine annual diabetic eye screening.',
            'Optimize systemic parameters (HbA1c < 7.0%, blood pressure control).',
            'Patient education on early visual distortion symptoms (Amsler Grid).'
          ]
        };
      case 1:
        return {
          title: 'NCI-DME',
          gradeCode: 'Grade 1',
          color: '#f59e0b',
          bannerClass: 'grade-1',
          badgeText: 'Non-Center-Involving DME',
          icon: AlertTriangle,
          pathology: 'Intraretinal fluid accumulation and hyporeflective cystoid spaces localized to inner/outer macular rings, sparing the 1-mm central foveal subfield. Retinal layer architecture partially compromised peripherally.',
          recommendation: [
            'Short-interval clinical monitoring with repeat OCT scan in 3 to 4 months.',
            'Assess for glycemic instability or concurrent hypertensive retinopathy.',
            'Consider focal/grid laser photocoagulation if lesions progress near central 1-mm subfield.'
          ]
        };
      case 2:
        return {
          title: 'CI-DME',
          gradeCode: 'Grade 2',
          color: '#ef4444',
          bannerClass: 'grade-2',
          badgeText: 'Center-Involving DME',
          icon: AlertOctagon,
          pathology: 'Significant macular edema with diffuse intraretinal cystoid cavities and/or subretinal neurosensory detachment directly engaging the 1-mm central macular subfield. Blunting of the foveal depression and elevated central subfield thickness (CST).',
          recommendation: [
            'Urgent referral to retina specialist for anti-VEGF intravitreal pharmacotherapy (Aflibercept, Ranibizumab, or Faricimab).',
            'Consider intravitreal dexamethasone implant in pseudophakic or vitrectomized patients.',
            'Immediate intervention required to prevent irreversible photoreceptor apoptosis and visual acuity decline.'
          ]
        };
      default:
        return {
          title: 'Unknown',
          gradeCode: 'N/A',
          color: '#64748b',
          bannerClass: '',
          badgeText: 'Unknown',
          icon: Info,
          pathology: '',
          recommendation: []
        };
    }
  };

  const diagDetails = results ? getDiagnosisDetails(results.predicted_class_id) : null;

  // Calculate SVG stroke offset for confidence gauge
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = results
    ? circumference - (results.confidence_percentage / 100) * circumference
    : circumference;

  return (
    <div className="app-container">
      {/* ====================================================================
          1. Professional Medical UI Header
         ==================================================================== */}
      <header className="clinical-header">
        <div className="header-branding">
          <div className="brand-icon-wrapper">
            <Eye size={28} />
          </div>
          <div className="header-titles">
            <h1 className="platform-title">
              Diabetic Macular Edema (DME) AI Diagnostic Platform
              <span className="platform-badge">Clinical v1.0</span>
            </h1>
            <p className="platform-subtitle">
              Deep Learning Spectral-Domain OCT B-Scan Decision Support System • ConvNeXt-Tiny & Grad-CAM
            </p>
          </div>
        </div>

        <div className="header-system-status">
          <div className="status-pill">
            <span
              className={`status-indicator ${
                systemHealth.status === 'online' ? 'online' : 'offline'
              }`}
            />
            <span>
              {systemHealth.status === 'online'
                ? 'Backend Ready'
                : 'Connecting to AI Backend...'}
            </span>
            {systemHealth.status === 'online' && (
              <span className="device-badge">
                <Cpu size={12} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} />
                {systemHealth.device}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* ====================================================================
          2. Main Layout Grid (Input Controls + Results Dashboard)
         ==================================================================== */}
      <main className="main-dashboard-grid">
        {/* ==================================================================
            LEFT COLUMN: Metadata & Scan Input
           ================================================================== */}
        <section className="input-column">
          {/* Patient Metadata Panel */}
          <div className="glass-card metadata-panel">
            <div className="card-header-bar">
              <div className="card-title-group">
                <User size={18} className="card-icon" />
                <h2 className="card-title">Patient Metadata</h2>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>HIPAA / Pseudo-ID</span>
            </div>

            {/* Patient ID Input */}
            <div className="form-group">
              <label className="form-label" htmlFor="patient-id-input">
                <span>Patient ID / Medical Record Number</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Required</span>
              </label>
              <div className="input-with-button">
                <input
                  id="patient-id-input"
                  type="text"
                  className="text-input"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g. OCT-PT-84210"
                />
                <button
                  type="button"
                  className="btn-secondary-sm"
                  onClick={handleRandomizePatient}
                  title="Generate new pseudonymous patient ID"
                >
                  <RefreshCw size={13} />
                  <span>Random</span>
                </button>
              </div>
            </div>

            {/* Eye Selection OD / OS / OU */}
            <div className="form-group">
              <span className="form-label">
                <span>Eye Examined (Laterality)</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Clinical laterality</span>
              </span>
              <div className="eye-selector-group">
                <div
                  className={`eye-pill ${eyeSelection === 'OD' ? 'active' : ''}`}
                  onClick={() => setEyeSelection('OD')}
                >
                  <span className="eye-code">OD</span>
                  <span className="eye-desc">Right Eye (Dexter)</span>
                </div>
                <div
                  className={`eye-pill ${eyeSelection === 'OS' ? 'active' : ''}`}
                  onClick={() => setEyeSelection('OS')}
                >
                  <span className="eye-code">OS</span>
                  <span className="eye-desc">Left Eye (Sinister)</span>
                </div>
                <div
                  className={`eye-pill ${eyeSelection === 'OU' ? 'active' : ''}`}
                  onClick={() => setEyeSelection('OU')}
                >
                  <span className="eye-code">OU</span>
                  <span className="eye-desc">Bilateral</span>
                </div>
              </div>
            </div>

            {/* Quick Demo Presets */}
            <div className="presets-section">
              <span className="form-label">
                <span>Quick Clinical Presets</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>1-Click Verification</span>
              </span>
              <div className="preset-pills-row">
                <button
                  type="button"
                  className="preset-pill-btn preset-normal"
                  onClick={() => handleLoadSample('sample_grade0_normal.jpg', 'Normal')}
                >
                  <span className="preset-grade">Grade 0</span>
                  <span className="preset-name">Normal</span>
                </button>
                <button
                  type="button"
                  className="preset-pill-btn preset-ncidme"
                  onClick={() => handleLoadSample('sample_grade1_ncidme.jpg', 'NCI-DME')}
                >
                  <span className="preset-grade">Grade 1</span>
                  <span className="preset-name">NCI-DME</span>
                </button>
                <button
                  type="button"
                  className="preset-pill-btn preset-cidme"
                  onClick={() => handleLoadSample('sample_grade2_cidme.jpg', 'CI-DME')}
                >
                  <span className="preset-grade">Grade 2</span>
                  <span className="preset-name">CI-DME</span>
                </button>
              </div>
            </div>
          </div>

          {/* Image Upload Area */}
          <div className="glass-card">
            <div className="card-header-bar">
              <div className="card-title-group">
                <FileImage size={18} className="card-icon" />
                <h2 className="card-title">OCT B-Scan Upload</h2>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                PNG, JPG, TIFF (224x224 resampled)
              </span>
            </div>

            {!previewUrl ? (
              <div
                className={`dropzone ${isDragging ? 'drag-active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="dropzone-icon-box">
                  <UploadCloud size={28} />
                </div>
                <p className="dropzone-prompt">
                  <span>Click to browse</span> or drag and drop OCT B-scan here
                </p>
                <p className="dropzone-subtext">
                  Supports spectral-domain macula central scans (.png, .jpg, .jpeg, .tif)
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden-file-input"
                  accept="image/png, image/jpeg, image/jpg, image/tiff, image/tif"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="preview-card">
                <div className="preview-image-container">
                  <img src={previewUrl} alt="Patient OCT B-Scan Preview" className="preview-img" />
                </div>
                <div className="preview-overlay-bar">
                  <div className="file-info">
                    <FileImage size={14} />
                    <span>{selectedFile?.name || 'Uploaded OCT Scan'}</span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      ({selectedFile ? (selectedFile.size / 1024).toFixed(1) : 0} KB)
                    </span>
                  </div>
                  <button
                    type="button"
                    className="btn-remove-preview"
                    onClick={handleClearFile}
                    title="Remove scan and select another"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="error-banner" style={{ marginTop: 16 }}>
                <AlertOctagon size={18} style={{ flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}

            {/* Analyze Scan Button */}
            <div style={{ marginTop: 20 }}>
              <button
                type="button"
                className="btn-analyze-scan"
                disabled={!selectedFile || isAnalyzing}
                onClick={handleAnalyzeScan}
              >
                {isAnalyzing ? (
                  <>
                    <div className="spinner" />
                    <span>Executing ConvNeXt Inference & Grad-CAM...</span>
                  </>
                ) : (
                  <>
                    <Activity size={18} />
                    <span>Analyze Scan</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* ==================================================================
            RIGHT COLUMN: Clinical Decision Support & Explainability Dashboard
           ================================================================== */}
        <section className="results-column">
          {!results ? (
            /* Empty State */
            <div className="results-empty-state">
              <div className="empty-state-icon">
                <Activity size={36} />
              </div>
              <h2 className="empty-state-title">Awaiting OCT B-Scan for Analysis</h2>
              <p className="empty-state-desc">
                Select or drag a retinal OCT scan, choose an eye examination laterality, and click{' '}
                <strong>"Analyze Scan"</strong> to generate deep learning classification and explainability heatmaps.
              </p>
            </div>
          ) : (
            /* Results Presentation */
            <>
              {/* Primary Diagnosis Banner */}
              <div className={`diagnosis-banner ${diagDetails?.bannerClass}`}>
                <div className="diagnosis-info-left">
                  <div className="diagnosis-subtitle-tag">
                    <Sparkles size={14} />
                    <span>AI Model Primary Diagnosis • Patient: {patientId} ({eyeSelection})</span>
                  </div>
                  <div className="diagnosis-headline">
                    {diagDetails?.title}
                  </div>
                  <div className="diagnosis-description">
                    {diagDetails?.badgeText}
                  </div>
                </div>

                <div className="diagnosis-badge-container">
                  <span className="grade-chip">
                    {diagDetails?.gradeCode}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    PyTorch ConvNeXt-Tiny
                  </span>
                </div>
              </div>

              {/* Metric Cards: Confidence Gauge & Model Stats */}
              <div className="metric-cards-grid">
                {/* Confidence Card with Animated SVG Circular Gauge */}
                <div className="metric-card">
                  <div className="confidence-gauge-container">
                    <svg className="gauge-svg" viewBox="0 0 72 72">
                      <circle
                        className="gauge-bg-circle"
                        cx="36"
                        cy="36"
                        r={radius}
                      />
                      <circle
                        className="gauge-progress-circle"
                        cx="36"
                        cy="36"
                        r={radius}
                        stroke={diagDetails?.color}
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                      />
                    </svg>
                    <div className="gauge-center-text">
                      {results.confidence_percentage}%
                    </div>
                  </div>
                  <div className="metric-details">
                    <span className="metric-label">Confidence Score</span>
                    <span className="metric-value">{results.confidence_percentage}%</span>
                    <span className="metric-sub">
                      {results.confidence_percentage >= 90
                        ? 'High Diagnostic Agreement'
                        : 'Moderate Confidence'}
                    </span>
                  </div>
                </div>

                {/* Model Architecture Card */}
                <div className="metric-card">
                  <div className="metric-details">
                    <span className="metric-label">Model Architecture</span>
                    <span className="metric-value">ConvNeXt-Tiny</span>
                    <span className="metric-sub">Trained Backbone (PyTorch & timm)</span>
                  </div>
                </div>

                {/* Processing Resolution & Layer */}
                <div className="metric-card">
                  <div className="metric-details">
                    <span className="metric-label">Explainability Layer</span>
                    <span className="metric-value">Grad-CAM</span>
                    <span className="metric-sub">stages[-1].blocks[-1].norm</span>
                  </div>
                </div>
              </div>

              {/* Per-Class Probability Breakdown Bars */}
              <div className="glass-card probabilities-card">
                <div className="card-header-bar" style={{ marginBottom: 10, paddingBottom: 10 }}>
                  <div className="card-title-group">
                    <Activity size={16} className="card-icon" />
                    <h3 className="card-title" style={{ fontSize: '0.98rem' }}>
                      Per-Class Softmax Probability Distribution
                    </h3>
                  </div>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Sum: 100%</span>
                </div>

                {/* Grade 0: No DME */}
                <div className={`prob-row ${results.predicted_class_id === 0 ? 'active' : ''}`}>
                  <div className="prob-labels">
                    <div className="prob-name-group">
                      <span className="prob-dot" style={{ backgroundColor: '#10b981' }} />
                      <span>Grade 0: No DME (Normal)</span>
                    </div>
                    <span className="prob-val">
                      {(results.class_probabilities?.['No DME'] ?? 0).toFixed(2)}%
                    </span>
                  </div>
                  <div className="prob-bar-track">
                    <div
                      className="prob-bar-fill"
                      style={{
                        width: `${results.class_probabilities?.['No DME'] ?? 0}%`,
                        backgroundColor: '#10b981'
                      }}
                    />
                  </div>
                </div>

                {/* Grade 1: NCI-DME */}
                <div className={`prob-row ${results.predicted_class_id === 1 ? 'active' : ''}`}>
                  <div className="prob-labels">
                    <div className="prob-name-group">
                      <span className="prob-dot" style={{ backgroundColor: '#f59e0b' }} />
                      <span>Grade 1: NCI-DME (Non-Center-Involving)</span>
                    </div>
                    <span className="prob-val">
                      {(results.class_probabilities?.['NCI-DME'] ?? 0).toFixed(2)}%
                    </span>
                  </div>
                  <div className="prob-bar-track">
                    <div
                      className="prob-bar-fill"
                      style={{
                        width: `${results.class_probabilities?.['NCI-DME'] ?? 0}%`,
                        backgroundColor: '#f59e0b'
                      }}
                    />
                  </div>
                </div>

                {/* Grade 2: CI-DME */}
                <div className={`prob-row ${results.predicted_class_id === 2 ? 'active' : ''}`}>
                  <div className="prob-labels">
                    <div className="prob-name-group">
                      <span className="prob-dot" style={{ backgroundColor: '#ef4444' }} />
                      <span>Grade 2: CI-DME (Center-Involving)</span>
                    </div>
                    <span className="prob-val">
                      {(results.class_probabilities?.['CI-DME'] ?? 0).toFixed(2)}%
                    </span>
                  </div>
                  <div className="prob-bar-track">
                    <div
                      className="prob-bar-fill"
                      style={{
                        width: `${results.class_probabilities?.['CI-DME'] ?? 0}%`,
                        backgroundColor: '#ef4444'
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Side-by-Side Visual Comparison & Explainability Heatmap */}
              <div className="glass-card visual-comparison-card">
                <div className="comparison-toolbar">
                  <div className="card-title-group">
                    <Layers size={18} className="card-icon" />
                    <h3 className="card-title">Side-by-Side Explainability Comparison</h3>
                  </div>

                  {/* View Mode Switcher */}
                  <div className="view-mode-tabs">
                    <button
                      type="button"
                      className={`view-mode-tab ${viewMode === 'side-by-side' ? 'active' : ''}`}
                      onClick={() => setViewMode('side-by-side')}
                    >
                      Side-by-Side
                    </button>
                    <button
                      type="button"
                      className={`view-mode-tab ${viewMode === 'blend' ? 'active' : ''}`}
                      onClick={() => setViewMode('blend')}
                    >
                      Overlay Blend
                    </button>
                    <button
                      type="button"
                      className={`view-mode-tab ${viewMode === 'heatmap-only' ? 'active' : ''}`}
                      onClick={() => setViewMode('heatmap-only')}
                    >
                      Heatmap Only
                    </button>
                    <button
                      type="button"
                      className={`view-mode-tab ${viewMode === 'raw-only' ? 'active' : ''}`}
                      onClick={() => setViewMode('raw-only')}
                    >
                      Raw Scan
                    </button>
                  </div>
                </div>

                {/* Optional Blend Opacity Slider */}
                {viewMode === 'blend' && (
                  <div className="overlay-slider-control">
                    <Sliders size={14} />
                    <span>Grad-CAM Overlay Intensity:</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={blendOpacity}
                      onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
                      className="slider-input"
                    />
                    <span>{Math.round(blendOpacity * 100)}%</span>
                  </div>
                )}

                {/* Viewports Rendering */}
                {viewMode === 'side-by-side' && (
                  <div className="comparison-viewport-grid">
                    {/* Left: Raw OCT Scan */}
                    <div className="viewport-panel">
                      <div className="viewport-header">
                        <span>Patient Raw OCT B-Scan</span>
                        <span className="viewport-tag">Input Image</span>
                      </div>
                      <div className="viewport-body">
                        <img
                          src={previewUrl}
                          alt="Raw OCT Scan"
                          className="viewport-img"
                        />
                      </div>
                    </div>

                    {/* Right: Grad-CAM Explainability Heatmap */}
                    <div className="viewport-panel">
                      <div className="viewport-header">
                        <span>Grad-CAM Explainability Heatmap</span>
                        <span className="viewport-tag" style={{ color: 'var(--accent-cyan)' }}>
                          Feature Attention
                        </span>
                      </div>
                      <div className="viewport-body">
                        <img
                          src={`data:image/png;base64,${results.gradcam_heatmap_base64}`}
                          alt="Grad-CAM Explainability Heatmap"
                          className="viewport-img"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {viewMode === 'blend' && (
                  <div className="comparison-viewport-grid single-view">
                    <div className="viewport-panel">
                      <div className="viewport-header">
                        <span>OCT B-Scan with Grad-CAM Blended Overlay</span>
                        <span className="viewport-tag" style={{ color: 'var(--accent-cyan)' }}>
                          Opacity: {Math.round(blendOpacity * 100)}%
                        </span>
                      </div>
                      <div className="viewport-body">
                        <div className="blend-container">
                          <img
                            src={previewUrl}
                            alt="Raw OCT Scan Base"
                            className="blend-base-img"
                          />
                          <img
                            src={`data:image/png;base64,${results.gradcam_heatmap_base64}`}
                            alt="Grad-CAM Overlay"
                            className="blend-heat-img"
                            style={{ opacity: blendOpacity }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {viewMode === 'heatmap-only' && (
                  <div className="comparison-viewport-grid single-view">
                    <div className="viewport-panel">
                      <div className="viewport-header">
                        <span>Grad-CAM Feature Activation Heatmap</span>
                        <span className="viewport-tag" style={{ color: 'var(--accent-cyan)' }}>
                          Model Attention
                        </span>
                      </div>
                      <div className="viewport-body">
                        <img
                          src={`data:image/png;base64,${results.gradcam_heatmap_base64}`}
                          alt="Grad-CAM Heatmap"
                          className="viewport-img"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {viewMode === 'raw-only' && (
                  <div className="comparison-viewport-grid single-view">
                    <div className="viewport-panel">
                      <div className="viewport-header">
                        <span>Patient Raw OCT B-Scan</span>
                        <span className="viewport-tag">High-Resolution</span>
                      </div>
                      <div className="viewport-body">
                        <img
                          src={previewUrl}
                          alt="Raw OCT Scan"
                          className="viewport-img"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Heatmap Colormap Interpretation Legend */}
                <div className="heatmap-legend-bar">
                  <span>Low Activation (Normal Retinal Tissue)</span>
                  <div className="legend-gradient-line" />
                  <span style={{ color: '#f87171', fontWeight: 600 }}>
                    High Activation (Fluid Cavity / Edema Focus)
                  </span>
                </div>
              </div>

              {/* Dynamic Clinical Summary Notes Box */}
              <div className="glass-card clinical-summary-box">
                <div className="card-header-bar">
                  <div className="card-title-group">
                    <ShieldCheck size={18} className="card-icon" />
                    <h3 className="card-title">Automated Clinical Decision Support Summary</h3>
                  </div>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    Generated for {patientId} • Eye: {eyeSelection}
                  </span>
                </div>

                <div className="summary-sections-grid">
                  {/* Morphologic Pathology Findings */}
                  <div className="summary-block">
                    <div className="summary-block-title">
                      <Info size={14} />
                      <span>Retinal Morphologic Findings</span>
                    </div>
                    <p className="summary-text">{diagDetails?.pathology}</p>
                  </div>

                  {/* Clinical Recommendations & Protocol */}
                  <div className="summary-block">
                    <div className="summary-block-title">
                      <ChevronRight size={14} />
                      <span>Actionable Ophthalmologic Guidance</span>
                    </div>
                    <ul className="summary-bullet-list">
                      {diagDetails?.recommendation.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Diagnostic Action Toolbar */}
                <div className="action-toolbar">
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    * Decision support aid. All algorithmic findings should be correlated with comprehensive slit-lamp fundus biomicroscopy.
                  </div>
                  <div className="action-buttons-group">
                    <button
                      type="button"
                      className="btn-action btn-action-outline"
                      onClick={() => {
                        const a = document.createElement('a');
                        a.href = `data:image/png;base64,${results.gradcam_heatmap_base64}`;
                        a.download = `${patientId}_${eyeSelection}_gradcam_heatmap.png`;
                        a.click();
                      }}
                    >
                      <Download size={14} />
                      <span>Export Heatmap</span>
                    </button>

                    <button
                      type="button"
                      className="btn-action btn-action-primary"
                      onClick={() => window.print()}
                    >
                      <Printer size={14} />
                      <span>Print Diagnostic Report</span>
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
