import React, { useState, useEffect, useCallback, useRef } from 'react';

import {
  LayoutDashboard,
  Video,
  Camera,
  BarChart3,
  Settings,
  Search,
  CheckCircle2,
  Filter,
  RefreshCw,
  User,
  Clock,
  Radio,
  Upload,
  X,
  AlertTriangle,
} from 'lucide-react';

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import validHygieneClasses from '../hygiene_classes.json';

// Set VITE_API_URL for a deployed backend; local development uses FastAPI's
// standard address.  Keeping one source of truth prevents GET/POST split-brain.
const BACKEND_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const WEBSOCKET_URL = BACKEND_URL.replace(/^http/, 'ws');
const VALID_HYGIENE_CLASSES = new Set(validHygieneClasses);
const normalizeHygieneClass = (value) => String(value || '')
  .trim().toLowerCase().replace(/[-_]/g, ' ').replace(/\s+/g, ' ');

const Dashboard = () => {
  /* =========================================================
     GENERAL STATE
  ========================================================= */

  const [activeTab, setActiveTab] = useState('Dashboard');
  const [alerts, setAlerts] = useState([]);

  /* =========================================================
     ANALYTICS STATE
  ========================================================= */

  const [timeframe, setTimeframe] = useState('1h');

  const [analytics, setAnalytics] = useState({
    timeframe: '1h',
    total_detections: 0,
    compliance_score: 100,
    class_counts: {},
    status_counts: {
      compliant: 0,
      violation: 0,
    },
  });

  const [highlights, setHighlights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastSynced, setLastSynced] = useState(null);

  /* =========================================================
     VIDEO ANALYSIS STATE
  ========================================================= */

  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState('');
  const [detectionStats, setDetectionStats] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  const fileInputRef = useRef(null);

  /* =========================================================
     LIVE STREAM STATE
  ========================================================= */

  const [isStreamActive, setIsStreamActive] = useState(false);
  const [streamError, setStreamError] = useState(false);

  /* =========================================================
     SIDEBAR
  ========================================================= */

  const sidebarLinks = [
    {
      name: 'Dashboard',
      icon: LayoutDashboard,
    },
    {
      name: 'Live Monitoring',
      icon: Camera,
    },
    {
      name: 'Video Analysis',
      icon: Video,
    },
    {
      name: 'Reports',
      icon: BarChart3,
    },
    {
      name: 'Settings',
      icon: Settings,
    },
  ];

  /* =========================================================
     FETCH ANALYTICS
  ========================================================= */

  const fetchAnalytics = useCallback(async () => {
    try {
      const response = await fetch(
        `${BACKEND_URL}/api/reports/analytics?timeframe=${timeframe}`
      );

      if (!response.ok) {
        throw new Error(`Analytics request failed: ${response.status}`);
      }

      const data = await response.json();

      setAnalytics({
        timeframe: data.timeframe || timeframe,
        total_detections: Number(data.total_detections) || 0,
        compliance_score:
          typeof data.compliance_score === 'number'
            ? data.compliance_score
            : 100,
        class_counts: Object.fromEntries(Object.entries(data.class_counts || {}).filter(([name]) =>
          VALID_HYGIENE_CLASSES.has(normalizeHygieneClass(name))
        )),
        status_counts: {
          compliant: Number(data.status_counts?.compliant) || 0,
          violation: Number(data.status_counts?.violation) || 0,
        },
      });
    } catch (error) {
      console.warn('Analytics fetch error:', error);
    }
  }, [timeframe]);

  /* =========================================================
     FETCH HIGHLIGHTS
  ========================================================= */

  const fetchHighlights = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/highlights`);

      if (!response.ok) {
        throw new Error(`Highlights request failed: ${response.status}`);
      }

      const data = await response.json();

      setHighlights(Array.isArray(data) ? data.filter((clip) => {
        const name = normalizeHygieneClass(clip.class_name || clip.reason?.replace(/^Violation:\s*/i, ''));
        return VALID_HYGIENE_CLASSES.has(name) && name.startsWith('no ');
      }) : []);
    } catch (error) {
      console.warn('Highlights fetch error:', error);
    }
  }, []);

  /* =========================================================
     REFRESH DATA
  ========================================================= */

  const refreshAllData = useCallback(async () => {
    setLoading(true);

    try {
      await Promise.all([
        fetchAnalytics(),
        fetchHighlights(),
      ]);

      setLastSynced(new Date().toLocaleTimeString());
    } catch (error) {
      console.warn('Refresh error:', error);
    } finally {
      setLoading(false);
    }
  }, [fetchAnalytics, fetchHighlights]);

  /* =========================================================
     AUTO REFRESH
  ========================================================= */

  useEffect(() => {
    refreshAllData();

    const interval = setInterval(() => {
      refreshAllData();
    }, 5000);

    return () => clearInterval(interval);
  }, [refreshAllData]);

  /* =========================================================
     WEBSOCKET ALERTS
  ========================================================= */

  useEffect(() => {
    let ws;

    try {
      ws = new WebSocket(`${WEBSOCKET_URL}/ws/alerts`);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const alertData = JSON.parse(event.data);

          setAlerts((previous) => [
            alertData,
            ...previous,
          ].slice(0, 8));
        } catch {
          setAlerts((previous) => [
            {
              violation: event.data,
              time: new Date().toLocaleTimeString(),
            },
            ...previous,
          ].slice(0, 8));
        }
      };

      ws.onerror = () => {
        console.warn('WebSocket connection error');
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
      };
    } catch (error) {
      console.warn('WebSocket failed:', error);
    }

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  /* =========================================================
     TAB CHANGE
  ========================================================= */

  const handleTabChange = (tabName) => {
    setActiveTab(tabName);

    if (tabName !== 'Live Monitoring') {
      setIsStreamActive(false);
    }

    setAnalysisError(null);
  };

  /* =========================================================
     VIDEO VALIDATION
  ========================================================= */

  const validateVideoFile = (file) => {
    if (!file) {
      return false;
    }

    const allowedExtensions = [
      '.mp4',
      '.webm',
      '.mov',
      '.avi',
      '.mkv',
    ];

    const fileName = file.name.toLowerCase();

    const validExtension = allowedExtensions.some(
      (extension) => fileName.endsWith(extension)
    );

    const validMime =
      !file.type ||
      file.type.startsWith('video/');

    return validExtension && validMime;
  };

  /* =========================================================
     SELECT VIDEO
  ========================================================= */

  const selectVideoFile = (file) => {
    console.log('📁 File selected:', file);

    if (!file) {
      return;
    }

    if (!validateVideoFile(file)) {
      console.error('❌ Invalid video file');

      setSelectedFile(null);
      setAnalysisError(
        'Invalid video file. Please select MP4, WebM, MOV, AVI or MKV.'
      );

      return;
    }

    setSelectedFile(file);
    setProcessedVideoUrl('');
    setDetectionStats(null);
    setAnalysisError(null);

    console.log('✅ Video accepted:', file.name);
  };

  /* =========================================================
     FILE INPUT
  ========================================================= */

  const handleFileChange = (event) => {
    event.stopPropagation();

    const file = event.target.files?.[0];

    selectVideoFile(file);
  };

  /* =========================================================
     OPEN FILE PICKER
  ========================================================= */

  const openFilePicker = () => {
    if (isProcessing) {
      return;
    }

    fileInputRef.current?.click();
  };

  /* =========================================================
     DRAG & DROP
  ========================================================= */

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (isProcessing) {
      return;
    }

    const file = event.dataTransfer.files?.[0];

    selectVideoFile(file);
  };

  /* =========================================================
     NORMALIZE URL
  ========================================================= */

  const getVideoUrl = (url) => {
    if (!url) {
      return '';
    }

    if (
      url.startsWith('http://') ||
      url.startsWith('https://')
    ) {
      return url;
    }

    if (url.startsWith('/')) {
      return `${BACKEND_URL}${url}`;
    }

    return `${BACKEND_URL}/${url}`;
  };

  const formatVideoTime = (seconds) => {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  };

  /* =========================================================
     START AI ANALYSIS
  ========================================================= */

  const handleStartAnalysis = async (event) => {
    /*
      VERY IMPORTANT:
      Stop click from reaching upload/drop zone.
    */

    event?.preventDefault();
    event?.stopPropagation();

    console.log('');
    console.log('==========================================');
    console.log('🔥 START AI ANALYSIS');
    console.log('==========================================');

    if (isProcessing) {
      console.warn('⚠️ Analysis already running');
      return;
    }

    if (!selectedFile) {
      console.error('❌ No file selected');

      setAnalysisError(
        'Please select a video file first.'
      );

      return;
    }

    if (!validateVideoFile(selectedFile)) {
      console.error('❌ Invalid video');

      setAnalysisError(
        'Please select a valid video file.'
      );

      return;
    }

    console.log('📁 Filename:', selectedFile.name);
    console.log('📁 Type:', selectedFile.type);
    console.log(
      '📁 Size:',
      `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
    );

    setIsProcessing(true);
    setAnalysisError(null);
    setProcessedVideoUrl('');
    setDetectionStats(null);

    const formData = new FormData();

    /*
      Backend expects exactly:

      file: UploadFile = File(...)
    */

    formData.append('file', selectedFile);

    console.log(
      '🌐 POST:',
      `${BACKEND_URL}/api/analyze_video`
    );

    console.log('📤 Sending request to FastAPI...');

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/analyze_video`,
        {
          method: 'POST',
          body: formData,
        }
      );

      console.log(
        '📥 HTTP STATUS:',
        response.status
      );

      const responseText = await response.text();

      console.log(
        '📥 RAW BACKEND RESPONSE:',
        responseText
      );

      let data = null;

      try {
        data = responseText
          ? JSON.parse(responseText)
          : null;
      } catch (parseError) {
        console.error(
          '❌ JSON parse error:',
          parseError
        );

        throw new Error(
          `Backend returned invalid response. HTTP ${response.status}`
        );
      }

      if (!response.ok) {
        console.error(
          '❌ Backend returned error:',
          data
        );

        throw new Error(
          data?.detail ||
            `Video analysis failed. HTTP ${response.status}`
        );
      }

      console.log('✅ Backend analysis completed');
      console.log('📊 Response:', data);

      if (!data?.video_url) {
        throw new Error(
          'Backend completed the analysis but did not return video_url.'
        );
      }

      const finalVideoUrl = getVideoUrl(
        data.video_url
      );

      console.log(
        '🎬 FINAL VIDEO URL:',
        finalVideoUrl
      );

      setProcessedVideoUrl(finalVideoUrl);

      setDetectionStats({
        active_staff: Number(data.active_staff) || 0,
        gloves_detected:
          Number(data.gloves_detected) || 0,
        mask_detected:
          Number(data.mask_detected) || 0,
        hairnet_detected:
          Number(data.hairnet_detected) || 0,
        no_mask: Number(data.no_mask) || 0,
        no_gloves: Number(data.no_gloves) || 0,
        compliant: Number(data.compliant) || 0,
        total_alerts:
          Number(data.total_alerts) || 0,
        total_checks:
          Number(data.total_checks) || 0,
        compliance_score:
          typeof data.compliance_score === 'number'
            ? data.compliance_score
            : 100,
      });

      // The upload endpoint completes only once clip rows are committed, so
      // refresh immediately instead of waiting for the polling interval.
      await refreshAllData();

      console.log('🎉 VIDEO ANALYSIS SUCCESSFUL');
    } catch (error) {
      console.error('');
      console.error(
        '🔥 VIDEO ANALYSIS ERROR'
      );
      console.error(error);
      console.error('');

      if (
        error instanceof TypeError &&
        error.message.toLowerCase().includes('fetch')
      ) {
        setAnalysisError(
          'Cannot connect to FastAPI. Make sure backend is running at http://127.0.0.1:8000.'
        );
      } else {
        setAnalysisError(
          error?.message ||
            'Unable to analyze video.'
        );
      }
    } finally {
      setIsProcessing(false);

      console.log(
        '🏁 Analysis request finished'
      );
      console.log(
        '=========================================='
      );
    }
  };

  /* =========================================================
     RESET
  ========================================================= */

  const handleResetAnalysis = (event) => {
    event?.preventDefault();
    event?.stopPropagation();

    console.log('🔄 Reset video analysis');

    setSelectedFile(null);
    setProcessedVideoUrl('');
    setDetectionStats(null);
    setAnalysisError(null);
    setIsProcessing(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  /* =========================================================
     CHART DATA
  ========================================================= */

  const barColors = [
    '#10b981',
    '#06b6d4',
    '#8b5cf6',
    '#f43f5e',
    '#f97316',
  ];

  const barChartData = Object.entries(
    analytics.class_counts || {}
  ).filter(([key]) => VALID_HYGIENE_CLASSES.has(normalizeHygieneClass(key))).map(([key, value]) => ({
    name: normalizeHygieneClass(key)
      .toUpperCase(),
    count: Number(value) || 0,
  }));

  const displayBarData =
    barChartData.length > 0
      ? barChartData
      : [
          { name: 'MASK', count: 0 },
          { name: 'GLOVES', count: 0 },
          { name: 'HAIRNET', count: 0 },
          { name: 'NO MASK', count: 0 },
          { name: 'NO GLOVES', count: 0 },
        ];

  /* =========================================================
     PIE DATA
  ========================================================= */

  const compliantCount =
    analytics.status_counts?.compliant || 0;

  const violationCount =
    analytics.status_counts?.violation || 0;

  const isPieEmpty =
    compliantCount === 0 &&
    violationCount === 0;

  const displayPieData = isPieEmpty
    ? [
        {
          name: 'Compliant',
          value: 100,
          color: '#10b981',
        },
      ]
    : [
        {
          name: 'Compliant',
          value: compliantCount,
          color: '#10b981',
        },
        {
          name: 'Violations',
          value: violationCount,
          color: '#ef4444',
        },
      ];

  /* =========================================================
     HIGHLIGHTS
  ========================================================= */

  /* =========================================================
     RENDER
  ========================================================= */

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans overflow-hidden w-full">

      {/* SIDEBAR */}

      <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col justify-between shrink-0">

        <div>

          <div className="h-20 flex items-center px-6 border-b border-slate-800">

            <CheckCircle2 className="w-8 h-8 text-emerald-500 mr-3" />

            <span className="font-bold text-xl text-white tracking-wide">
              Kitchen{' '}
              <span className="text-emerald-500">
                Hygiene
              </span>
            </span>

          </div>

          <nav className="mt-6 px-4 space-y-1">

            {sidebarLinks.map((link) => {
              const Icon = link.icon;

              const isActive =
                activeTab === link.name;

              return (
                <button
                  type="button"
                  key={link.name}
                  onClick={() =>
                    handleTabChange(link.name)
                  }
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/10'
                      : 'hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <Icon
                    className={`w-5 h-5 ${
                      isActive
                        ? 'text-white'
                        : 'text-slate-400'
                    }`}
                  />

                  {link.name}
                </button>
              );
            })}

          </nav>

        </div>

        <div className="p-4 border-t border-slate-800 flex items-center gap-3 bg-slate-950/40">

          <div className="w-10 h-10 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-sm">
            HR
          </div>

          <div className="overflow-hidden">

            <h4 className="text-sm font-semibold text-white truncate">
              Hassan Rasheed
            </h4>

            <p className="text-xs text-slate-500 truncate">
              System Admin
            </p>

          </div>

        </div>

      </aside>

      {/* MAIN */}

      <main className="flex-1 flex flex-col overflow-hidden">

        {/* HEADER */}

        <header className="h-20 bg-white border-b border-gray-200 flex items-center justify-between px-8 shrink-0">

          <div className="flex items-center gap-4">

            <h1 className="text-xl font-bold text-gray-900">
              {activeTab}
            </h1>

            {activeTab === 'Dashboard' && (
              <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">

                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />

                5s Live Sync

              </span>
            )}

          </div>

          <div className="flex items-center gap-4">

            {activeTab === 'Dashboard' && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-200 bg-gray-50 text-xs text-gray-600">

                <Filter className="w-3.5 h-3.5 text-gray-400" />

                <span className="font-medium">
                  Timeframe:
                </span>

                <select
                  value={timeframe}
                  onChange={(event) =>
                    setTimeframe(event.target.value)
                  }
                  className="bg-transparent font-bold text-emerald-600 focus:outline-none cursor-pointer"
                >
                  <option value="30m">
                    Last 30m
                  </option>

                  <option value="1h">
                    Last 1h
                  </option>

                  <option value="24h">
                    Last 24h
                  </option>

                  <option value="7d">
                    Last 7d
                  </option>
                </select>

              </div>
            )}

            <button
              type="button"
              onClick={refreshAllData}
              title="Refresh Data"
              className="p-2 text-gray-400 hover:text-gray-600 bg-gray-50 rounded-lg border border-gray-200"
            >
              <RefreshCw
                className={`w-4 h-4 ${
                  loading
                    ? 'animate-spin text-emerald-600'
                    : ''
                }`}
              />
            </button>

            <div className="relative">

              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />

              <input
                type="text"
                placeholder="Search..."
                className="pl-9 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 w-56 bg-gray-50"
              />

            </div>

          </div>

        </header>

        {/* BODY */}

        <div className="flex-1 overflow-y-auto p-8">

          {/* =================================================
              DASHBOARD
          ================================================= */}

          {activeTab === 'Dashboard' && (
            <div className="space-y-8">

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">

                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">

                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Compliance Score
                  </span>

                  <h3
                    className={`text-3xl font-extrabold mt-2 ${
                      analytics.compliance_score >= 80
                        ? 'text-emerald-600'
                        : 'text-rose-600'
                    }`}
                  >
                    {analytics.compliance_score}%
                  </h3>

                  <div className="mt-4 text-xs py-1 px-2.5 rounded-full w-fit text-emerald-600 bg-emerald-50">
                    {analytics.compliance_score >= 80
                      ? 'High Compliance'
                      : 'Attention Needed'}
                  </div>

                </div>

                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">

                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Total Detections
                  </span>

                  <h3 className="text-3xl font-extrabold text-gray-900 mt-2">
                    {analytics.total_detections}
                  </h3>

                  <p className="mt-4 text-xs text-gray-500">
                    {timeframe} window count
                  </p>

                </div>

                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">

                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Compliant Actions
                  </span>

                  <h3 className="text-3xl font-extrabold text-emerald-600 mt-2">
                    {compliantCount}
                  </h3>

                  <p className="mt-4 text-xs text-gray-500">
                    Verified PPE checks
                  </p>

                </div>

                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">

                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Violations
                  </span>

                  <h3 className="text-3xl font-extrabold text-rose-500 mt-2">
                    {violationCount}
                  </h3>

                  <div className="mt-4 text-xs text-rose-600 bg-rose-50 py-1 px-2.5 rounded-full w-fit">
                    Requires Review
                  </div>

                </div>

              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                <div className="lg:col-span-7 bg-white rounded-xl border border-gray-200 shadow-sm p-6">

                  <div className="border-b border-gray-100 pb-3 mb-4 flex justify-between">

                    <h3 className="font-bold text-base text-gray-900">
                      Hygiene Class Detection Distribution
                    </h3>

                    <span className="text-xs text-gray-400">
                      Detected Counts
                    </span>

                  </div>

                  <div className="w-full h-56">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >

                      <BarChart
                        data={displayBarData}
                        margin={{
                          top: 10,
                          right: 10,
                          left: -20,
                          bottom: 0,
                        }}
                      >

                        <XAxis
                          dataKey="name"
                          tick={{
                            fill: '#64748b',
                            fontSize: 11,
                          }}
                          stroke="#e2e8f0"
                        />

                        <YAxis
                          tick={{
                            fill: '#64748b',
                            fontSize: 11,
                          }}
                          stroke="#e2e8f0"
                          allowDecimals={false}
                        />

                        <Tooltip />

                        <Bar
                          dataKey="count"
                          radius={[4, 4, 0, 0]}
                        >

                          {displayBarData.map(
                            (_, index) => (
                              <Cell
                                key={index}
                                fill={
                                  barColors[
                                    index %
                                      barColors.length
                                  ]
                                }
                              />
                            )
                          )}

                        </Bar>

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                </div>

                <div className="lg:col-span-5 bg-white rounded-xl border border-gray-200 shadow-sm p-6">

                  <div className="border-b border-gray-100 pb-3 mb-4 flex justify-between">

                    <h3 className="font-bold text-base">
                      Compliance Ratio
                    </h3>

                    <span className="text-xs text-gray-400">
                      {timeframe}
                    </span>

                  </div>

                  <div className="flex items-center justify-center gap-8 h-56">

                    <div className="w-40 h-40 relative">

                      <ResponsiveContainer
                        width="100%"
                        height="100%"
                      >

                        <PieChart>

                          <Pie
                            data={displayPieData}
                            cx="50%"
                            cy="50%"
                            innerRadius={45}
                            outerRadius={65}
                            paddingAngle={4}
                            dataKey="value"
                          >

                            {displayPieData.map(
                              (entry, index) => (
                                <Cell
                                  key={index}
                                  fill={entry.color}
                                />
                              )
                            )}

                          </Pie>

                          <Tooltip />

                        </PieChart>

                      </ResponsiveContainer>

                      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">

                        <span className="text-xl font-extrabold text-gray-900">
                          {analytics.compliance_score}%
                        </span>

                        <span className="text-[10px] uppercase text-gray-400">
                          Score
                        </span>

                      </div>

                    </div>

                    <div className="space-y-4">

                      <div className="flex items-center gap-3">

                        <span className="w-3 h-3 rounded-full bg-emerald-500" />

                        <span className="text-sm font-semibold">
                          Compliant
                        </span>

                        <span className="font-bold">
                          {compliantCount}
                        </span>

                      </div>

                      <div className="flex items-center gap-3">

                        <span className="w-3 h-3 rounded-full bg-rose-500" />

                        <span className="text-sm font-semibold">
                          Violations
                        </span>

                        <span className="font-bold">
                          {violationCount}
                        </span>

                      </div>

                    </div>

                  </div>

                </div>

              </div>

              <div className="space-y-4">

                <div className="flex items-center justify-between">

                  <div>

                    <h2 className="text-lg font-bold text-gray-900">
                      Incident Highlights
                    </h2>

                    <p className="text-xs text-gray-400">
                      Latest captured violation clips
                    </p>

                  </div>

                  <span className="text-xs text-gray-400">
                    {lastSynced
                      ? `Updated ${lastSynced}`
                      : 'Syncing...'}
                  </span>

                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

                  {highlights.map(
                    (clip, index) => {

                      const videoSrc =
                        clip.clip_url
                          ? getVideoUrl(
                              clip.clip_url
                            )
                          : null;

                      return (
                        <div
                          key={
                            clip.id ||
                            `clip-${index}`
                          }
                          className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
                        >

                          <div className="aspect-video bg-slate-900 relative flex items-center justify-center">

                            {videoSrc ? (
                              <video
                                src={videoSrc}
                                controls
                                preload="metadata"
                                className="w-full h-full object-cover"
                                onError={() => {
                                  console.error('Unable to play highlight:', videoSrc);
                                }}
                              />
                            ) : (
                              <div className="text-center">

                                <Video className="w-10 h-10 text-slate-600 mx-auto mb-2" />

                                <p className="text-xs font-semibold text-slate-300">
                                  Cam Slot {index + 1}
                                </p>

                                <p className="text-[10px] text-slate-500">
                                  Awaiting Clip
                                </p>

                              </div>
                            )}

                            <span className="absolute top-3 left-3 text-[10px] font-bold bg-slate-900/90 text-white px-2 py-1 rounded">
                              CAM {index + 1}
                            </span>

                          </div>

                          <div className="p-4 space-y-3">

                            <div className="flex justify-between items-center">

                              <div className="flex items-center gap-2 text-xs font-bold">

                                <User className="w-3.5 h-3.5 text-emerald-600" />

                                {clip.person_id}

                              </div>

                              <span className="text-[11px] font-semibold text-red-700 bg-red-50 border border-red-200 px-2 py-1 rounded">
                                {clip.reason}
                              </span>

                            </div>

                            <p className="text-sm font-semibold text-gray-900">
                              {clip.title || `${clip.class_name || 'Hygiene'} Detected`}
                            </p>

                            <div className="border-t pt-2 flex justify-between text-xs text-gray-400">

                              <span className="flex items-center gap-1">

                                <Clock className="w-3 h-3" />

                                {formatVideoTime(clip.start_seconds)}

                              </span>

                              <span>
                                {`${formatVideoTime(clip.start_seconds)}–${formatVideoTime(clip.end_seconds)}`}
                              </span>

                            </div>

                          </div>

                        </div>
                      );
                    }
                  )}

                  {highlights.length === 0 && (
                    <div className="col-span-full bg-white rounded-xl border border-gray-200 p-10 text-center text-sm text-gray-500">
                      No violation highlights have been generated yet.
                    </div>
                  )}

                </div>

              </div>

            </div>
          )}

          {/* =================================================
              LIVE MONITORING
          ================================================= */}

          {activeTab === 'Live Monitoring' && (
            <div className="space-y-8 max-w-4xl mx-auto">

              <div className="flex justify-between items-center border-b pb-3">

                <div className="flex items-center gap-2">

                  <Camera className="w-6 h-6 text-emerald-600" />

                  <h2 className="text-xl font-bold">
                    Live AI Camera Feed
                  </h2>

                </div>

                <div className="flex items-center gap-2">

                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      isStreamActive
                        ? 'bg-emerald-500 animate-pulse'
                        : 'bg-gray-400'
                    }`}
                  />

                  <span className="text-xs font-semibold">
                    {isStreamActive
                      ? 'Live Stream Active'
                      : 'Stream Offline'}
                  </span>

                </div>

              </div>

              <div className="bg-white rounded-xl border shadow-sm p-6 space-y-6">

                <div className="aspect-video bg-black rounded-lg overflow-hidden relative">

                  {isStreamActive ? (
                    <img
                      src={`${BACKEND_URL}/api/live_feed`}
                      alt="Live AI Camera Feed"
                      className="w-full h-full object-contain"
                      onError={() => {
                        setStreamError(true);
                        setIsStreamActive(false);
                      }}
                    />
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">

                      <Camera className="w-16 h-16 mb-4" />

                      <p className="font-semibold">
                        Camera stream is stopped.
                      </p>

                    </div>
                  )}

                  {isStreamActive && (
                    <div className="absolute bottom-3 left-3 bg-slate-900/80 px-3 py-1.5 rounded text-xs text-white flex items-center gap-2">

                      <Radio className="w-3 h-3 text-rose-500 animate-pulse" />

                      Live AI Detection

                    </div>
                  )}

                </div>

                <div className="flex gap-4 justify-center">

                  <button
                    type="button"
                    onClick={() => {
                      setStreamError(false);
                      setIsStreamActive(true);
                    }}
                    disabled={isStreamActive}
                    className="px-6 py-2.5 rounded-lg bg-emerald-600 text-white font-semibold disabled:bg-gray-200 disabled:text-gray-400"
                  >
                    Start Camera Stream
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setIsStreamActive(false)
                    }
                    disabled={!isStreamActive}
                    className="px-6 py-2.5 rounded-lg bg-rose-600 text-white font-semibold disabled:bg-gray-200 disabled:text-gray-400"
                  >
                    Stop Camera Stream
                  </button>

                </div>

                {streamError && (
                  <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg">
                    Unable to connect to camera stream.
                  </div>
                )}

              </div>

            </div>
          )}

          {/* =================================================
              VIDEO ANALYSIS
          ================================================= */}

          {activeTab === 'Video Analysis' && (
            <div className="space-y-8 max-w-4xl mx-auto w-full">

              {/* PROCESSING */}

              {isProcessing && (
                <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-12 text-center min-h-[400px] flex flex-col items-center justify-center">

                  <div className="relative w-20 h-20 mb-6">

                    <div className="absolute inset-0 rounded-full border-4 border-gray-200" />

                    <div className="absolute inset-0 rounded-full border-4 border-emerald-500 border-t-transparent animate-spin" />

                  </div>

                  <h3 className="text-xl font-bold text-gray-900">
                    Processing Video
                  </h3>

                  <p className="text-sm text-gray-500 mt-2">
                    YOLOv8 AI is analyzing the video frame-by-frame.
                  </p>

                  <p className="text-xs text-gray-400 mt-4">
                    {selectedFile?.name}
                  </p>

                  <p className="text-xs text-emerald-600 mt-3 font-semibold">
                    Please wait. Large videos may take some time.
                  </p>

                </div>
              )}

              {/* RESULT */}

              {!isProcessing &&
                processedVideoUrl && (
                  <div className="bg-white rounded-xl border shadow-sm p-6 space-y-6">

                    <div className="flex justify-between items-center border-b pb-4">

                      <div>

                        <h3 className="font-bold text-lg">
                          Analysis Result
                        </h3>

                        <p className="text-xs text-gray-400 mt-1">
                          AI processed video
                        </p>

                      </div>

                      <button
                        type="button"
                        onClick={handleResetAnalysis}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm"
                      >
                        Process Another Video
                      </button>

                    </div>

                    <div className="aspect-video bg-black rounded-lg overflow-hidden">

                      <video
                        src={processedVideoUrl}
                        controls
                        className="w-full h-full object-contain"
                        onError={() => {
                          setAnalysisError(
                            'Processed video could not be loaded. Check the FastAPI static folder.'
                          );
                        }}
                      />

                    </div>

                    {analysisError && (
                      <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
                        {analysisError}
                      </div>
                    )}

                    {detectionStats && (
                      <div className="space-y-5">

                        <h4 className="font-bold text-sm text-gray-500 uppercase">
                          AI Detection Metrics
                        </h4>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

                          <div className="bg-gray-50 p-4 rounded-lg border">

                            <div className="text-xs text-gray-400 font-semibold uppercase">
                              Compliance
                            </div>

                            <div
                              className={`text-2xl font-extrabold mt-1 ${
                                detectionStats.compliance_score >= 80
                                  ? 'text-emerald-600'
                                  : 'text-rose-600'
                              }`}
                            >
                              {detectionStats.compliance_score}%
                            </div>

                          </div>

                          <div className="bg-gray-50 p-4 rounded-lg border">

                            <div className="text-xs text-gray-400 font-semibold uppercase">
                              Staff Detected
                            </div>

                            <div className="text-2xl font-extrabold mt-1">
                              {detectionStats.active_staff}
                            </div>

                          </div>

                          <div className="bg-gray-50 p-4 rounded-lg border">

                            <div className="text-xs text-gray-400 font-semibold uppercase">
                              Compliant Checks
                            </div>

                            <div className="text-2xl font-extrabold text-emerald-600 mt-1">
                              {detectionStats.compliant}
                            </div>

                          </div>

                          <div className="bg-gray-50 p-4 rounded-lg border">

                            <div className="text-xs text-gray-400 font-semibold uppercase">
                              Total Alerts
                            </div>

                            <div className="text-2xl font-extrabold text-rose-500 mt-1">
                              {detectionStats.total_alerts}
                            </div>

                          </div>

                        </div>

                        <div className="bg-gray-50 p-6 rounded-lg border space-y-4">

                          <h5 className="text-xs font-bold text-gray-500 uppercase border-b pb-2">
                            PPE Validation Summary
                          </h5>

                          <div className="flex justify-between text-sm">

                            <span className="font-medium">
                              Masks Detected
                            </span>

                            <span className="font-bold text-emerald-600">

                              {detectionStats.mask_detected}

                              <span className="text-gray-400 font-normal">
                                {' '}
                                (
                                {detectionStats.no_mask}{' '}
                                violations)
                              </span>

                            </span>

                          </div>

                          <div className="flex justify-between text-sm">

                            <span className="font-medium">
                              Gloves Detected
                            </span>

                            <span className="font-bold text-emerald-600">

                              {detectionStats.gloves_detected}

                              <span className="text-gray-400 font-normal">
                                {' '}
                                (
                                {detectionStats.no_gloves}{' '}
                                violations)
                              </span>

                            </span>

                          </div>

                          <div className="flex justify-between text-sm">

                            <span className="font-medium">
                              Hairnets Detected
                            </span>

                            <span className="font-bold text-sky-600">
                              {detectionStats.hairnet_detected}
                            </span>

                          </div>

                        </div>

                      </div>
                    )}

                  </div>
                )}

              {/* UPLOAD */}

              {!isProcessing &&
                !processedVideoUrl && (
                  <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8">

                    {/* 
                      IMPORTANT:
                      Upload area itself opens file picker.
                      Action buttons are outside this clickable area.
                    */}

                    <div
                      onClick={openFilePicker}
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      className={`border-2 border-dashed rounded-xl p-12 transition-all cursor-pointer flex flex-col items-center justify-center text-center ${
                        selectedFile
                          ? 'border-emerald-500 bg-emerald-50/30'
                          : 'border-gray-300 hover:border-emerald-500 hover:bg-slate-50'
                      }`}
                    >

                      <Upload
                        className={`w-14 h-14 mb-5 ${
                          selectedFile
                            ? 'text-emerald-500'
                            : 'text-gray-400'
                        }`}
                      />

                      {selectedFile ? (
                        <>
                          <p className="text-emerald-600 font-bold text-lg">
                            {selectedFile.name}
                          </p>

                          <p className="text-xs text-gray-400 mt-2">
                            {(
                              selectedFile.size /
                              (1024 * 1024)
                            ).toFixed(2)}{' '}
                            MB
                          </p>

                          <p className="text-xs text-gray-400 mt-2">
                            Click to choose another video
                          </p>
                        </>
                      ) : (
                        <>
                          <p className="text-gray-700 font-semibold text-lg">
                            Drag & Drop Video Here
                          </p>

                          <p className="text-gray-400 text-sm mt-2">
                            or click to browse
                          </p>

                          <p className="text-xs text-gray-400 mt-3">
                            MP4, WebM, MOV, AVI, MKV
                          </p>
                        </>
                      )}

                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".mp4,.webm,.mov,.avi,.mkv,video/*"
                        onChange={handleFileChange}
                        className="hidden"
                        onClick={(event) =>
                          event.stopPropagation()
                        }
                      />

                    </div>

                    {/* ERROR */}

                    {analysisError && (
                      <div className="mt-5 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-start gap-3">

                        <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />

                        <div className="flex-1">

                          <p className="font-semibold text-sm">
                            Video Analysis Error
                          </p>

                          <p className="text-sm mt-1">
                            {analysisError}
                          </p>

                        </div>

                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setAnalysisError(null);
                          }}
                          className="text-red-400 hover:text-red-700"
                        >
                          <X className="w-4 h-4" />
                        </button>

                      </div>
                    )}

                    {/* ACTIONS */}

                    {selectedFile && (
                      <div
                        className="mt-6 flex items-center justify-center gap-4"
                        onClick={(event) =>
                          event.stopPropagation()
                        }
                      >

                        <button
                          type="button"
                          onClick={handleResetAnalysis}
                          disabled={isProcessing}
                          className="px-6 py-3 rounded-lg border border-gray-300 text-gray-600 font-semibold hover:bg-gray-50 disabled:opacity-50"
                        >
                          Remove
                        </button>

                        <button
                          type="button"
                          onClick={handleStartAnalysis}
                          disabled={
                            isProcessing ||
                            !selectedFile
                          }
                          className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow-lg shadow-emerald-600/10 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
                        >

                          {isProcessing ? (
                            <>
                              <RefreshCw className="w-5 h-5 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <CheckCircle2 className="w-5 h-5" />
                              Start AI Analysis
                            </>
                          )}

                        </button>

                      </div>
                    )}

                  </div>
                )}

            </div>
          )}

          {/* =================================================
              REPORTS
          ================================================= */}

          {activeTab === 'Reports' && (
            <div className="bg-white rounded-xl border shadow-sm p-12 text-center min-h-[400px] flex flex-col items-center justify-center">

              <BarChart3 className="w-16 h-16 text-emerald-500 mb-4" />

              <h3 className="text-xl font-bold">
                Reports
              </h3>

              <p className="text-sm text-gray-500 mt-2">
                Reporting dashboard is connected to the backend.
              </p>

            </div>
          )}

          {/* =================================================
              SETTINGS
          ================================================= */}

          {activeTab === 'Settings' && (
            <div className="bg-white rounded-xl border shadow-sm p-12 text-center min-h-[400px] flex flex-col items-center justify-center">

              <Settings className="w-16 h-16 text-emerald-500 mb-4" />

              <h3 className="text-xl font-bold">
                Settings
              </h3>

              <p className="text-sm text-gray-500 mt-2">
                System configuration panel.
              </p>

            </div>
          )}

        </div>

      </main>

    </div>
  );
};

export default Dashboard;
