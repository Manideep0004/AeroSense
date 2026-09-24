import React, { useState } from 'react';
import { Activity, CloudRain, Thermometer, Wind, RefreshCw, AlertCircle, UploadCloud } from 'lucide-react';

export default function InferenceTab() {
  const [season, setSeason] = useState('Winter');
  const [formData, setFormData] = useState({
    pm25: 150.0,
    temperature: 15.0,
    humidity: 60.0,
    wind_speed: 2.0,
    wind_direction: 180.0,
    pressure: 1010.0,
    precipitation: 0.0,
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    
    // Auto-fill required lag/temporal features with dummy defaults for UI
    const payloadFeatures = {
      ...formData,
      pm25_lag_1: formData.pm25 * 0.95,
      pm25_lag_3: formData.pm25 * 0.90,
      pm25_lag_6: formData.pm25 * 0.85,
      pm25_lag_12: formData.pm25 * 0.80,
      pm25_lag_24: formData.pm25 * 0.75,
      pm25_mean_3h: formData.pm25,
      pm25_mean_6h: formData.pm25,
      pm25_mean_12h: formData.pm25,
      pm25_mean_24h: formData.pm25,
      wind_x: formData.wind_speed * 0.5,
      wind_y: formData.wind_speed * 0.5,
      hour_sin: 0.5, hour_cos: 0.5,
      day_of_week_sin: 0.5, day_of_week_cos: 0.5,
      month_sin: 0.5, month_cos: 0.5,
      day_of_year_sin: 0.5, day_of_year_cos: 0.5
    };
    
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ season, features: payloadFeatures })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getAQIColor = (val) => {
    if (val <= 50) return 'text-emerald-600 border-emerald-200 bg-emerald-50 dark:text-emerald-400 dark:border-emerald-400/30 dark:bg-emerald-400/10';
    if (val <= 100) return 'text-yellow-600 border-yellow-200 bg-yellow-50 dark:text-yellow-400 dark:border-yellow-400/30 dark:bg-yellow-400/10';
    if (val <= 150) return 'text-orange-600 border-orange-200 bg-orange-50 dark:text-orange-400 dark:border-orange-400/30 dark:bg-orange-400/10';
    if (val <= 200) return 'text-rose-600 border-rose-200 bg-rose-50 dark:text-rose-400 dark:border-rose-400/30 dark:bg-rose-400/10';
    return 'text-purple-600 border-purple-200 bg-purple-50 dark:text-purple-400 dark:border-purple-400/30 dark:bg-purple-400/10';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-display font-semibold text-slate-900 dark:text-white mb-2">Real-time Inference Engine</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">Query the active seasonal ML models to forecast PM2.5 concentrations.</p>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded border border-slate-200 dark:border-slate-700 transition flex items-center gap-2">
            <UploadCloud size={14} /> Batch CSV
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Column */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm dark:shadow-xl transition-colors duration-200">
          <form onSubmit={handlePredict} className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400 tracking-wider">Target Season</label>
                <select 
                  className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white text-sm rounded-lg focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500 p-2.5 transition-colors"
                  value={season} onChange={e => setSeason(e.target.value)}
                >
                  {['Winter', 'Spring', 'Summer', 'Autumn'].map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400 tracking-wider">Current PM2.5 (µg/m³)</label>
                <input 
                  type="number" step="0.1" required
                  className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white text-sm rounded-lg focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500 p-2.5 transition-colors"
                  value={formData.pm25} onChange={e => setFormData({...formData, pm25: parseFloat(e.target.value)})}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4 bg-slate-50 dark:bg-slate-950/50 rounded-lg border border-slate-200 dark:border-slate-800/50 transition-colors">
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 flex items-center gap-1.5"><Thermometer size={14}/> Temp (°C)</label>
                <input type="number" step="0.1" required className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white text-sm rounded p-2 transition-colors" value={formData.temperature} onChange={e => setFormData({...formData, temperature: parseFloat(e.target.value)})} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 flex items-center gap-1.5"><CloudRain size={14}/> Humidity (%)</label>
                <input type="number" step="0.1" required className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white text-sm rounded p-2 transition-colors" value={formData.humidity} onChange={e => setFormData({...formData, humidity: parseFloat(e.target.value)})} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 flex items-center gap-1.5"><Wind size={14}/> Speed (m/s)</label>
                <input type="number" step="0.1" required className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-white text-sm rounded p-2 transition-colors" value={formData.wind_speed} onChange={e => setFormData({...formData, wind_speed: parseFloat(e.target.value)})} />
              </div>
            </div>

            <button 
              type="submit" disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-sm dark:shadow-[0_0_20px_rgba(6,182,212,0.2)] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <RefreshCw className="animate-spin" size={18} /> : <Activity size={18} />}
              Execute Inference
            </button>
          </form>
        </div>

        {/* Results Column */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm dark:shadow-xl flex flex-col justify-center transition-colors duration-200">
          {error && (
            <div className="bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/20 text-rose-600 dark:text-rose-400 p-4 rounded-lg flex items-start gap-3">
              <AlertCircle size={18} className="mt-0.5 flex-shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {!error && !result && !loading && (
            <div className="text-center text-slate-400 dark:text-slate-500 space-y-3">
              <Activity size={32} className="mx-auto opacity-20" />
              <p className="text-sm">Enter telemetry parameters and execute inference to view results.</p>
            </div>
          )}

          {loading && (
            <div className="text-center text-cyan-500 space-y-4">
              <RefreshCw size={32} className="mx-auto animate-spin opacity-50" />
              <p className="text-sm font-mono tracking-widest uppercase animate-pulse">Computing Inference...</p>
            </div>
          )}

          {result && !loading && (
            <div className={`p-6 rounded-xl border ${getAQIColor(result.prediction)} text-center space-y-4 animate-in zoom-in-95 duration-300`}>
              <h3 className="text-sm font-semibold uppercase tracking-widest opacity-80">Predicted PM2.5</h3>
              <div className="text-6xl font-display font-bold font-mono">
                {result.prediction.toFixed(1)}
              </div>
              <p className="text-xs font-mono opacity-60">µg/m³</p>
              
              <div className="pt-4 mt-4 border-t border-current/20 flex justify-between items-center text-xs text-left">
                <div>
                  <p className="opacity-60 mb-1">Model Engine</p>
                  <p className="font-mono font-medium">{result.model_used}</p>
                </div>
                <div className="text-right">
                  <p className="opacity-60 mb-1">Season</p>
                  <p className="font-mono font-medium">{result.season}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
