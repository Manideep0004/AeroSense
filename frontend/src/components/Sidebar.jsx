import React from 'react';
import { Activity, Server, ShieldCheck, Settings2, Wind, Sun, Moon } from 'lucide-react';

export default function Sidebar({ health, activeTab, setActiveTab, isDarkMode, toggleDarkMode }) {
  const isHealthy = health?.status === 'ok';

  return (
    <div className="w-72 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col h-screen overflow-y-auto transition-colors duration-200">
      <div className="p-6 flex justify-between items-center">
        <div>
          <div className="flex items-center gap-3 text-cyan-600 dark:text-cyan-400 mb-2">
            <Wind size={28} />
            <h1 className="text-xl font-display font-bold tracking-tight text-slate-900 dark:text-white">AaroSense</h1>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tracking-wider">MLOPS FORECASTING</p>
        </div>
        <button 
          onClick={toggleDarkMode}
          className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          title="Toggle Theme"
        >
          {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>

      <div className="px-4 pb-6 border-b border-slate-200 dark:border-slate-800">
        <div className="bg-slate-50 dark:bg-slate-950 p-4 rounded-xl border border-slate-200 dark:border-slate-800/60 shadow-inner">
          <div className="flex items-center gap-2 mb-3 text-slate-600 dark:text-slate-300">
            <Server size={16} className={isHealthy ? "text-emerald-500 dark:text-emerald-400" : "text-rose-500 dark:text-rose-400"} />
            <h3 className="text-sm font-semibold uppercase tracking-wider">API Status</h3>
          </div>
          {health ? (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isHealthy ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
                  <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isHealthy ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
                </span>
                <span className="text-sm text-slate-900 dark:text-white font-medium">{isHealthy ? 'Online & Ready' : 'Degraded'}</span>
              </div>
              <div className="mt-4 space-y-2">
                <h4 className="text-xs text-slate-500 font-semibold uppercase">Active Models</h4>
                {health.active_models?.length > 0 ? (
                  health.active_models.map((m) => (
                    <div key={m.season} className="flex justify-between items-center text-xs bg-white dark:bg-slate-900 px-2 py-1.5 rounded border border-slate-200 dark:border-slate-800">
                      <span className="text-slate-700 dark:text-slate-300 font-medium">{m.season}</span>
                      <span className="text-cyan-600 dark:text-cyan-400 font-mono">{m.model_name}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 italic">No models loaded.</p>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-slate-500 text-sm">
              <Activity className="animate-spin" size={16} />
              <span>Connecting...</span>
            </div>
          )}
        </div>
      </div>

      <div className="p-4 space-y-2 flex-grow">
        <button
          onClick={() => setActiveTab('inference')}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all ${activeTab === 'inference' ? 'bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.05)] dark:shadow-[0_0_15px_rgba(6,182,212,0.15)]' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'}`}
        >
          <Settings2 size={18} />
          Real-time Inference
        </button>

        <button
          onClick={() => setActiveTab('drift')}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all ${activeTab === 'drift' ? 'bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.05)] dark:shadow-[0_0_15px_rgba(6,182,212,0.15)]' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'}`}
        >
          <ShieldCheck size={18} />
          Drift Monitoring
        </button>
      </div>

      <div className="p-4 border-t border-slate-200 dark:border-slate-800 text-center">
        <p className="text-[10px] text-slate-500 dark:text-slate-600 font-mono">v1.0.0-PROD</p>
      </div>
    </div>
  );
}
