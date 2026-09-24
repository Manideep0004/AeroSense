import React from 'react';
import { ShieldCheck, AlertOctagon, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function DriftTab({ health }) {
  const driftStatus = health?.drift_status || {};
  
  const getSeverityIcon = (severity) => {
    switch(severity) {
      case 'CRITICAL': return <AlertOctagon className="text-rose-500 dark:text-rose-400" size={24} />;
      case 'WARNING': return <AlertTriangle className="text-amber-500 dark:text-amber-400" size={24} />;
      default: return <CheckCircle2 className="text-emerald-500 dark:text-emerald-400" size={24} />;
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'CRITICAL': return 'bg-rose-50 border-rose-200 text-rose-600 dark:bg-rose-500/10 dark:border-rose-500/20 dark:text-rose-400';
      case 'WARNING': return 'bg-amber-50 border-amber-200 text-amber-600 dark:bg-amber-500/10 dark:border-amber-500/20 dark:text-amber-400';
      default: return 'bg-emerald-50 border-emerald-200 text-emerald-600 dark:bg-emerald-500/10 dark:border-emerald-500/20 dark:text-emerald-400';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-display font-semibold text-slate-900 dark:text-white mb-2">Drift Telemetry</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">Real-time background monitoring using KS-Test and PSI on live inference batches.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {['Winter', 'Spring', 'Summer', 'Autumn'].map((season) => {
          const status = driftStatus[season] || { latest_severity: 'NONE', latest_drift_fraction: 0.0, total_reports: 0 };
          const fraction = (status.latest_drift_fraction * 100).toFixed(1);
          
          return (
            <div key={season} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm relative overflow-hidden group transition-colors duration-200">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-white">{season}</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-1">v{new Date().getFullYear()}</p>
                </div>
                {getSeverityIcon(status.latest_severity)}
              </div>
              
              <div className="space-y-1 mt-6">
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-500 dark:text-slate-400">Drift Fraction</span>
                  <span className="font-mono text-slate-900 dark:text-white">{fraction}%</span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-950 rounded-full h-1.5 border border-slate-200 dark:border-slate-800">
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ${
                      status.latest_severity === 'CRITICAL' ? 'bg-rose-500' : 
                      status.latest_severity === 'WARNING' ? 'bg-amber-500' : 'bg-emerald-500'
                    }`} 
                    style={{ width: `${Math.min(100, Math.max(0, fraction))}%` }}
                  ></div>
                </div>
              </div>
              
              <div className={`mt-4 inline-flex items-center px-2 py-1 rounded text-[10px] font-mono font-bold border ${getSeverityColor(status.latest_severity)}`}>
                STATUS: {status.latest_severity}
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-sm dark:shadow-xl overflow-hidden transition-colors duration-200">
        <div className="px-6 py-5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50 dark:bg-slate-900/50">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck size={16} className="text-cyan-600 dark:text-cyan-400" /> Recent Audit Trail
          </h3>
        </div>
        <div className="p-8 text-center text-slate-500 dark:text-slate-500 text-sm">
          <p>Real-time audit log streaming will appear here once inference batches trigger drift thresholds.</p>
          <p className="text-xs mt-2 opacity-60">(Check logs/drift_alerts/ on the server for physical JSONL records)</p>
        </div>
      </div>
    </div>
  );
}
