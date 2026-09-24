import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import InferenceTab from './components/InferenceTab';
import DriftTab from './components/DriftTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('inference');
  const [health, setHealth] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Handle dark mode toggle
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Handle health polling
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        } else {
          setHealth({ status: 'error' });
        }
      } catch (err) {
        setHealth({ status: 'down' });
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-[#090d16] text-slate-900 dark:text-slate-300 font-sans selection:bg-cyan-500/30 transition-colors duration-200">
      <Sidebar 
        health={health} 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isDarkMode={isDarkMode}
        toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
      />
      <main className="flex-1 h-screen overflow-y-auto p-8 relative">
        {/* Background glow effects - subtle in light mode, prominent in dark */}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-500/10 dark:bg-cyan-500/5 blur-[120px] rounded-full pointer-events-none"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/10 dark:bg-purple-500/5 blur-[120px] rounded-full pointer-events-none"></div>
        
        <div className="relative z-10">
          {activeTab === 'inference' && <InferenceTab />}
          {activeTab === 'drift' && <DriftTab health={health} />}
        </div>
      </main>
    </div>
  );
}
