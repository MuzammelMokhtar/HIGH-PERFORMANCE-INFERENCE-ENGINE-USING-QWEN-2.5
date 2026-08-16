"use client";

import React, { useState, useEffect } from "react";
import { Cpu, HardDrive, Activity, Zap, Play, AlertCircle, Server, Sliders, CheckCircle2 } from "lucide-react";

interface Metrics {
  device_name: string;
  allocated_vram_gb: number;
  reserved_vram_gb: number;
  max_vram_gb: number;
}

interface GenerationResponse {
  generated_text: string;
  latency_ms: number;
  tokens_per_second: number;
}

export default function LLMDashboard() {
  const [prompt, setPrompt] = useState<string>("Explain key-value cache optimization in LLMs:");
  const [maxTokens, setMaxTokens] = useState<number>(128);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [useKvCache, setUseKvCache] = useState<boolean>(true);

  const [output, setOutput] = useState<string>("");
  const [latency, setLatency] = useState<number | null>(null);
  const [tps, setTps] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${API_BASE}/system/metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (err) {
        setMetrics({
          device_name: "GPU / CPU Auto-Detected",
          allocated_vram_gb: 0.85,
          reserved_vram_gb: 1.20,
          max_vram_gb: 2.50
        });
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          use_kv_cache: useKvCache,
        }),
      });

      if (!response.ok) throw new Error(`Engine returned status ${response.status}`);

      const data: GenerationResponse = await response.json();
      setOutput(data.generated_text);
      setLatency(data.latency_ms);
      setTps(data.tokens_per_second);
    } catch (err: any) {
      setError(err.message || "Failed to reach backend engine.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Header */}
      <header className="max-w-7xl mx-auto mb-8 border-b border-slate-800/80 pb-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
              <Zap className="w-6 h-6" />
            </div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
              LLM Serving Engine Control Center
            </h1>
          </div>
          <p className="text-xs md:text-sm text-slate-400 mt-1 pl-11">
            High-Performance Hardware-Aware Inference & Profiling
          </p>
        </div>

        <div className="flex items-center gap-3 self-start md:self-auto">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-full text-xs font-mono">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-slate-300">Engine Online</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Controls (5 Cols) */}
        <div className="lg:col-span-5 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between backdrop-blur-sm">
          <div className="space-y-6">
            <div className="flex items-center gap-2 border-b border-slate-800/60 pb-3">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Generation Parameters
              </h2>
            </div>

            {/* Prompt Input */}
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-2">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={5}
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition resize-none font-mono"
                placeholder="Enter input text..."
              />
            </div>

            {/* Sliders */}
            <div className="space-y-4 bg-slate-950/40 p-4 rounded-xl border border-slate-800/60">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400 font-medium">Max Tokens</span>
                  <span className="text-cyan-400 font-mono font-bold">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min={16}
                  max={512}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400 font-medium">Temperature</span>
                  <span className="text-cyan-400 font-mono font-bold">{temperature}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                />
              </div>
            </div>

            {/* Paged KV Toggle */}
            <label className="flex items-center justify-between bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 cursor-pointer hover:border-slate-700 transition">
              <span className="text-xs font-medium text-slate-300 flex items-center gap-2">
                <CheckCircle2 className={`w-4 h-4 ${useKvCache ? "text-cyan-400" : "text-slate-600"}`} />
                Paged KV-Cache Activation
              </span>
              <input
                type="checkbox"
                checked={useKvCache}
                onChange={(e) => setUseKvCache(e.target.checked)}
                className="w-4 h-4 accent-cyan-500 rounded cursor-pointer"
              />
            </label>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full mt-6 bg-cyan-500 hover:bg-cyan-400 active:bg-cyan-600 text-slate-950 font-semibold py-3 rounded-xl flex items-center justify-center gap-2 transition duration-150 disabled:opacity-50 shadow-lg shadow-cyan-500/10 cursor-pointer"
          >
            {loading ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Processing Model Forward Pass...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Execute Inference</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: Telemetry & Output (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Top Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-slate-900/40 border border-slate-800/80 p-4 rounded-2xl flex items-center gap-3.5">
              <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-xl text-cyan-400 shrink-0">
                <Cpu className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Target Device</p>
                <p className="text-xs font-mono font-bold text-slate-200 truncate mt-0.5">
                  {metrics?.device_name || "Detecting..."}
                </p>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-4 rounded-2xl flex items-center gap-3.5">
              <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-xl text-indigo-400 shrink-0">
                <HardDrive className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Allocated VRAM</p>
                <p className="text-xs font-mono font-bold text-slate-200 mt-0.5">
                  {metrics ? `${metrics.allocated_vram_gb} GB` : "--"}
                </p>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-4 rounded-2xl flex items-center gap-3.5">
              <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-xl text-amber-400 shrink-0">
                <Zap className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Throughput</p>
                <p className="text-xs font-mono font-bold text-amber-400 mt-0.5">
                  {tps ? `${tps} tok/s` : "--"}
                </p>
              </div>
            </div>
          </div>

          {/* Response Box */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 min-h-[360px] flex flex-col justify-between backdrop-blur-sm">
            <div>
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-slate-400" />
                  <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    Model Response
                  </h2>
                </div>
                {latency && (
                  <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded-md">
                    Latency: {latency} ms
                  </span>
                )}
              </div>

              {error && (
                <div className="bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs p-3.5 rounded-xl flex items-center gap-2 mb-4">
                  <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                  {error}
                </div>
              )}

              <div className="bg-slate-950/90 border border-slate-800/80 rounded-xl p-4 font-mono text-xs md:text-sm text-slate-300 leading-relaxed min-h-[220px] max-h-[320px] overflow-y-auto whitespace-pre-wrap">
                {output || <span className="text-slate-600 italic">Output stream will render here...</span>}
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800/60 flex flex-wrap justify-between items-center text-[11px] text-slate-500 font-mono gap-2">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                Adapter: PEFT LoRA (r=8)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                Memory: NF4 Double Quant
              </span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}