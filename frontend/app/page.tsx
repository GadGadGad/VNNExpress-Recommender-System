"use client";

import React, { useEffect, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { NewsCard } from './components/NewsCard';
import { EmbeddingPlot } from './components/EmbeddingPlot';
import api, { endpoints } from '@/lib/api';
import { Search, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { ThemeToggle } from './components/ThemeToggle';

export default function Home() {
  const [loading, setLoading] = useState(true);
  const [initData, setInitData] = useState<any>(null);

  // State
  const [selectedUser, setSelectedUser] = useState<string>("");
  const [config, setConfig] = useState({
    model_choice: 'MA-HCL',
    alpha: 0.5,
    k: 10
  });
  const [cats, setCats] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]);

  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);

  const [vizData, setVizData] = useState<any[]>([]);
  const [showViz, setShowViz] = useState(false);

  // Init
  useEffect(() => {
    async function init() {
      try {
        const res = await api.get(endpoints.config.init);
        setInitData(res.data);
        if (res.data.users.length > 0) setSelectedUser(res.data.users[0]);
      } catch (e) {
        console.error("Init failed", e);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  // Fetch Recommendations when inputs change
  useEffect(() => {
    if (!selectedUser || loading) return;

    // Fetch function
    const fetchRecs = async () => {
      // Don't set global loading true to avoid flickering whole page
      try {
        const res = await api.post(endpoints.recommend.get, {
          user_id: selectedUser,
          history: history,
          model_choice: config.model_choice,
          alpha: config.alpha,
          k: 20,
          filters: cats
        });
        setRecommendations(res.data.recommendations);
      } catch (e) {
        console.error(e);
      }
    };

    // Only fetch if not searching
    if (!searchQuery) fetchRecs();

  }, [selectedUser, config, cats, loading, searchQuery, history]);

  // Search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    const timeout = setTimeout(async () => {
      try {
        const res = await api.post(endpoints.recommend.search, {
          query: searchQuery,
          filters: cats
        });
        setSearchResults(res.data.results);
      } catch (e) { }
    }, 500);

    return () => clearTimeout(timeout);
  }, [searchQuery, cats]);

  // Visuals Fetch
  const fetchVisuals = async () => {
    if (!showViz) {
      setShowViz(true);
      try {
        const res = await api.post(endpoints.visuals.embedding, {
          user_id: selectedUser,
          history: history,
          rec_urls: recommendations.map((r: any) => r.url),
          model_name: config.model_choice
        });
        setVizData(res.data.plot_data);
      } catch (e) { }
    } else {
      setShowViz(false);
    }
  };


  if (!initData) return <div className="flex h-screen items-center justify-center text-primary"><Loader2 className="animate-spin" size={40} /></div>;

  const displayList = searchQuery ? searchResults : recommendations;

  return (
    <div className="min-h-screen flex text-foreground bg-background transition-colors duration-300">
      <Sidebar
        users={initData.users}
        selectedUser={selectedUser}
        onUserChange={setSelectedUser}
        models={initData.model_options}
        config={config}
        onConfigChange={(k, v) => setConfig(prev => ({ ...prev, [k]: v }))}
        categories={initData.categories}
        selectedCats={cats}
        onCatChange={setCats}
      />

      <main className="flex-1 ml-80 p-10 relative">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-[0.03] pointer-events-none" />

        {/* Header / Search */}
        <div className="max-w-6xl mx-auto mb-12 flex gap-4 items-center relative z-10">
          <div className="relative flex-1 group">
            <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors" size={20} />
            <input
              type="text"
              placeholder="Search global intelligence..."
              className="w-full pl-14 pr-6 py-5 rounded-2xl bg-secondary/30 dark:bg-secondary/50 border border-border  shadow-xl focus:ring-2 focus:ring-primary/50 focus:border-primary outline-none transition-all text-lg placeholder:text-muted-foreground text-foreground backdrop-blur-sm"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <button
            onClick={fetchVisuals}
            className={cn(
              "px-8 py-5 rounded-2xl font-bold transition-all flex items-center gap-3 shadow-lg border",
              showViz
                ? "bg-primary text-primary-foreground border-primary shadow-primary/25"
                : "bg-secondary text-foreground border-border hover:bg-secondary/80 hover:border-primary/50"
            )}
          >
            {showViz ? "Hide Atlas" : "Open Atlas"} <span className="text-xs opacity-60">⌘K</span>
          </button>
        </div>

        {/* Visualizer Area */}
        <AnimatePresence>
          {showViz && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="max-w-6xl mx-auto mb-12 overflow-hidden"
            >
              <div className="p-1 border border-primary/30 rounded-3xl bg-primary/5">
                <EmbeddingPlot data={vizData} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content Grid */}
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-3xl font-bold text-foreground">
              {searchQuery ? `Intel Results: "${searchQuery}"` : `Feed: ${selectedUser}`}
            </h2>
            <div className="text-xs font-mono text-muted-foreground bg-secondary px-3 py-1 rounded-full border border-border">
              {displayList.length} ITEMS FOUND
            </div>
          </div>

          <motion.div
            layout
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            <AnimatePresence>
              {displayList.map((item: any, idx: number) => (
                <NewsCard
                  key={idx + item.url}
                  article={item}
                  variant={searchQuery ? 'search' : 'rec'}
                  onClick={() => {
                    window.open(item.url, '_blank');
                    setHistory(prev => [...prev, item.url]);
                  }}
                  onLike={() => {
                    setHistory(prev => [...prev, item.url]);
                  }}
                />
              ))}
            </AnimatePresence>

            {displayList.length === 0 && (
              <div className="col-span-full text-center py-32 text-muted-foreground">
                <div className="mb-4 text-6xl opacity-20"></div>
                No intelligence found matching your criteria.
              </div>
            )}
          </motion.div>
        </div>

      </main>
    </div>
  );
}
