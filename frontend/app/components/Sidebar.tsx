"use client";

import React from 'react';
import { Settings, Users, Filter, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
    users: string[];
    selectedUser: string;
    onUserChange: (u: string) => void;
    models: any;
    config: any;
    onConfigChange: (key: string, val: any) => void;
    categories: any;
    selectedCats: string[];
    onCatChange: (cats: string[]) => void;
}

export function Sidebar({
    users, selectedUser, onUserChange,
    models, config, onConfigChange,
    categories, selectedCats, onCatChange
}: SidebarProps) {

    return (
        <div className="w-80 h-screen fixed left-0 top-0 p-4 z-40">
            <div className="w-full h-full glass-panel rounded-2xl flex flex-col p-6 overflow-y-auto no-scrollbar transition-colors duration-300">

                {/* Header */}
                <div className="flex items-center gap-3 mb-10">
                    <div className="p-2.5 bg-primary/20 rounded-xl border border-primary/30 shadow-[0_0_15px_rgba(99,102,241,0.3)]">
                        <Zap size={24} className="fill-primary text-primary" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold tracking-tight text-foreground leading-none">NewsEngine</h1>
                        <span className="text-[10px] font-medium text-primary/80 uppercase tracking-[0.2em]">Intel Core</span>
                    </div>
                </div>

                {/* User Section */}
                <div className="mb-8">
                    <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Users size={14} className="text-primary" /> Target Profile
                    </label>
                    <div className="relative group">
                        <select
                            value={selectedUser}
                            onChange={(e) => onUserChange(e.target.value)}
                            className="w-full p-3 bg-secondary/50 border border-border rounded-xl text-sm focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all font-medium text-foreground appearance-none hover:border-primary/50"
                        >
                            {users.map(u => (
                                <option key={u} value={u} className="bg-card text-foreground">{u}</option>
                            ))}
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground group-hover:text-primary transition-colors">
                            ▼
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="mb-8">
                    <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Settings size={14} className="text-primary" /> Neural Config
                    </label>

                    <div className="space-y-5">
                        <div className="p-4 rounded-xl bg-secondary/40 border border-border/50">
                            <span className="text-xs font-medium text-muted-foreground block mb-2">Inference Model</span>
                            <div className="relative">
                                <select
                                    value={config.model_choice}
                                    onChange={(e) => onConfigChange('model_choice', e.target.value)}
                                    className="w-full p-2 bg-background border border-border rounded-lg text-sm text-foreground focus:outline-none focus:border-primary appearance-none"
                                >
                                    {models?.CF?.map((m: string) => (
                                        <option key={m} value={m} className="bg-background">{m}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="p-4 rounded-xl bg-secondary/40 border border-border/50">
                            <div className="flex justify-between text-xs mb-3">
                                <span className="font-medium text-muted-foreground">Fusion Alpha</span>
                                <span className="text-primary font-mono font-bold bg-primary/10 px-2 rounded-md">{config.alpha}</span>
                            </div>
                            <input
                                type="range" min="0" max="1" step="0.1"
                                value={config.alpha}
                                onChange={(e) => onConfigChange('alpha', parseFloat(e.target.value))}
                                className="w-full h-1.5 bg-secondary rounded-full appearance-none cursor-pointer accent-primary hover:accent-primary/80 transition-all"
                            />
                            <div className="flex justify-between text-[10px] text-muted-foreground mt-2 font-medium tracking-wide">
                                <span>CONTENT</span>
                                <span>SOCIAL</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Categories */}
                <div className="mb-8">
                    <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Filter size={14} className="text-primary" /> Signal Filters
                    </label>
                    <div className="flex flex-wrap gap-2">
                        {Object.entries(categories || {}).map(([key, label]: any) => (
                            <button
                                key={key}
                                onClick={() => {
                                    if (selectedCats.includes(key)) onCatChange(selectedCats.filter(c => c !== key));
                                    else onCatChange([...selectedCats, key]);
                                }}
                                className={cn(
                                    "text-xs px-3 py-1.5 rounded-lg border transition-all duration-300 font-medium",
                                    selectedCats.includes(key)
                                        ? "bg-primary text-primary-foreground border-primary shadow-[0_0_10px_rgba(99,102,241,0.4)]"
                                        : "bg-secondary/50 text-muted-foreground border-border hover:border-primary/50 hover:text-primary"
                                )}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="mt-auto pt-6 border-t border-border/50 text-center">
                    <p className="text-[10px] text-muted-foreground">
                        Powered by <span className="text-foreground font-bold">FastAPI</span> & <span className="text-foreground font-bold">Next.js</span>
                    </p>
                </div>

            </div>
        </div>
    );
}
