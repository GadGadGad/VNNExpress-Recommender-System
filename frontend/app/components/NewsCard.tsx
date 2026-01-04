"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Calendar, Tag, ThumbsUp, ArrowUpRight } from 'lucide-react';

interface NewsCardProps {
    article: any;
    onClick: () => void;
    onLike?: () => void;
    variant?: 'rec' | 'search' | 'history';
}

export function NewsCard({ article, onClick, onLike, variant = 'rec' }: NewsCardProps) {
    const isBoosted = article.source?.includes('Boosted');

    const badgeStyle =
        variant === 'search' ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20' :
            article.source?.includes('Social') ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' :
                'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ y: -5, transition: { duration: 0.2 } }}
            className={cn(
                "glass-card rounded-xl p-5 border border-slate-700/50 shadow-lg relative overflow-hidden group cursor-pointer flex flex-col h-full",
                isBoosted && "border-l-4 border-l-red-500"
            )}
            onClick={onClick}
        >
            {/* Glowing effect on hover */}
            <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="flex justify-between items-start mb-4 relative z-10">
                <span className={cn("text-[10px] font-bold px-2.5 py-1 rounded-md uppercase tracking-wider border", badgeStyle)}>
                    {article.source || variant}
                </span>

                <div className="flex gap-2">
                    {onLike && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onLike(); }}
                            className="p-1.5 rounded-full hover:bg-slate-700 text-slate-500 hover:text-rose-500 transition-colors"
                        >
                            <ThumbsUp size={14} />
                        </button>
                    )}
                    <div className="p-1.5 rounded-full hover:bg-slate-700 text-slate-500 group-hover:text-indigo-400 transition-colors">
                        <ArrowUpRight size={14} />
                    </div>
                </div>
            </div>

            <h3 className="font-bold text-lg text-slate-200 mb-3 leading-snug group-hover:text-white group-hover:text-glow transition-all relative z-10">
                {article.title}
            </h3>

            <p className="text-slate-400 text-sm line-clamp-3 mb-6 flex-1 relative z-10">
                {article.description || "No description available."}
            </p>

            <div className="flex items-center gap-4 text-xs text-slate-500 mt-auto relative z-10 border-t border-slate-700/50 pt-4 w-full">
                <div className="flex items-center gap-1.5">
                    <Calendar size={12} />
                    <span className="font-medium">{article.published_at?.split('T')[0] || "Just now"}</span>
                </div>
                {article.score > 0 && (
                    <div className="flex items-center gap-1 font-bold text-indigo-400 ml-auto bg-indigo-500/10 px-2 py-0.5 rounded-md">
                        Score: {article.score.toFixed(3)}
                    </div>
                )}
            </div>

        </motion.div>
    );
}
