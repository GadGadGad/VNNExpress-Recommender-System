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
        variant === 'search' ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/20' :
            article.source?.includes('Social') ? 'bg-primary/10 text-primary border-primary/20' :
                'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ y: -5, transition: { duration: 0.2 } }}
            className={cn(
                "glass-card rounded-xl p-5 border border-border/50 shadow-lg relative overflow-hidden group cursor-pointer flex flex-col h-full",
                isBoosted && "border-l-4 border-l-destructive"
            )}
            onClick={onClick}
        >
            {/* Glowing effect on hover */}
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="flex justify-between items-start mb-4 relative z-10">
                <span className={cn("text-[10px] font-bold px-2.5 py-1 rounded-md uppercase tracking-wider border", badgeStyle)}>
                    {article.source || variant}
                </span>

                <div className="flex gap-2">
                    {onLike && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onLike(); }}
                            className="p-1.5 rounded-full hover:bg-secondary text-muted-foreground hover:text-destructive transition-colors"
                        >
                            <ThumbsUp size={14} />
                        </button>
                    )}
                </div>
            </div>

            <h3 className="text-lg font-bold text-foreground mb-3 leading-tight group-hover:text-primary transition-colors flex-1">
                {article.title}
            </h3>

            <p className="text-sm text-muted-foreground mb-6 line-clamp-3 leading-relaxed relative z-10">
                {article.short_description || article.description || "No description available."}
            </p>

            <div className="mt-auto flex items-center justify-between text-[11px] font-medium text-muted-foreground border-t border-border/50 pt-4 relative z-10 w-full">
                <div className="flex items-center gap-4">
                    {article.category_name && (
                        <span className="flex items-center gap-1.5">
                            <Tag size={12} className="text-primary" />
                            {article.category_name}
                        </span>
                    )}
                    <span className="flex items-center gap-1.5">
                        <Calendar size={12} className="text-primary" />
                        {article.published_at?.split('T')[0] || "Just now"}
                    </span>
                </div>

                {article.score > 0 && (
                    <div className="flex items-center gap-1 font-bold text-primary ml-auto bg-primary/10 px-2 py-0.5 rounded-md">
                        {article.score.toFixed(3)}
                    </div>
                )}
            </div>
        </motion.div>
    );
}
