"use client";

import React from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';

// Dynamically import Plotly (Client side only)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export function EmbeddingPlot({ data }: { data: any[] }) {
    if (!data || data.length === 0) return null;

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-panel rounded-2xl p-4 h-[600px] w-full"
        >
            <Plot
                data={data}
                layout={{
                    autosize: true,
                    title: { text: 'Semantic Atlas', font: { color: '#fff' } },
                    hovermode: 'closest',
                    xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    margin: { l: 20, r: 20, t: 50, b: 20 },
                    legend: { x: 1, y: 1 },
                    plot_bgcolor: "rgba(0,0,0,0)",
                    paper_bgcolor: "rgba(0,0,0,0)",
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
            />
        </motion.div>
    );
}
