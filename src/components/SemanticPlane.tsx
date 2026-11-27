import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { WordItem } from '../types';

interface SemanticPlaneProps {
  words: WordItem[];
  selectedWordIds: Set<string>;
  onToggleSelectWord: (id: string) => void;
  onRemoveWord: (id: string) => void;
}

interface HoveredLineInfo {
  wordA: WordItem;
  wordB: WordItem;
  screenMidX: number;
  screenMidY: number;
  cosineSim: number;
  cosineDist: number;
  euclidean384: number;
  euclidean2D: number;
}

export const SemanticPlane: React.FC<SemanticPlaneProps> = ({
  words,
  selectedWordIds,
  onToggleSelectWord,
  onRemoveWord,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [zoom, setZoom] = useState<number>(1);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredWordId, setHoveredWordId] = useState<string | null>(null);
  const [hoveredLine, setHoveredLine] = useState<HoveredLineInfo | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    wordId: string;
    word: string;
    x: number;
    y: number;
  } | null>(null);

  // Close context menu on outside click or escape
  useEffect(() => {
    const handleDismiss = () => setContextMenu(null);
    window.addEventListener('click', handleDismiss);
    window.addEventListener('contextmenu', (e) => {
      // If right click outside canvas, close
      if ((e.target as HTMLElement).tagName !== 'CANVAS') {
        setContextMenu(null);
      }
    });
    return () => {
      window.removeEventListener('click', handleDismiss);
    };
  }, []);

  // Auto-fit all points to viewport with guaranteed visibility
  const fitToView = useCallback(() => {
    if (words.length === 0) {
      setPan({ x: 0, y: 0 });
      setZoom(1);
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const w of words) {
      if (w.x < minX) minX = w.x;
      if (w.x > maxX) maxX = w.x;
      if (w.y < minY) minY = w.y;
      if (w.y > maxY) maxY = w.y;
    }

    if (words.length === 1) {
      setPan({ x: -minX, y: minY });
      setZoom(1.0);
      return;
    }

    // Account for word label bounds around each node (~50px horizontal, ~25px vertical)
    const spanX = Math.max(maxX - minX + 80, 80);
    const spanY = Math.max(maxY - minY + 60, 60);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const paddingX = 140;
    const paddingY = 140;
    const usableWidth = Math.max(canvas.clientWidth - paddingX, 100);
    const usableHeight = Math.max(canvas.clientHeight - paddingY, 100);

    // Use Math.min so BOTH width and height fit inside the viewport!
    const zoomX = usableWidth / spanX;
    const zoomY = usableHeight / spanY;
    const optimalZoom = Math.min(zoomX, zoomY);

    const safeZoom = Math.min(Math.max(optimalZoom, 0.05), 1.2);

    setZoom(safeZoom);
    // Invert Y pan because world Y is positive upwards, screen Y is positive downwards
    setPan({
      x: -centerX * safeZoom,
      y: centerY * safeZoom,
    });
  }, [words]);

  // Automatically zoom out to fit everything whenever word count increases
  const prevWordCountRef = useRef(words.length);
  useEffect(() => {
    if (words.length > prevWordCountRef.current) {
      fitToView();
    }
    prevWordCountRef.current = words.length;
  }, [words.length, fitToView]);

  const worldToScreen = useCallback(
    (worldX: number, worldY: number, width: number, height: number) => {
      const originX = width / 2 + pan.x;
      const originY = height / 2 + pan.y;
      const screenX = originX + worldX * zoom;
      const screenY = originY - worldY * zoom;
      return { x: screenX, y: screenY };
    },
    [pan, zoom]
  );

  // Selected word items
  const selectedWords = useMemo(() => {
    return words.filter(w => selectedWordIds.has(w.id));
  }, [words, selectedWordIds]);

  // Pairs of selected words to connect
  const selectedPairs = useMemo(() => {
    const pairs: [WordItem, WordItem][] = [];
    if (selectedWords.length < 2) return pairs;

    for (let i = 0; i < selectedWords.length; i++) {
      for (let j = i + 1; j < selectedWords.length; j++) {
        pairs.push([selectedWords[i], selectedWords[j]]);
      }
    }
    return pairs;
  }, [selectedWords]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    // 1. Plain White Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // 2. Faint minimal grid lines
    const originX = width / 2 + pan.x;
    const originY = height / 2 + pan.y;
    const gridSize = 60 * zoom;

    ctx.lineWidth = 1;
    ctx.strokeStyle = '#f8fafc';

    if (gridSize > 25) {
      const startX = (originX % gridSize) - gridSize;
      for (let x = startX; x < width + gridSize; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      const startY = (originY % gridSize) - gridSize;
      for (let y = startY; y < height + gridSize; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    // 3. Subtle axes
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(0, originY);
    ctx.lineTo(width, originY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(originX, 0);
    ctx.lineTo(originX, height);
    ctx.stroke();

    // 4. Draw Connecting Lines between Selected Words
    for (const [a, b] of selectedPairs) {
      const ptA = worldToScreen(a.x, a.y, width, height);
      const ptB = worldToScreen(b.x, b.y, width, height);

      const isLineHovered =
        hoveredLine &&
        ((hoveredLine.wordA.id === a.id && hoveredLine.wordB.id === b.id) ||
          (hoveredLine.wordA.id === b.id && hoveredLine.wordB.id === a.id));

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(ptA.x, ptA.y);
      ctx.lineTo(ptB.x, ptB.y);

      if (isLineHovered) {
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
      } else {
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 5. Draw Nodes (All nodes have the exact same color)
    const NODE_COLOR = '#111827';
    const RADIUS = 6;

    for (const item of words) {
      const pt = worldToScreen(item.x, item.y, width, height);
      const isSelected = selectedWordIds.has(item.id);

      ctx.save();

      // If selected: draw clean outer outline ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, RADIUS + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Circle node (same color for all nodes)
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = NODE_COLOR;
      ctx.fill();

      // Word Label
      ctx.font = isSelected
        ? '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
        : '400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

      ctx.fillStyle = isSelected ? '#000000' : '#4b5563';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(item.word, pt.x, pt.y + RADIUS + 4);

      ctx.restore();
    }

    ctx.restore();
  }, [words, selectedWordIds, selectedPairs, hoveredLine, pan, zoom, worldToScreen]);

  // Compute difference between two vectors
  const computeMathDifference = (
    a: WordItem,
    b: WordItem,
    screenMidX: number,
    screenMidY: number
  ): HoveredLineInfo => {
    const dim = Math.min(a.vector.length, b.vector.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let dist384Sq = 0;

    for (let i = 0; i < dim; i++) {
      const va = a.vector[i];
      const vb = b.vector[i];
      dot += va * vb;
      normA += va * va;
      normB += vb * vb;
      const diff = va - vb;
      dist384Sq += diff * diff;
    }

    normA = Math.sqrt(normA) || 1;
    normB = Math.sqrt(normB) || 1;
    const cosineSim = Math.max(-1, Math.min(1, dot / (normA * normB)));
    const cosineDist = 1 - cosineSim;
    const euclidean384 = Math.sqrt(dist384Sq);

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const euclidean2D = Math.sqrt(dx * dx + dy * dy);

    return {
      wordA: a,
      wordB: b,
      screenMidX,
      screenMidY,
      cosineSim,
      cosineDist,
      euclidean384,
      euclidean2D,
    };
  };

  // Detect hover over lines between selected words
  const checkLineHover = (screenX: number, screenY: number): HoveredLineInfo | null => {
    const canvas = canvasRef.current;
    if (!canvas || selectedPairs.length === 0) return null;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    for (const [a, b] of selectedPairs) {
      const ptA = worldToScreen(a.x, a.y, width, height);
      const ptB = worldToScreen(b.x, b.y, width, height);

      const dx = ptB.x - ptA.x;
      const dy = ptB.y - ptA.y;
      const lenSq = dx * dx + dy * dy;

      if (lenSq === 0) continue;

      let t = ((screenX - ptA.x) * dx + (screenY - ptA.y) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));

      const projX = ptA.x + t * dx;
      const projY = ptA.y + t * dy;

      const distSq = (screenX - projX) * (screenX - projX) + (screenY - projY) * (screenY - projY);

      if (distSq <= 10 * 10) {
        return computeMathDifference(a, b, (ptA.x + ptB.x) / 2, (ptA.y + ptB.y) / 2);
      }
    }

    return null;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (contextMenu) setContextMenu(null);
    if (e.button !== 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    const hit = findWordAtScreenPos(screenX, screenY);
    if (hit) {
      onToggleSelectWord(hit.id);
      return;
    }

    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    const hit = findWordAtScreenPos(screenX, screenY);
    if (hit) {
      setHoveredLine(null);
      setContextMenu({
        wordId: hit.id,
        word: hit.word,
        x: screenX,
        y: screenY,
      });
    } else {
      setContextMenu(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    } else {
      const hit = findWordAtScreenPos(screenX, screenY);
      setHoveredWordId(hit ? hit.id : null);

      if (!hit) {
        const lineHit = checkLineHover(screenX, screenY);
        setHoveredLine(lineHit);
      } else {
        setHoveredLine(null);
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Non-passive wheel event listener to strictly zoom only the plane
  // and prevent browser page zoom or page scrolling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Trackpad pinch-to-zoom sets e.ctrlKey and produces fine delta values
      let factor = 1;
      if (e.ctrlKey) {
        factor = Math.exp(-e.deltaY * 0.01);
      } else {
        factor = e.deltaY < 0 ? 1.08 : 0.92;
      }

      setZoom(currentZoom => {
        const newZoom = Math.min(Math.max(currentZoom * factor, 0.04), 6.0);

        setPan(currentPan => {
          const originX = canvas.clientWidth / 2 + currentPan.x;
          const originY = canvas.clientHeight / 2 + currentPan.y;

          const newPanX =
            mouseX - (mouseX - originX) * (newZoom / currentZoom) - canvas.clientWidth / 2;
          const newPanY =
            mouseY - (mouseY - originY) * (newZoom / currentZoom) - canvas.clientHeight / 2;

          return { x: newPanX, y: newPanY };
        });

        return newZoom;
      });
    };

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleNativeWheel);
    };
  }, []);

  const findWordAtScreenPos = (screenX: number, screenY: number): WordItem | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    for (let i = words.length - 1; i >= 0; i--) {
      const item = words[i];
      const pt = worldToScreen(item.x, item.y, width, height);

      const dx = screenX - pt.x;
      const dy = screenY - pt.y;
      if (dx * dx + dy * dy <= 16 * 16) {
        return item;
      }
      if (screenX >= pt.x - 30 && screenX <= pt.x + 30 && screenY >= pt.y && screenY <= pt.y + 24) {
        return item;
      }
    }
    return null;
  };

  return (
    <div className="relative w-full h-full bg-white overflow-hidden select-none">
      {/* Top Right Fit View Icon Button */}
      <button
        onClick={fitToView}
        className="absolute top-3 right-3 text-gray-500 hover:text-black bg-white hover:bg-gray-50 border border-gray-200 p-1.5 rounded shadow-sm transition-colors cursor-pointer z-10"
        title="Fit all words in view"
      >
        <svg
          className="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
        </svg>
      </button>

      {/* 2D Plane Canvas */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleContextMenu}
        onMouseLeave={() => {
          setIsDragging(false);
          setHoveredLine(null);
        }}
        className={`w-full h-full block ${isDragging ? 'cursor-grabbing' : hoveredWordId ? 'cursor-pointer' : hoveredLine ? 'cursor-crosshair' : 'cursor-grab'}`}
      />

      {/* Right-Click Node Delete Context Menu */}
      {contextMenu && (
        <div
          className="absolute z-30 bg-white border border-gray-200 rounded shadow-md py-1 min-w-[110px] text-xs select-none"
          style={{
            left: `${contextMenu.x}px`,
            top: `${contextMenu.y}px`,
          }}
          onClick={e => e.stopPropagation()}
        >
          <button
            onClick={() => {
              onRemoveWord(contextMenu.wordId);
              setContextMenu(null);
            }}
            className="w-full text-left px-3 py-1.5 text-red-600 hover:bg-red-50 flex items-center justify-between cursor-pointer transition-colors"
          >
            <span>Delete</span>
            <span className="text-gray-400 text-[10px] ml-2">"{contextMenu.word}"</span>
          </button>
        </div>
      )}

      {/* Tooltip on Line Hover */}
      {hoveredLine && (
        <div
          className="absolute pointer-events-none z-20 bg-white border border-gray-200 rounded p-2.5 shadow-sm text-xs text-gray-900"
          style={{
            left: `${hoveredLine.screenMidX + 12}px`,
            top: `${hoveredLine.screenMidY - 16}px`,
            transform: 'translateY(-50%)',
          }}
        >
          <div className="flex items-center gap-1.5 font-medium mb-2">
            <span>{hoveredLine.wordA.word}</span>
            <span className="text-gray-400">↔</span>
            <span>{hoveredLine.wordB.word}</span>
          </div>

          <div className="space-y-1 text-gray-500">
            <div className="flex justify-between gap-4">
              <span>Similarity</span>
              <span className="text-gray-900">
                {Math.round(Math.max(0, hoveredLine.cosineSim) * 100)}%
              </span>
            </div>
            <div className="flex justify-between gap-4">
              <span>Semantic distance</span>
              <span className="text-gray-900">
                {hoveredLine.cosineDist.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
