import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { WordItem, SearchResult } from './types';
import { computePCA } from './utils/pca';
import { embeddingService } from './services/embeddingService';
import { SemanticPlane } from './components/SemanticPlane';
import { SimilarityPanel } from './components/SimilarityPanel';
import { InputBar } from './components/InputBar';

export const App: React.FC = () => {
  const [words, setWords] = useState<WordItem[]>([]);
  const [selectedWordIds, setSelectedWordIds] = useState<Set<string>>(new Set());
  const [similarWords, setSimilarWords] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [isEmbedding, setIsEmbedding] = useState<boolean>(false);
  const [focusTrigger, setFocusTrigger] = useState<number>(0);

  // Initialize embedding service on startup
  useEffect(() => {
    embeddingService.init();
  }, []);

  // Deselect all
  const handleDeselectAll = useCallback(() => {
    setSelectedWordIds(new Set());
    setFocusTrigger(t => t + 1);
  }, []);

  // Global 'Escape' key listener to deselect all words
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleDeselectAll();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleDeselectAll]);

  // Set of current lowercase words on canvas for fast lookup
  const canvasWordsSet = useMemo(() => {
    return new Set(words.map(w => w.word.toLowerCase()));
  }, [words]);

  // Recalculate semantic similarity whenever selected words change
  const updateSimilarity = useCallback(async () => {
    const selected = words.filter(w => selectedWordIds.has(w.id));
    if (selected.length === 0) {
      setSimilarWords([]);
      return;
    }

    setIsSearching(true);
    try {
      const results = await embeddingService.findSimilarToAllWords(
        selected.map(s => ({ word: s.word, vector: s.vector })),
        canvasWordsSet,
        20
      );
      setSimilarWords(results);
    } finally {
      setIsSearching(false);
    }
  }, [words, selectedWordIds, canvasWordsSet]);

  useEffect(() => {
    updateSimilarity();
  }, [selectedWordIds, updateSimilarity]);

  // Add word to canvas & automatically select it
  const handleAddWord = useCallback(
    async (wordStr: string): Promise<boolean> => {
      const clean = wordStr.toLowerCase().trim();
      if (!clean) return false;

      // If already on canvas, select it
      const existing = words.find(w => w.word.toLowerCase() === clean);
      if (existing) {
        setSelectedWordIds(prev => new Set([...prev, existing.id]));
        setFocusTrigger(t => t + 1);
        return true;
      }

      setIsEmbedding(true);
      try {
        const vector = await embeddingService.getWordVector(clean);
        const newId = `word-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
        const newItem: WordItem = {
          id: newId,
          word: clean,
          vector,
          x: 0,
          y: 0,
          addedAt: Date.now(),
        };

        const nextWords = [...words, newItem];

        // Recompute PCA for all words on the plane
        const pcaRes = computePCA(nextWords.map(w => ({ id: w.id, vector: w.vector })));
        const pointMap = new Map(pcaRes.points.map(p => [p.id, p]));

        const updated = nextWords.map(item => {
          const pt = pointMap.get(item.id);
          return pt ? { ...item, x: pt.x, y: pt.y } : item;
        });

        setWords(updated);
        // Automatically select the newly added word
        setSelectedWordIds(prev => new Set([...prev, newId]));
        setFocusTrigger(t => t + 1);
        return true;
      } finally {
        setIsEmbedding(false);
      }
    },
    [words]
  );

  // Remove word from canvas
  const handleRemoveWord = useCallback((id: string) => {
    setWords(prevWords => {
      const nextWords = prevWords.filter(w => w.id !== id);
      setSelectedWordIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });

      if (nextWords.length === 0) {
        return [];
      }

      const pcaRes = computePCA(nextWords.map(w => ({ id: w.id, vector: w.vector })));
      const pointMap = new Map(pcaRes.points.map(p => [p.id, p]));
      return nextWords.map(item => {
        const pt = pointMap.get(item.id);
        return pt ? { ...item, x: pt.x, y: pt.y } : item;
      });
    });
    setFocusTrigger(t => t + 1);
  }, []);

  // Toggle selection for a word
  const handleToggleSelectWord = useCallback((id: string) => {
    setSelectedWordIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
    setFocusTrigger(t => t + 1);
  }, []);

  return (
    <div className="flex flex-col w-screen h-screen bg-white text-gray-900 overflow-hidden">
      {/* Clean Single-line Input Bar (Always Focused) */}
      <InputBar
        onAddWord={handleAddWord}
        isEmbedding={isEmbedding}
        focusTrigger={focusTrigger}
      />

      {/* Main Plane & Sidebar */}
      <div className="flex-1 flex overflow-hidden">
        {/* 2D Plane */}
        <div className="flex-1 h-full">
          <SemanticPlane
            words={words}
            selectedWordIds={selectedWordIds}
            onToggleSelectWord={handleToggleSelectWord}
            onRemoveWord={handleRemoveWord}
          />
        </div>

        {/* Sidebar: Only selected words and their similar words */}
        <SimilarityPanel
          words={words}
          selectedWordIds={selectedWordIds}
          similarWords={similarWords}
          isSearching={isSearching}
          onToggleSelectWord={handleToggleSelectWord}
          onAddWord={handleAddWord}
          onDeselectAll={handleDeselectAll}
        />
      </div>
    </div>
  );
};

export default App;
