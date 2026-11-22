import React from 'react';
import { WordItem, SearchResult } from '../types';

interface SimilarityPanelProps {
  words: WordItem[];
  selectedWordIds: Set<string>;
  similarWords: SearchResult[];
  isSearching: boolean;
  onToggleSelectWord: (id: string) => void;
  onAddWord: (word: string) => void;
  onDeselectAll: () => void;
}

export const SimilarityPanel: React.FC<SimilarityPanelProps> = ({
  words,
  selectedWordIds,
  similarWords,
  isSearching,
  onToggleSelectWord,
  onAddWord,
  onDeselectAll,
}) => {
  const selectedWords = words.filter(w => selectedWordIds.has(w.id));

  return (
    <div className="w-64 h-full border-l border-gray-200 bg-white p-6 overflow-y-auto text-sm">
      {/* Selected Words Section */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Selected
          </div>
          {selectedWords.length > 0 && (
            <button
              onClick={onDeselectAll}
              className="text-xs text-gray-400 hover:text-black transition-colors cursor-pointer"
            >
              Deselect all
            </button>
          )}
        </div>

        {selectedWords.length === 0 ? (
          <div className="text-gray-400 text-xs">Click a word on the plane to select</div>
        ) : (
          <ul className="space-y-1.5">
            {selectedWords.map(w => (
              <li
                key={w.id}
                onClick={() => onToggleSelectWord(w.id)}
                className="cursor-pointer font-medium text-gray-900 hover:text-gray-500 transition-colors"
                title="Click to deselect"
              >
                {w.word}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Returned Similar Words */}
      {selectedWords.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">
            Similar
          </div>

          {isSearching ? (
            <div className="text-xs text-gray-400">Loading...</div>
          ) : (
            <ul className="space-y-1.5 text-gray-600">
              {similarWords.map(item => (
                <li
                  key={item.word}
                  onClick={() => onAddWord(item.word)}
                  className="cursor-pointer hover:text-black transition-colors"
                  title="Click to add to plane"
                >
                  {item.word}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};
