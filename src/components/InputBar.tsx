import React, { useState, useRef, useEffect } from 'react';

interface InputBarProps {
  onAddWord: (word: string) => Promise<boolean>;
  isEmbedding: boolean;
  focusTrigger?: number;
}

export const InputBar: React.FC<InputBarProps> = ({
  onAddWord,
  isEmbedding,
  focusTrigger,
}) => {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Keep input focused automatically
  useEffect(() => {
    inputRef.current?.focus();
  }, [focusTrigger, isEmbedding]);

  // Global listener to keep text field focused when clicking around
  useEffect(() => {
    const handleGlobalClick = (e: MouseEvent) => {
      // Don't steal focus if user is selecting text or clicking interactive elements
      const target = e.target as HTMLElement;
      if (target.tagName === 'BUTTON' || target.tagName === 'A' || target.tagName === 'INPUT') {
        return;
      }
      inputRef.current?.focus();
    };

    window.addEventListener('click', handleGlobalClick);
    return () => window.removeEventListener('click', handleGlobalClick);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim().toLowerCase();
    if (!trimmed) return;

    setValue('');
    try {
      await onAddWord(trimmed);
    } finally {
      // Always re-focus immediately after adding a word
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  };

  return (
    <div className="border-b border-gray-200 px-6 py-3 bg-white">
      <form onSubmit={handleSubmit} className="max-w-md">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          placeholder="Type an English word and press Enter..."
          autoFocus
          className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-black transition-colors"
        />
      </form>
    </div>
  );
};
