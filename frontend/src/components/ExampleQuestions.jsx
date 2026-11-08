import React, { useState } from 'react';
import { HelpCircle, Sparkles, Languages, AlertCircle } from 'lucide-react';

const MANDATORY_QUESTIONS = [
  {
    id: 1,
    lang: 'Hindi',
    title: 'SLV-III उपग्रह प्रक्षेपण',
    text: 'SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?',
    target: 'रोहिणी (1980) • पृष्ठ 8',
  },
  {
    id: 2,
    lang: 'Hindi',
    title: 'भारत रत्न सम्मान',
    text: 'कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?',
    target: 'वर्ष 1997 • पृष्ठ 16',
  },
  {
    id: 3,
    lang: 'Hindi',
    title: 'पोखरण-II परमाणु परीक्षण',
    text: 'पोखरण-II में कलाम की क्या भूमिका थी?',
    target: 'मुख्य समन्वयक (1998) • पृष्ठ 10',
  },
  {
    id: 4,
    lang: 'English',
    title: 'Aeronautical Engineering',
    text: 'Which institution did Kalam attend to study aeronautical engineering?',
    target: 'MIT Chennai • Page 6',
  },
  {
    id: 5,
    lang: 'English',
    title: 'Autobiography Wings of Fire',
    text: 'Who co-wrote the autobiography, and in what year was it published?',
    target: 'Arun Tiwari (1999) • Page 3',
  },
  {
    id: 6,
    lang: 'English',
    title: 'Passing in Shillong (2015)',
    text: 'How and where did Kalam die in 2015?',
    target: 'IIM Shillong (27 July 2015) • Page 15',
  },
];

const REFUSAL_TEST_QUESTIONS = [
  {
    id: 'unans_1',
    lang: 'Hindi',
    title: 'असमर्थित तथ्य परीक्षण',
    text: 'कलाम ने किस वर्ष मंगल ग्रह (Mars) पर भारतीय अंतरिक्ष स्टेशन स्थापित किया?',
    target: 'दस्तावेज़ अस्वीकार (Refusal)',
  },
  {
    id: 'unans_2',
    lang: 'English',
    title: 'Out-of-Domain Check',
    text: "What are the names of Dr. Kalam's children and what was his favorite car?",
    target: 'Strict Grounding Refusal',
  },
];

export default function ExampleQuestions({ onSelectQuestion }) {
  const [filter, setFilter] = useState('all'); // 'all', 'hindi', 'english', 'refusal'

  const getFilteredQuestions = () => {
    if (filter === 'hindi') return MANDATORY_QUESTIONS.filter((q) => q.lang === 'Hindi');
    if (filter === 'english') return MANDATORY_QUESTIONS.filter((q) => q.lang === 'English');
    if (filter === 'refusal') return REFUSAL_TEST_QUESTIONS;
    return MANDATORY_QUESTIONS;
  };

  const questions = getFilteredQuestions();

  return (
    <div className="w-full max-w-2xl mx-auto my-4 px-2">
      {/* Category Filter Chips */}
      <div className="flex items-center justify-between mb-3 text-xs">
        <div className="flex items-center space-x-1.5 font-medium text-gray-500 dark:text-gray-400">
          <HelpCircle className="w-3.5 h-3.5 text-blue-500" />
          <span>Curated Benchmark Queries</span>
        </div>

        <div className="flex items-center space-x-1 bg-gray-100 dark:bg-[#1a1a22] p-0.5 rounded-lg text-[10px]">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`px-2 py-0.5 rounded-md transition-colors ${
              filter === 'all'
                ? 'bg-white dark:bg-[#252530] text-gray-900 dark:text-white font-medium shadow-xs'
                : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            All 6 Mandatory
          </button>
          <button
            type="button"
            onClick={() => setFilter('hindi')}
            className={`px-2 py-0.5 rounded-md transition-colors ${
              filter === 'hindi'
                ? 'bg-white dark:bg-[#252530] text-amber-600 dark:text-amber-400 font-medium shadow-xs'
                : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            Hindi (3)
          </button>
          <button
            type="button"
            onClick={() => setFilter('english')}
            className={`px-2 py-0.5 rounded-md transition-colors ${
              filter === 'english'
                ? 'bg-white dark:bg-[#252530] text-blue-600 dark:text-blue-400 font-medium shadow-xs'
                : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            English (3)
          </button>
          <button
            type="button"
            onClick={() => setFilter('refusal')}
            className={`px-2 py-0.5 rounded-md transition-colors ${
              filter === 'refusal'
                ? 'bg-white dark:bg-[#252530] text-rose-600 dark:text-rose-400 font-medium shadow-xs'
                : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            Refusal Test
          </button>
        </div>
      </div>

      {/* Questions Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {questions.map((q) => (
          <button
            key={q.id}
            type="button"
            onClick={() => onSelectQuestion(q.text)}
            className="group text-left p-3 rounded-xl border border-gray-200/90 dark:border-[#282834] bg-white dark:bg-[#18181e] hover:border-blue-500 dark:hover:border-blue-500 hover:bg-blue-50/20 dark:hover:bg-[#1d1d26] transition-all shadow-xs text-xs"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                  q.lang === 'Hindi'
                    ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                    : 'bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300'
                }`}
              >
                {q.lang}
              </span>
              <span className="text-[10px] text-gray-400 dark:text-gray-500 font-medium">
                {q.title}
              </span>
            </div>

            <p className="text-gray-800 dark:text-gray-200 font-medium leading-relaxed group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
              {q.text}
            </p>

            <div className="mt-2 pt-1.5 border-t border-gray-100 dark:border-[#22222a] flex items-center justify-between text-[10px] text-gray-400 dark:text-gray-500">
              <span className="truncate">{q.target}</span>
              <span className="text-blue-500 dark:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
                Ask →
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
