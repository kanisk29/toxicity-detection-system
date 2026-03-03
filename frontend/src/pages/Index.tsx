import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, RotateCcw, ShieldCheck } from "lucide-react";
import { analyzeComment, type AnalysisResult } from "@/lib/classifier";
import { ToxicityCategoryCard } from "@/components/ToxicityCategoryCard";
import { OverallVerdict } from "@/components/OverallVerdict";

const EXAMPLE_COMMENTS = [
  "This is a great article, thanks for sharing!",
  "You are such an idiot, go away loser",
  "I will find you and destroy everything you love",
  "What a wonderful day to learn something new",
];

const Index = () => {
  const [comment, setComment] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState<AnalysisResult[]>([]);

  const handleAnalyze = async () => {
    if (!comment.trim()) return;
    setIsAnalyzing(true);
    setResult(null);

    // Simulate network delay
    await new Promise((r) => setTimeout(r, 800 + Math.random() * 600));

    const analysis = analyzeComment(comment);
    setResult(analysis);
    setHistory((prev) => [analysis, ...prev].slice(0, 10));
    setIsAnalyzing(false);
  };

  const handleReset = () => {
    setComment("");
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-background grid-pattern">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-10">
        <div className="container max-w-5xl mx-auto px-4 py-4 flex items-center gap-3">
          <ShieldCheck className="w-7 h-7 text-primary" />
          <div>
            <h1 className="text-lg font-bold font-mono tracking-tight text-foreground">
              TOXICITY CLASSIFIER
            </h1>
            <p className="text-xs text-muted-foreground font-mono">
              Jigsaw Multi-Label Classification • LSTM / GRU / CNN
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-safe/10 text-safe text-xs font-mono border border-safe/20">
              <span className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse" />
              MODEL READY
            </span>
          </div>
        </div>
      </header>

      <main className="container max-w-5xl mx-auto px-4 py-8">
        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <label className="block text-sm font-mono text-muted-foreground mb-2 uppercase tracking-wider">
            Input Comment
          </label>
          <div className="relative">
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Type or paste a comment to analyze for toxicity..."
              className="w-full h-32 px-4 py-3 rounded-lg border border-border bg-card text-foreground placeholder:text-muted-foreground font-sans text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleAnalyze();
              }}
            />
            <div className="absolute bottom-3 right-3 flex gap-2">
              {comment && (
                <button
                  onClick={handleReset}
                  className="p-2 rounded-md bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={handleAnalyze}
                disabled={!comment.trim() || isAnalyzing}
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-mono text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-all"
              >
                {isAnalyzing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                {isAnalyzing ? "ANALYZING" : "ANALYZE"}
              </button>
            </div>
          </div>

          {/* Quick Examples */}
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-xs text-muted-foreground font-mono">EXAMPLES:</span>
            {EXAMPLE_COMMENTS.map((ex, i) => (
              <button
                key={i}
                onClick={() => setComment(ex)}
                className="text-xs px-2 py-1 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors truncate max-w-[200px]"
              >
                {ex}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Results */}
        <AnimatePresence mode="wait">
          {isAnalyzing && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center py-16 gap-4"
            >
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
              <p className="font-mono text-sm text-muted-foreground">
                Running inference...
              </p>
            </motion.div>
          )}

          {result && !isAnalyzing && (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {/* Overall Verdict */}
              <div className="mb-6">
                <OverallVerdict result={result} />
              </div>

              {/* Category Grid */}
              <div className="mb-4">
                <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
                  Category Breakdown
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {result.results.map((r, i) => (
                    <ToxicityCategoryCard key={r.label} result={r} index={i} />
                  ))}
                </div>
              </div>

              {/* Analyzed Text */}
              <div className="mt-6 p-4 rounded-lg border border-border bg-card">
                <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                  Analyzed Text
                </span>
                <p className="mt-2 text-sm text-foreground/80 italic">
                  "{result.comment}"
                </p>
                <p className="mt-2 text-xs text-muted-foreground font-mono">
                  {result.timestamp.toLocaleTimeString()} • Threshold: 0.3
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* History */}
        {history.length > 0 && !isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-10 border-t border-border pt-6"
          >
            <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
              Recent Analyses ({history.length})
            </h2>
            <div className="space-y-2">
              {history.map((h, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 p-3 rounded-lg bg-card border border-border hover:border-border/80 transition-colors cursor-pointer"
                  onClick={() => {
                    setComment(h.comment);
                    setResult(h);
                  }}
                >
                  <span
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${
                      h.overallSafe ? "bg-safe" : "bg-danger"
                    }`}
                  />
                  <span className="text-sm text-foreground/80 truncate flex-1">
                    {h.comment}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground flex-shrink-0">
                    {h.results.filter((r) => r.isToxic).length}/6
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default Index;
