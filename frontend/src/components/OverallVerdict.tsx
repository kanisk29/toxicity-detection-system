import { motion } from "framer-motion";
import { Shield, ShieldAlert } from "lucide-react";
import type { AnalysisResult } from "@/lib/classifier";

interface OverallVerdictProps {
  result: AnalysisResult;
}

export const OverallVerdict = ({ result }: OverallVerdictProps) => {
  const flaggedCount = result.results.filter((r) => r.isToxic).length;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`rounded-xl border p-6 text-center ${
        result.overallSafe
          ? "border-safe/30 bg-safe/5 glow-safe"
          : "border-danger/30 bg-danger/5 glow-danger"
      }`}
    >
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
        className="flex justify-center mb-3"
      >
        {result.overallSafe ? (
          <Shield className="w-12 h-12 text-safe" />
        ) : (
          <ShieldAlert className="w-12 h-12 text-danger" />
        )}
      </motion.div>

      <h3 className={`text-xl font-bold font-mono ${result.overallSafe ? "text-safe" : "text-danger"}`}>
        {result.overallSafe ? "SAFE" : "FLAGGED"}
      </h3>

      <p className="text-sm text-muted-foreground mt-1">
        {result.overallSafe
          ? "No toxicity detected in this comment."
          : `${flaggedCount} ${flaggedCount === 1 ? "category" : "categories"} flagged as toxic.`}
      </p>
    </motion.div>
  );
};
