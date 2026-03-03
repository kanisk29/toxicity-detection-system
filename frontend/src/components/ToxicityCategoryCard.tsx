import { motion } from "framer-motion";
import type { ToxicityResult } from "@/lib/classifier";

interface ToxicityCategoryCardProps {
  result: ToxicityResult;
  index: number;
}

export const ToxicityCategoryCard = ({ result, index }: ToxicityCategoryCardProps) => {
  const getStatusColor = () => {
    if (!result.isToxic) return "safe";
    if (result.score > 0.6) return "danger";
    return "warning";
  };

  const status = getStatusColor();

  const colorMap = {
    safe: {
      bg: "bg-safe/10",
      border: "border-safe/30",
      text: "text-safe",
      bar: "bg-safe",
      glow: "glow-safe",
    },
    warning: {
      bg: "bg-warning/10",
      border: "border-warning/30",
      text: "text-warning",
      bar: "bg-warning",
      glow: "glow-warning",
    },
    danger: {
      bg: "bg-danger/10",
      border: "border-danger/30",
      text: "text-danger",
      bar: "bg-danger",
      glow: "glow-danger",
    },
  };

  const colors = colorMap[status];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.4 }}
      className={`relative rounded-lg border ${colors.border} ${colors.bg} p-4 transition-all hover:scale-[1.02] ${result.isToxic ? colors.glow : ""}`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-sm font-medium text-foreground">
          {result.label}
        </span>
        <span className={`font-mono text-xs font-bold ${colors.text}`}>
          {result.isToxic ? "FLAGGED" : "CLEAR"}
        </span>
      </div>

      <div className="w-full h-2 rounded-full bg-secondary overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${result.score * 100}%` }}
          transition={{ delay: index * 0.08 + 0.3, duration: 0.6, ease: "easeOut" }}
          className={`h-full rounded-full ${colors.bar}`}
        />
      </div>

      <div className="mt-2 flex justify-between">
        <span className="text-xs text-muted-foreground">Confidence</span>
        <span className={`font-mono text-xs ${colors.text}`}>
          {(result.score * 100).toFixed(1)}%
        </span>
      </div>
    </motion.div>
  );
};
