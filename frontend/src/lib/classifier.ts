export interface ToxicityResult {
  label: string;
  score: number;
  isToxic: boolean;
}

export interface AnalysisResult {
  results: ToxicityResult[];
  overallSafe: boolean;
  timestamp: Date;
  comment: string;
}

const LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];

const LABEL_DISPLAY: Record<string, string> = {
  toxic: "Toxic",
  severe_toxic: "Severe Toxic",
  obscene: "Obscene",
  threat: "Threat",
  insult: "Insult",
  identity_hate: "Identity Hate",
};

// Simulated inference — replace with actual API call to your model backend
export function analyzeComment(comment: string): AnalysisResult {
  const words = comment.toLowerCase().split(/\s+/);
  const length = words.length;

  // Simple heuristic simulation based on keyword presence
  const toxicKeywords: Record<string, string[]> = {
    toxic: ["stupid", "idiot", "hate", "kill", "die", "ugly", "dumb", "loser", "suck", "worst"],
    severe_toxic: ["kill", "die", "murder"],
    obscene: ["damn", "hell", "crap"],
    threat: ["kill", "murder", "destroy", "bomb", "attack"],
    insult: ["stupid", "idiot", "dumb", "loser", "ugly", "moron", "fool"],
    identity_hate: ["hate", "racist"],
  };

  const results: ToxicityResult[] = LABELS.map((label) => {
    const keywords = toxicKeywords[label] || [];
    const matchCount = words.filter((w) => keywords.includes(w)).length;
    const baseScore = Math.min(matchCount * 0.25, 0.85);
    const noise = Math.random() * 0.1;
    const score = Math.min(baseScore + noise + (length > 20 ? 0.05 : 0), 1);

    return {
      label: LABEL_DISPLAY[label] || label,
      score: parseFloat(score.toFixed(3)),
      isToxic: score > 0.3,
    };
  });

  return {
    results,
    overallSafe: results.every((r) => !r.isToxic),
    timestamp: new Date(),
    comment,
  };
}
