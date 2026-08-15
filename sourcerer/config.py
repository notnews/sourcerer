"""Central configuration: model pin, request defaults, truncation limits.

These lived as hardcoded defaults in fifteen function signatures, which is how
the model pin silently went stale: the string was changed nowhere when the model
it named was retired, and nothing failed until a request was actually made.
One definition, imported everywhere.
"""

from __future__ import annotations

# Pinned for reproducibility; record alongside any output. Update deliberately,
# not incidentally -- a model change changes the labels and therefore the
# results, so it invalidates any figure computed under the previous pin.
MODEL = "claude-sonnet-5"

# The gold-standard tier used only to label validation samples, never the
# production path. Reserving the expensive model for the sample is what makes
# a second-coder comparison affordable.
VALIDATION_MODEL = "claude-opus-5"

# Claude Sonnet 5 runs adaptive thinking by default, and max_tokens caps
# thinking *plus* response text. Claim extraction and evidence classification
# are structured-output tasks driven by a forced tool_choice, not reasoning
# ones: with thinking on, a 4096-token budget is consumed before the tool call
# is emitted and the response truncates mid-call. Disable it explicitly rather
# than inflating max_tokens to pay for reasoning we do not want.
THINKING: dict[str, str] = {"type": "disabled"}

MAX_TOKENS = 4096

# Truncation is by characters, not words. The run that produced the shipped
# results used a 3,000-WORD cap, which silently discarded roughly 47% of the
# mean transcript -- claims in the back half of an hour of television were
# never seen, and nothing recorded that they had been dropped.
EXTRACTION_MAX_CHARS = 50_000
EVIDENCE_MAX_CHARS = 30_000
