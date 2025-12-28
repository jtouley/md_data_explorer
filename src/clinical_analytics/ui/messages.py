"""
UI Messages - Centralized message constants.

Simple module-level constants for consistent UI messaging.
Keep it lightweight - no classes or complex structures.
"""

# Low-confidence feedback messages
LOW_CONFIDENCE_WARNING = "⚠️ I'm not completely sure about this analysis. Please review and confirm:"
SEMANTIC_LAYER_NOT_READY = "Semantic layer not ready. Please wait..."
COLLISION_SUGGESTION_WARNING = "⚠️ Some terms matched multiple columns. Please select the correct one:"

# Analysis execution messages
CONFIRM_AND_RUN = "✅ Confirm and Run Analysis"
START_OVER = "🔄 Start Over"
CLEAR_RESULTS = "🗑️ Clear Results"
RESULTS_CLEARED = "Results cleared!"

# Dataset selection messages
NO_DATASETS_AVAILABLE = "No datasets available. Please upload data first."

# Natural language query messages
NL_QUERY_UNAVAILABLE = "Natural language queries are only available for datasets with semantic layers."
NL_QUERY_ERROR = "Error parsing natural language query: {error}"

# Analysis result messages
ANALYSIS_RUNNING = "Running analysis..."
UNDERSTANDING_QUESTION = "Understanding your question..."
