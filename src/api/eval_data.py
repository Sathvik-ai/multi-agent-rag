"""
Evaluation Dataset for the /evaluate endpoint.
These are ground-truth question-answer pairs that we can use to systematically
test our RAG pipeline's accuracy and correctness.

Format: { "question": str, "expected_keywords": list[str], "source_hint": str }
"""

EVAL_DATASET = [
    {
        "question": "What is the Transformer model architecture based on?",
        "expected_keywords": ["attention", "self-attention", "encoder", "decoder"],
        "source_hint": "attention_is_all_you_need"
    },
    {
        "question": "How many attention heads does the Transformer use in the base model?",
        "expected_keywords": ["8", "multi-head", "heads"],
        "source_hint": "attention_is_all_you_need"
    },
    {
        "question": "What memory bottleneck does FlashAttention solve?",
        "expected_keywords": ["HBM", "SRAM", "memory", "IO", "bandwidth"],
        "source_hint": "flash_attention"
    },
    {
        "question": "What does RAG stand for and what problem does it solve?",
        "expected_keywords": ["retrieval-augmented generation", "knowledge", "parametric"],
        "source_hint": "rag_original"
    },
    {
        "question": "How does ReAct combine reasoning and acting?",
        "expected_keywords": ["reasoning", "acting", "interleaved", "traces", "actions"],
        "source_hint": "react"
    },
    {
        "question": "Which drug in the medical trials dataset has the highest efficacy score?",
        "expected_keywords": ["BreathMax", "0.95", "Asthma"],
        "source_hint": "medical_trials"
    },
]
