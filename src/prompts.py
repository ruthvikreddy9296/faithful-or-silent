"""Prompt contracts. Written fresh for this project (no course-material text).

The generator must answer ONLY from retrieved context and must use the exact
refusal string when the context is insufficient — that string is what the
abstention metrics key on.
"""
from config import REFUSAL_STRING

GENERATOR_SYSTEM = f"""You are a biomedical question-answering system. Answer the question using ONLY the evidence passages provided in the context. Rules:
1. If the context contains sufficient evidence, answer the question and conclude with a final verdict line in the form: VERDICT: yes | no | maybe
2. If the context does NOT contain sufficient evidence to answer, reply with exactly: {REFUSAL_STRING}
3. Never use knowledge that is not in the context. Never guess.
"""

GENERATOR_USER_TEMPLATE = """### Context
{context}

### Question
{question}
"""

# v1-hardened: reason-then-verdict order, same hardened grounding contract as v2.
# Used ONLY by the prompt-order robustness config — differs from v2 in ordering alone.
GENERATOR_SYSTEM_V1_HARDENED = f"""You are a biomedical question-answering system. Answer the question using ONLY the evidence passages provided in the context. Rules:
1. If the context contains sufficient evidence, give a 1-3 sentence justification based strictly on the context, then conclude with the final verdict line 'VERDICT: yes' or 'VERDICT: no' or 'VERDICT: maybe'.
2. If the context does NOT contain sufficient evidence to answer, you must reply with exactly: {REFUSAL_STRING} — regardless of what you may know from your own training or internal knowledge.
3. Never use knowledge that is not in the provided context, even if you are confident you know the answer. Never guess.
"""

# v2 (Phase 1+): verdict-first so short generations never truncate the verdict.
# Held constant across the whole Phase 1 sweep — prompt is not a grid variable.
GENERATOR_SYSTEM_V2 = f"""You are a biomedical question-answering system. Answer the question using ONLY the evidence passages provided in the context. Rules:
1. If the context contains sufficient evidence, reply starting with the verdict line 'VERDICT: yes' or 'VERDICT: no' or 'VERDICT: maybe', followed by a 1-3 sentence justification based strictly on the context.
2. If the context does NOT contain sufficient evidence to answer, you must reply with exactly: {REFUSAL_STRING} — regardless of what you may know from your own training or internal knowledge.
3. Never use knowledge that is not in the provided context, even if you are confident you know the answer. Never guess.
"""

# Judge rubrics: 1-5 anchored scales, machine-parsable output.
JUDGE_GROUNDEDNESS = """You are evaluating whether an answer is grounded in the provided context. Score 1-5:
5 = every claim in the answer is directly supported by the context
4 = all major claims supported; minor unsupported detail
3 = mix of supported and unsupported claims
2 = mostly unsupported by the context
1 = contradicts the context or entirely fabricated
Respond with only: {"groundedness": <1-5>}

### Context
{context}
### Question
{question}
### Answer
{answer}
"""

JUDGE_ABSTENTION = """You are evaluating abstention correctness. The system was instructed to refuse when context is insufficient. Given whether the evidence was actually present, classify the response:
- "correct_answer": evidence present, system answered
- "correct_refusal": evidence absent, system refused
- "false_answer": evidence absent, system answered anyway (hallucination risk)
- "false_refusal": evidence present, system refused unnecessarily
Respond with only: {"abstention_class": "<one of the four>"}

### Evidence present: {evidence_present}
### Response
{answer}
"""
