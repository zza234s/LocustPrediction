# -*- coding: utf-8 -*-
"""
GRPO reward functions for locust outbreak prediction - VERL Integration
Adapted for RateLimitedRewardManager with rate limiting support
Enhanced with original prompt context for LLM-as-judge
"""
import re
from dataclasses import dataclass
from typing import Dict, Optional

from openai import AsyncOpenAI



# =========================
# Config
# =========================

@dataclass
class RewardWeights:
    alpha: float = 1.0      # R_cls weight
    gamma: float = 0.0      # R_cot weight  
    delta: float = 0.0      # R_format weight
    epsilon: float = 0.0    # R_verdict weight (新增)
    
    judge_model: str = "qwen-plus"
    enable_r_cot: bool = False
    enable_r_verdict: bool = False

REQUIRED_SUIT_TAGS = ["veg", "water", "soil_moisture", "temperature"]

label_map = {0: "0", 1: "1"}
label_dict = {v.lower(): k for k, v in label_map.items()}

# Async client
async_client = AsyncOpenAI(
    api_key="sk-d9534dba69b84e84ad5136a0f8125040",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# =========================
# Utilities
# =========================

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_tag_content(text: str, tag: str) -> str:
    if not text:
        return ""
    matches = re.findall(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_prediction(text: str) -> str:
    t = text or ""
    matches = re.findall(r"<class>\s*([^<]+?)\s*</class>", t, flags=re.IGNORECASE)
    if matches:
        candidate = _normalize_text(matches[-1])
        if candidate in ("0", "1"):
            return candidate
    return None


def safe_label_id(gt: str) -> int:
    g = _normalize_text(gt)
    if g in label_dict:
        return label_dict[g]
    return -1


def parse_suit_tag(text: str, tag: str) -> str:
    v = extract_tag_content(text, tag)
    return _normalize_text(v)


# =========================
# R_cls: Classification Correctness
# =========================

def compute_R_cls(output_text: str, ground_truth: str) -> tuple[float, int, int]:
    """Classification correctness reward."""
    pred_str = extract_prediction(output_text)
    label_id = safe_label_id(ground_truth)
    
    if pred_str is None:
        return 0.0, -1, label_id
    
    pred_id = safe_label_id(pred_str)
    if pred_id not in (0, 1):
        return 0.0, -1, label_id
    
    r_cls = 1.0 if pred_id == label_id else 0.0
    return r_cls, pred_id, label_id


# =========================
# R_cot: Chain-of-Thought Quality (Enhanced)
# =========================

async def compute_R_cot(
    output_text: str, 
    weights: RewardWeights,
    original_prompt: Optional[str] = None
) -> float:
    """
    Enhanced chain-of-thought quality scoring with original prompt context.
    
    Args:
        output_text: Model's complete output
        weights: Reward configuration
        original_prompt: The original task prompt (optional but recommended)
    """
    if not weights.enable_r_cot:
        return 0.0
    
    thinking_text = extract_tag_content(output_text, "thinking")
    if not thinking_text:
        return 0.0
    
    pred_str = extract_prediction(output_text)
    pred_label = "High Risk (Class 1)" if pred_str == "1" else "Low Risk (Class 0)" if pred_str == "0" else "Unknown"
    
    # 提取prompt中的关键信息
    prompt_context = ""
    # if original_prompt:
    #     prompt_context = original_prompt

    # 构建增强的评估prompt
    evaluation_prompt = f"""You are an expert evaluator for a Desert Locust Presence Risk Prediction task. 

    **Model's Reasoning to Evaluate:**
    {thinking_text}

    **Final Prediction:** {pred_label}

    **Evaluation Task:**
    Assess the quality of the reasoning based on the following criteria:

    1. **Scientific Accuracy**: 
    - Are the ecological interpretations scientifically sound?
    - Are threshold judgments reasonable (e.g., NDVI=0 → insufficient vegetation)?

    2. **Data Utilization**:
    - Did the model analyze the 12-month time series comprehensively?
    - Were trends/patterns identified (not just stating final month values)?

    3. **Logical Consistency**:
    - Does the [Summary] conclusion logically follow from the [Ecological Preconditions]?
    - Are the [Per-Aspect Suitability Verdicts] consistent with the final prediction?

    **Specific Red Flags**:
    - Summary says "insufficient vegetation, inadequate water, unsuitable temperature" BUT concludes "very high risk" → Major contradiction!
    - Missing analysis of any factors

    **Scoring Scale (0-10):**
    - 9-10: Excellent - scientifically sound, data-driven, perfectly consistent
    - 7-8: Good - minor inconsistencies or incomplete data analysis
    - 5-6: Fair - some contradictions or superficial analysis
    - 3-4: Poor - major logical contradictions (e.g., negative verdicts → high risk)
    - 1-2: Very Poor - ignores data or completely illogical
    - 0: Failed - no meaningful reasoning

    Return ONLY a single integer from 0 to 10.
    """
    try:
        resp = await async_client.chat.completions.create(
            model=weights.judge_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
        )
        result = (resp.choices[0].message.content or "").strip()
        m = re.search(r"(\d+\.?\d*)", result)
        if not m:
            return 0.0
        
        score = float(m.group(1))
        if 1.0 < score <= 10.0:
            return score / 10.0
        return score if 0.0 <= score <= 1.0 else 0.0
        
    except Exception as e:
        print(f"Error in R_cot scoring: {e}")
        return 0.0


# =========================
# R_format: Format Correctness
# =========================

def compute_R_format(output_text: str) -> tuple[float, float, float]:
    """Format correctness reward."""
    has_thinking = len(extract_tag_content(output_text, "thinking")) > 0
    pred_str = extract_prediction(output_text)
    
    F = 1.0 if (has_thinking and pred_str in ("0", "1")) else 0.0
    
    cov = sum(1 for tag in REQUIRED_SUIT_TAGS if parse_suit_tag(output_text, tag))
    C = 1.0 if cov == 4.0 else 0.0
    
    R_format = 0.5 * F + 0.5 * C
    return float(R_format), float(F), float(C)




# =========================
# R_verdict: 极简惩罚法
# =========================

# 最小关键词集合
BAD_VERDICT_WORDS = {"insufficient", "inadequate", "unsuitable", "too_wet", "too_dry", "marginal"}
GOOD_VERDICT_WORDS = {"sufficient", "adequate", "suitable"}


def compute_R_verdict(output_text: str) -> float:
    """
    极简verdict一致性检查 - 只惩罚最极端的矛盾
    
    Returns:
        1.0: 一致或可接受
        0.0: 明显矛盾（4个全负面但预测高风险，或4个全正面但预测低风险）
    """
    pred_str = extract_prediction(output_text)
    if pred_str not in ("0", "1"):
        return 1.0  # 没有有效预测，不惩罚
    
    pred_id = int(pred_str)
    
    # 提取4个verdict
    verdicts = [
        parse_suit_tag(output_text, "veg"),
        parse_suit_tag(output_text, "water"),
        parse_suit_tag(output_text, "soil_moisture"),
        parse_suit_tag(output_text, "temperature"),
    ]
    
    # 检查有多少个verdict包含负面词/正面词
    bad_count = sum(1 for v in verdicts if any(word in v for word in BAD_VERDICT_WORDS))
    good_count = sum(1 for v in verdicts if any(word in v for word in GOOD_VERDICT_WORDS))
    
    # 极端矛盾检查
    if bad_count == 4 and pred_id == 1:
        return 0.0  # 4个全负面但预测高风险
    
    if good_count == 4 and pred_id == 0:
        return 0.0  # 4个全正面但预测低风险
    
    return 1.0  # 其他情况都可以接受 

# =========================
# Main VERL Integration Function
# =========================

async def compute_score_locust(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    weights: Optional[RewardWeights] = None,
) -> Dict[str, float]:
    """
    Main reward function for VERL RateLimitedRewardManager.
    
    Args:
        data_source: Data source identifier (e.g., "locust")
        solution_str: Model output text
        ground_truth: Ground truth label
        extra_info: Additional metadata (should contain 'original_prompt' key)
        weights: Reward configuration
    
    Returns:
        Dictionary with 'score' and additional metrics
    """
    # breakpoint()
    w = RewardWeights()
    
    # Extract original prompt if available
    original_prompt =extra_info['raw_prompt'][0]['content'][1]['text']
    
    # 1) Classification correctness
    R_cls, pred_id, label_id = compute_R_cls(solution_str, ground_truth)
    
    # 2) Chain-of-thought quality (with prompt context)
    R_cot =0.0
    if w.enable_r_cot:
        R_cot = await compute_R_cot(solution_str, w, original_prompt)
    
    # 3) Format correctness
    R_format, F_flag, C_cov = compute_R_format(solution_str)
    
    # 4) Verdict consistency
    R_verdict = 0.0
    if w.enable_r_verdict:
        R_verdict = compute_R_verdict(solution_str)

    # prediction incorrect (or invalid) -> no "explanation" rewards
    if R_cls <= 0.0:
        R_cot = 0.0
        R_verdict = 0.0

    # Total reward
    total = (w.alpha * R_cls + 
             w.gamma * R_cot + 
             w.delta * R_format + 
             w.epsilon * R_verdict)
    
    return {
        "score": float(total),
        "R_cls": float(R_cls),
        "R_cot": float(R_cot),
        "R_format": float(R_format),
        "R_verdict": float(R_verdict),
        "F_flag": float(F_flag),
        "C_cov": float(C_cov),
        "pred": float(pred_id if pred_id in (0, 1) else -1),
        "label": float(label_id if label_id in (0, 1) else -1),
    }



