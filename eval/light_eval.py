from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _items() -> List[Dict[str, Any]]:
    """Return a small set of logic/arith/syllogism prompts with simple graders."""
    items: List[Dict[str, Any]] = []
    # Parity
    items.append({
        'prompt': 'Is 47 even or odd? Reply with just one word.',
        'check': lambda s: ('odd' in s.lower()) and ('even' not in s.lower()),
    })
    items.append({
        'prompt': 'Is 124 even or odd? Reply with just one word.',
        'check': lambda s: ('even' in s.lower()) and ('odd' not in s.lower()),
    })
    # 2-digit arithmetic (tiny traps)
    items.append({
        'prompt': 'Compute 19 + 24. Give only the number.',
        'check': lambda s: '43' in ''.join(c for c in s if c.isdigit()),
    })
    items.append({
        'prompt': 'Compute 35 - 17. Give only the number.',
        'check': lambda s: '18' in ''.join(c for c in s if c.isdigit()),
    })
    # Syllogism (simple)
    items.append({
        'prompt': 'All birds can fly. Tweety is a bird. Can Tweety fly? Answer Yes or No.',
        'check': lambda s: s.strip().lower().startswith('yes'),
    })
    items.append({
        'prompt': 'No reptiles are warm-blooded. A snake is a reptile. Is a snake warm-blooded? Answer Yes or No.',
        'check': lambda s: s.strip().lower().startswith('no'),
    })
    # Short word math
    items.append({
        'prompt': 'Tom has 3 red marbles and 2 blue marbles. How many marbles in total?',
        'check': lambda s: '5' in ''.join(c for c in s if c.isdigit()),
    })
    items.append({
        'prompt': 'How many letters are in the word "apple"?',
        'check': lambda s: '5' in ''.join(c for c in s if c.isdigit()),
    })
    return items


def run(engine, tokenizer, *, visible_cot: bool = False, temperature: float = 0.2, top_p: float = 0.95,
        max_new_tokens: int = 64) -> Dict[str, Any]:
    """
    Evaluate a handful of tiny prompts using the current engine + DecodingController.
    Returns {'results': [...], 'accuracy': float}
    Each result contains: prompt, text, policy, expert_weights, confidence, think_tokens_used, success.
    """
    items = _items()
    results: List[Dict[str, Any]] = []
    ok = 0
    for ex in items:
        prompt = ex['prompt']
        # Prepare messages for engine
        messages = [{"role": "user", "content": prompt}]
        text = engine.generate_cot(messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                   repetition_penalty=1.1, ignore_eos=False, stream=False, style_tag=None)
        # Collect policy/telemetry from engine
        st = dict(engine.last_stats) if isinstance(getattr(engine, 'last_stats', None), dict) else {}
        policy = {
            'temperature': st.get('gen_temperature'),
            'top_p': st.get('gen_top_p'),
            'rep_penalty': st.get('gen_rep_penalty'),
            'B_max': st.get('B_max'),
            'think_ratio': st.get('think_ratio'),
        }
        ew = st.get('expert_weights')
        conf = st.get('confidence')
        used = st.get('think_tokens_used') or st.get('think_tokens')
        # Grade
        try:
            success = bool(ex['check'](str(text)))
        except Exception:
            success = False
        ok += 1 if success else 0
        results.append({
            'prompt': prompt,
            'text': text,
            'policy': policy,
            'expert_weights': ew,
            'confidence': conf,
            'think_tokens_used': used,
            'success': success,
        })
    acc = float(ok) / float(max(1, len(items)))
    return {'results': results, 'accuracy': acc}

