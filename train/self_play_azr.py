from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import json
import torch

from train.safe_executor import execute, validate_task


@dataclass
class ProgramTask:
    mode: str  # 'deduction' | 'abduction' | 'induction'
    program: str
    inputs: Any
    outputs: Any
    message: str = ""
    reward_learnability: float = 0.0
    reward_solver: float = 0.0
    introspection_score: float = 0.0


@dataclass
class TaskBuffer:
    tasks: List[ProgramTask] = field(default_factory=list)

    def sample(self, k: int) -> List[ProgramTask]:
        if not self.tasks:
            return []
        return random.sample(self.tasks, min(k, len(self.tasks)))

    def append(self, task: ProgramTask) -> None:
        self.tasks.append(task)

    def prune(self, max_size: int) -> None:
        if max_size <= 0:
            return
        if len(self.tasks) > max_size:
            # keep most recent
            self.tasks = self.tasks[-max_size:]


class SelfPlayAZRTrainer:
    def __init__(self, model, tokenizer, config: Dict[str, Any], safe_exec=execute,
                 buffers: Optional[Dict[str, TaskBuffer]] = None):
        self.model = model
        self.tok = tokenizer
        self.cfg = config or {}
        self.exec = safe_exec
        self.buffers = buffers or {m: TaskBuffer() for m in (self.cfg.get('self_play', {}).get('task_types') or ['deduction','abduction','induction'])}
        # weights
        sp = self.cfg.get('self_play') or {}
        self.num_propose = int(sp.get('num_propose', 16))
        self.num_mc = int(sp.get('num_mc_rollouts', 4))
        self.buf_cap = int(sp.get('buffer_size', 4096))
        self.l_learn = float(sp.get('lambda_learnability', 1.0))
        self.l_solve = float(sp.get('lambda_solver', 1.0))
        self.l_format = float(sp.get('lambda_format', 0.1))
        self.introspect_coeff = float(sp.get('introspection_coeff', 0.0))
        # Optional introspection engine
        self.engine = None
        try:
            if self.model is not None and self.tok is not None:
                from tina.serve import IntrospectiveEngine, EngineConfig
                hidden = int(getattr(getattr(self.model, 'config', None), 'hidden_size', 2048))
                layers = int(getattr(getattr(self.model, 'config', None), 'num_hidden_layers', 24))
                self.engine = IntrospectiveEngine(self.model, self.tok, EngineConfig(visible_cot=False), hidden_size=hidden, num_layers=layers)
        except Exception:
            self.engine = None

    def _prompt_proposer(self, mode: str, examples: List[ProgramTask]) -> str:
        parts = [f"[MODE] {mode}"]
        for ex in examples:
            parts.append("".join([
                "[EXAMPLE]\n",
                "# Program\n",
                ex.program, "\n",
                "# Inputs\n",
                json.dumps(ex.inputs), "\n",
                "# Outputs\n",
                json.dumps(ex.outputs), "\n",
            ]))
        parts.append("""[TASK]
Provide a Python program defining solve(...) and a JSON inputs list and expected outputs.
Format:
```python
<program>
```
<inputs>
<outputs>
""")
        return "\n\n".join(parts)

    def _generate_text(self, prompt: str, max_new: int = 256, temperature: float = 0.8) -> str:
        if self.model is None or self.tok is None:
            # fallback synthetic program
            return """```python\ndef solve(x):\n    return x\n```\n[1]\n1\n"""
        enc = self.tok(prompt, return_tensors='pt')
        input_ids = enc['input_ids'].to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=temperature > 0, temperature=temperature, top_p=0.95, pad_token_id=self.tok.pad_token_id, eos_token_id=getattr(self.tok, 'eos_token_id', None))
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text[len(self.tok.decode(input_ids[0], skip_special_tokens=True)):] if text else text

    @staticmethod
    def _parse_candidate(text: str) -> Optional[ProgramTask]:
        prog = None
        # find code block
        if '```' in text:
            parts = text.split('```')
            for i in range(len(parts)-1):
                if parts[i].strip().endswith('python') or parts[i].strip() == '':
                    prog = parts[i+1]
                    break
        if prog is None:
            # fallback: use entire text as program
            prog = text.strip()
        # parse inputs/outputs lines after code block
        rest = text.split('```')[-1]
        lines = [l.strip() for l in rest.splitlines() if l.strip()]
        inputs = None
        outputs = None
        for l in lines:
            try:
                v = json.loads(l)
                if inputs is None:
                    inputs = v
                else:
                    outputs = v
                    break
            except Exception:
                continue
        if inputs is None:
            inputs = [1]
            outputs = 1
        if outputs is None:
            # best-effort: execute to get outputs
            ok, out = execute(prog, inputs)
            outputs = out if ok else None
        return ProgramTask(mode='deduction', program=prog, inputs=inputs, outputs=outputs or 0, message="")

    def propose_tasks(self, mode: str, n_examples: int) -> List[ProgramTask]:
        ex = self.buffers.get(mode, TaskBuffer()).sample(n_examples)
        prompt = self._prompt_proposer(mode, ex)
        text = self._generate_text(prompt)
        task = self._parse_candidate(text)
        if task is None:
            return []
        task.mode = mode
        # validate determinism + basic execution and estimate learnability via MC
        successes = 0
        for _ in range(max(1, self.num_mc)):
            ok, _ = execute(task.program, task.inputs)
            successes += 1 if ok else 0
        task.reward_learnability = float(successes / max(1, self.num_mc))
        # Introspection guidance: score based on plan/budget/confidence
        try:
            if self.engine is not None and self.introspect_coeff > 0.0:
                prompt = self._prompt_solver(task)
                enc = self.tok(prompt, return_tensors='pt')
                input_ids = enc['input_ids'].to(next(self.model.parameters()).device)
                _ = self.engine._estimate_budget(input_ids)
                st = getattr(self.engine, 'last_stats', {}) or {}
                plan_label = st.get('plan_label') or st.get('plan')
                conf = float(st.get('confidence') or 0.0)
                budget = int(st.get('think_budget') or 0)
                min_think = int(getattr(self.engine.cfg, 'min_think_tokens', 8))
                score = 0.0
                if isinstance(plan_label, str):
                    if plan_label == 'stop':
                        score += 1.0
                    elif plan_label == 'verify':
                        score += 0.5
                    else:
                        score += 0.2
                else:
                    score += 0.2
                if conf < 0.5:
                    score += 0.5
                if budget <= (min_think + 4):
                    score += 0.3
                # store and apply modulation
                task.introspection_score = float(score)
                task.reward_learnability = float(task.reward_learnability * (1.0 + self.introspect_coeff * score))
        except Exception:
            pass
        return [task]

    def _prompt_solver(self, task: ProgramTask) -> str:
        return """Solve the following task by returning only the final answer as a JSON value.\n
```python
%s
```\nInputs:\n%s\nAnswer:\n""" % (task.program, json.dumps(task.inputs))

    def solve_tasks(self, tasks: List[ProgramTask]) -> List[ProgramTask]:
        for t in tasks:
            # ground truth output via safe exec
            ok, gt = execute(t.program, t.inputs)
            if not ok:
                t.reward_solver = 0.0
                continue
            if self.model is None or self.tok is None:
                pred = gt
            else:
                prompt = self._prompt_solver(t)
                gen = self._generate_text(prompt, max_new=64, temperature=0.0)
                # parse first JSON on the line, else try int cast
                try:
                    pred = json.loads(gen.splitlines()[0].strip())
                except Exception:
                    try:
                        pred = int(gen.strip().split()[0])
                    except Exception:
                        pred = None
            t.outputs = gt
            t.reward_solver = 1.0 if (pred == gt) else 0.0
        return tasks

    def update_policy(self, tasks: List[ProgramTask]) -> Dict[str, float]:
        """
        Compute combined reward R = λ1*learnability + λ2*solver − λ3*format_penalty and apply a simple
        policy gradient by maximizing log-likelihood of the correct answer under the solver prompt.
        This is a simplified surrogate for TRR++ suitable for tests.
        """
        if self.model is None or self.tok is None:
            return {"loss": 0.0}
        device = next(self.model.parameters()).device
        R: List[float] = []
        losses: List[torch.Tensor] = []
        for t in tasks:
            # format penalty if program missing solve()
            fmt_pen = 0.0 if ('def solve' in t.program) else 1.0
            # Apply introspection-weighted learnability
            r_learn = float(t.reward_learnability) * (1.0)  # already modulated in propose_tasks if engine present
            r = self.l_learn * r_learn + self.l_solve * float(t.reward_solver) - self.l_format * fmt_pen
            R.append(r)
            # teacher-forcing target = JSON of ground truth
            prompt = self._prompt_solver(t)
            target = json.dumps(t.outputs)
            enc = self.tok(prompt, return_tensors='pt')
            lab = self.tok(target, return_tensors='pt', add_special_tokens=False)['input_ids'][0].to(device)
            inp = enc['input_ids'].to(device)
            with torch.no_grad():
                base_len = inp.shape[1]
            out = self.model(input_ids=inp, labels=None)
            # Next-token loss over the answer span appended to input
            # Build concatenated label sequence after base_len
            # For surrogate: compute LM loss on target given prompt
            concat = torch.cat([inp[0], lab], dim=0)[None, :]
            labels = concat.clone(); labels[:, :base_len] = -100
            logits = self.model(input_ids=concat, labels=None).logits
            B, T, V = logits.shape
            ce = torch.nn.functional.cross_entropy(logits.view(B*T, V), labels.view(B*T), ignore_index=-100)
            losses.append(ce * (-r))  # maximize reward → minimize negative reward-weighted CE
        if losses:
            loss = torch.stack(losses).mean()
            opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            return {"loss": float(loss.item()), "avg_reward": float(sum(R)/max(1,len(R)))}
        return {"loss": 0.0}

    def train_loop(self, steps: int = 10) -> Dict[str, Any]:
        modes = self.cfg.get('self_play', {}).get('task_types') or ['deduction','abduction','induction']
        stats: Dict[str, Any] = {}
        for it in range(int(steps)):
            for m in modes:
                new_tasks = self.propose_tasks(m, n_examples=2)
                solved = self.solve_tasks(new_tasks)
                upd = self.update_policy(solved)
                # push into buffers
                buf = self.buffers.setdefault(m, TaskBuffer())
                for t in solved:
                    buf.append(t)
                buf.prune(self.buf_cap)
                # average introspection score on this batch
                try:
                    intros_mean = sum(getattr(t, 'introspection_score', 0.0) for t in solved) / max(1, len(solved))
                except Exception:
                    intros_mean = 0.0
                stats[f"iter_{it}_{m}"] = {"buffer": len(buf.tasks), "introspection_score_mean": float(intros_mean), **upd}
        return stats
