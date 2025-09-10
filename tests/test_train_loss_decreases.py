import pytest
torch = pytest.importorskip("torch")

from train.runner import TinyLM, DummyTok
from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks
from train.data import pad_and_stack
from train.losses import compute_losses

def test_synthetic_training_loss_decreases():
    tok = DummyTok()
    ensure_reasoning_tokens(tok)
    text = "<think> plan steps </think> <answer> final answer </answer>"
    ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
    input_ids = batch["input_ids"]

    model = TinyLM(vocab_size=max(tok.vocab.values()) + 16, hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    def step():
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        loss_mask_t = batch["loss_mask"].bool()
        lab = torch.where(loss_mask_t, labels, torch.full_like(labels, -100))
        logits = model(input_ids)
        losses = compute_losses(logits, lab, weights={"answer_ce": 1.0})
        opt.zero_grad(); losses["total"].backward(); opt.step()
        return float(losses["total"].item())

    loss1 = step()
    loss2 = step()
    # Loss should not increase; usually decreases on second step
    assert loss2 <= loss1 + 1e-6

