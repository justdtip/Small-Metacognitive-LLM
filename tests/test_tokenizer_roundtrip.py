from tokenizer_utils import ensure_reasoning_tokens

class Tok:
    def __init__(self):
        self.unique_no_split_tokens=[]
        self.vocab={"-":10, ".":11, ",":12}
    def convert_tokens_to_ids(self, t): return self.vocab.get(t, 0)
    def add_tokens(self, toks, special_tokens=False):
        for t in toks: self.vocab[t]=len(self.vocab)+1

def test_reasoning_tokens_added_and_nosplit():
    tok = Tok()
    ids = ensure_reasoning_tokens(tok, model=None)
    for t in ["<think>","</think>","<answer>","</answer>"]:
        assert t in tok.vocab
        assert t in tok.unique_no_split_tokens

