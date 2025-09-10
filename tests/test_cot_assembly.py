from tina.serve import IntrospectiveEngine

def test_assemble_hidden_only_answer():
    think = "<think>some chain</think>"
    answer = "<answer>final answer</answer>"
    out = IntrospectiveEngine.assemble_cot_output(think, answer, visible_cot=False)
    assert out == "final answer"
    assert "<think>" not in out and "</think>" not in out
    assert "<answer>" not in out and "</answer>" not in out

def test_assemble_visible_concat():
    think = "<think>t</think>"
    answer = "<answer>a</answer>"
    out = IntrospectiveEngine.assemble_cot_output(think, answer, visible_cot=True)
    assert out == (think + answer)

