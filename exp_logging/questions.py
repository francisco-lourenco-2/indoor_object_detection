COMMON = [
 "What was the main goal of this experiment?",
 "What changed compared to the baseline/previous best?",
 "Any constraints, issues or edge cases worth noting?"
]
LV2_ONLY = ["What data-related changes did you introduce and why?"]
LV3_ONLY = ["Why this model/encoder and what effect did you expect?"]
LV4_ONLY = ["Which hyperparameters changed and why?"]

def ask(level:int, input_fn=input):
    qs = COMMON + (LV2_ONLY if level==2 else []) + (LV3_ONLY if level==3 else []) + (LV4_ONLY if level==4 else [])
    qs = qs[:5]
    answers = {}
    for q in qs:
        print("\n" + q)
        answers[q] = input_fn("> ").strip()
    return answers
