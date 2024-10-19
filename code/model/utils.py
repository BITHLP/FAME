import torch


def remove_prefix(prefix, main_string):
    if isinstance(main_string, list) and isinstance(prefix, list):
        return [remove_prefix(p, s) for p, s in zip(prefix, main_string)]
    elif isinstance(main_string, list):
        return [remove_prefix(prefix, mains) for mains in main_string]
    if isinstance(main_string, dict):
        main_string = main_string['content']
    if main_string.startswith(prefix):
        main_string = main_string[len(prefix):]
    return main_string


def remove_prefix_with_len(prefix, main_string):
    if isinstance(main_string, list) and isinstance(prefix, list):
        return [remove_prefix(p, s) for p, s in zip(prefix, main_string)]
    elif isinstance(main_string, list):
        return [remove_prefix(prefix, mains) for mains in main_string]
    if isinstance(main_string, dict):
        main_string = main_string['content']
    main_string = main_string[len(prefix):]
    return main_string


def perplexity(
    model,
    tok,
    text,
    max_input_length: int = None,
):
    if isinstance(text,list):
        return [
            perplexity(model,tok,text_i,max_input_length)
            for text_i in text
        ]
    return -1
    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(
        logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()