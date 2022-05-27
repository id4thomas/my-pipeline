import torch

def build_input_ref_pair(text, tokenizer):
    ref_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    # Input
    # input_ids = tokenizer(text, add_special_tokens = True, padding = "max_length", max_length = 128, return_tensors="pt")
    input_ids = tokenizer([text], add_special_tokens = True)
    input_ids = torch.tensor(input_ids["input_ids"])
    
    # Reference (Baseline) Input: fill with [pad]
    text_ids = tokenizer(text, add_special_tokens = False)
    ref_input = [bos_token_id]+ [ref_token_id]*len(text_ids["input_ids"]) + [eos_token_id]
    ref_input_ids = torch.tensor([ref_input])
    # ref_input = [bos_token_id]+ [ref_token_id]*(128-2) + [eos_token_id]
    # ref_input_ids = torch.tensor([ref_input])

    return input_ids, ref_input_ids