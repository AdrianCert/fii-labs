from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import torch

tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained("gpt2")
model: PreTrainedModel = GPT2LMHeadModel.from_pretrained("gpt2")

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


while True:
    input_text = input("$>")
    if not input_text or input_text in "q":
        break

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    target_length = 8

    while len(tokenizer.decode(input_ids[0]).split()) < target_length:
        output = model.generate(
            input_ids, max_length=input_ids.shape[1] + 1, do_sample=True
        )
        input_ids = torch.cat([input_ids, output[:, -1].unsqueeze(0)], dim=1)
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Generated sequence: {generated_text}")
