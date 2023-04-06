import torch

import fiftyone.zoo as foz

from pkg_resources import packaging

device = "cuda" if torch.cuda.is_available() else "cpu"

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
    dtype = torch.long
else:
    dtype = torch.int

model = foz.load_zoo_model("clip-vit-base32-torch")


def get_text_embedding(prompt, clip_model=model):
    tokenizer = clip_model._tokenizer

    # standard start-of-text token
    sot_token = tokenizer.encoder["<|startoftext|>"]

    # standard end-of-text token
    eot_token = tokenizer.encoder["<|endoftext|>"]

    prompt_tokens = tokenizer.encode(prompt)
    all_tokens = [[sot_token] + prompt_tokens + [eot_token]]

    text_features = torch.zeros(
        len(all_tokens),
        clip_model.config.context_length,
        dtype=dtype,
        device=device,
    )

    # insert tokens into feature vector
    text_features[0, : len(all_tokens[0])] = torch.tensor(all_tokens)

    # encode text
    embedding = clip_model._model.encode_text(text_features).to(device)

    # convert to list for Pinecone
    return embedding.tolist()
