#! /usr/bin/env python3


import torch
import argparse

from dtransformer import DTransformer, MultiHeadAttentionParams, EmbeddingParams
from data import NotGoodDatasetTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-pth", type=str, required=True)
    parser.add_argument("--path-to-vocab", type=str, required=True)
    args = parser.parse_args()

    tokenizer = NotGoodDatasetTokenizer(args.path_to_vocab)

    embedding_params = EmbeddingParams(
        vocab_size=tokenizer.vocab_size(),
        max_sentence_size=tokenizer.max_size(),
        embedding_dimension_size=16,
    )

    multi_head_attention_params = MultiHeadAttentionParams(
        attention_dimension_size=16,
        mid_dimension_size=16,
        out_dimension_size=embedding_params.embedding_dimension_size,
        num_heads=4,
    )
    model = DTransformer(4, 16, embedding_params, multi_head_attention_params)
    model.load_state_dict(
        torch.load(args.path_to_pth, map_location=torch.device("cpu"))
    )
    model.eval()

    while True:
        text = input("Enter text: ")
        if text == "exit":
            break

        tokens = torch.tensor(tokenizer.tokenize(text)).unsqueeze(0)
        print(tokenizer.detokenize(model(tokens).argmax(dim=-1)))
