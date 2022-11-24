import torch
from torch import nn


def update_queries(model, config, device):
    if not config.transformer.get("infer_queries", None):
        return model

    hidden_dim = config.transformer.hidden_dim
    num_queries = config.transformer.num_queries
    num_queries_infer = config.transformer.infer_queries.num_queries
    if num_queries == num_queries_infer:
        return model

    query_embed = model.query_embed
    pad_method = config.transformer.infer_queries.pad_method

    if num_queries > num_queries_infer:
        query_embed_infer = nn.Embedding(num_queries_infer, hidden_dim).to(device)
        query_embed_infer.weight.data = query_embed.weight[:num_queries_infer]
        model.query_embed = query_embed_infer
        return model

    if pad_method == "rand":
        query_embed_infer = nn.Embedding(num_queries_infer, hidden_dim).to(device)
        query_embed_infer.weight.data[:num_queries] = query_embed.weight
        query_embed_infer.weight.data[num_queries:] = torch.rand(
            (num_queries_infer - num_queries, hidden_dim)
        ).to(device)

    elif pad_method == "clone":
        query_embed_infer = nn.Embedding(num_queries_infer, hidden_dim).to(device)
        num_repeat = num_queries_infer // num_queries
        num_left = num_queries_infer % num_queries
        query_embed_infer_weight = torch.cat(
            [query_embed.weight] * num_repeat + [query_embed.weight[:num_left]], dim=0
        )
        query_embed_infer.weight.data = query_embed_infer_weight

    model.query_embed = query_embed_infer
