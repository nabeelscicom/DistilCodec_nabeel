import torch


def calc_codebook_ppl_usage(indices: torch.Tensor, codebook_size: int, only_calc_usage: bool = False):
    indices = indices.to('cpu')
    if not only_calc_usage:
        encoding_indices = indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], codebook_size, device=indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).cpu().numpy().tolist()
    else:
        perplexity = -1.0
    unique_indices = torch.unique(indices)
    usage = unique_indices.shape[0] / codebook_size
    
    return perplexity, usage, unique_indices.tolist()


def split_group_and_residual(codes: torch.Tensor):
    code_per_codebook = []
    group_codes = codes.split(split_size=1, dim=0)
    n_groups = len(group_codes)
    for gc in group_codes:
        rc = gc.split(split_size=1, dim=-1)
        n_residuals = len(rc)
        for c in rc:
            residual_codes_t = c.squeeze().flatten().cpu()
            code_per_codebook.append(residual_codes_t)
    
    return code_per_codebook, n_groups, n_residuals


def codebbok_analysis(codes: torch.Tensor, codebook_size: int):
    code_per_codebook = split_group_and_residual(codes)
    usages = []
    ppls = []
    for codes_t in code_per_codebook:
        codebook_ppl, codebook_usage = calc_codebook_ppl_usage(indices=codes_t, 
                                                               codebook_size=codebook_size)
        ppls.append(codebook_ppl)
        usages.append(codebook_usage)
    codebook_eval = {
        "perplexity": {
            "mean": sum(ppls) / len(ppls),
            "max": max(ppls),
            "min": min(ppls)
        },
        "usage": {
            "mean": sum(usages) / len(usages),
            "max": max(usages),
            "min": min(usages)
        }
    }
    
    return codebook_eval