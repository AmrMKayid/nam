def mae(logits, targets):
    return ((logits.view(-1) - targets.view(-1)).abs().sum() / logits.numel()).item()


def accuracy(logits, targets):
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()
