import torch


class ClassEmbeddingBank:
    def __init__(self, num_classes, embed_dim=None, momentum=0.9, device="cpu"):
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.device = torch.device(device)
        self.mean = None
        self.var = None
        self.count = torch.zeros(num_classes, device=self.device)
        if embed_dim is not None:
            self._init_tensors(embed_dim)

    def _init_tensors(self, embed_dim):
        self.embed_dim = embed_dim
        self.mean = torch.zeros(self.num_classes, embed_dim, device=self.device)
        self.var = torch.ones(self.num_classes, embed_dim, device=self.device)

    @torch.no_grad()
    def update(self, embeddings, labels):
        embeddings = embeddings.detach()
        labels = labels.detach()
        if self.mean is None:
            self._init_tensors(embeddings.size(1))
        for cls in labels.unique():
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue
            cls_emb = embeddings[labels == cls]
            if cls_emb.numel() == 0:
                continue
            batch_mean = cls_emb.mean(dim=0)
            batch_var = cls_emb.var(dim=0, unbiased=False) if cls_emb.size(0) > 1 else torch.zeros_like(batch_mean)
            if self.count[cls_idx] == 0:
                self.mean[cls_idx] = batch_mean
                self.var[cls_idx] = batch_var
            else:
                self.mean[cls_idx] = self.momentum * self.mean[cls_idx] + (1.0 - self.momentum) * batch_mean
                self.var[cls_idx] = self.momentum * self.var[cls_idx] + (1.0 - self.momentum) * batch_var
            self.count[cls_idx] += cls_emb.size(0)

    def lookup(self, labels):
        labels = labels.to(self.device)
        if self.mean is None:
            return None, None, None
        valid = self.count[labels].to(dtype=torch.bool)
        return self.mean[labels], self.var[labels], valid

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "momentum": self.momentum,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state):
        self.num_classes = state["num_classes"]
        self.embed_dim = state["embed_dim"]
        self.momentum = state.get("momentum", self.momentum)
        self.mean = state["mean"].to(self.device) if state["mean"] is not None else None
        self.var = state["var"].to(self.device) if state["var"] is not None else None
        self.count = state["count"].to(self.device)
