import torch

# Simulate embedding dims from file settings
EMB_DIM = 64
emb_dim = EMB_DIM
graph_out = emb_dim * 2
GLOBAL_DIM = 10
FP_DIM = 128

# Single-sample case
emb1 = torch.randn(graph_out).unsqueeze(0)  # (1, graph_out)
emb2 = torch.randn(graph_out).unsqueeze(0)
# per-graph features as 1D (old behavior that caused error)
gf1 = torch.randn(GLOBAL_DIM)
gf2 = torch.randn(GLOBAL_DIM)
fp1 = torch.randn(FP_DIM)
fp2 = torch.randn(FP_DIM)

# Ensure unsqueeze logic from SiameseGNN.forward
if gf1.dim() == 1:
    gf1 = gf1.unsqueeze(0)
if gf2.dim() == 1:
    gf2 = gf2.unsqueeze(0)
if fp1.dim() == 1:
    fp1 = fp1.unsqueeze(0)
if fp2.dim() == 1:
    fp2 = fp2.unsqueeze(0)

combined = torch.cat([
    emb1, emb2,
    (emb1 - emb2).abs(),
    emb1 * emb2,
    gf1, gf2,
    fp1, fp2,
], dim=1)
print('single combined shape:', combined.shape)

# Batched case
B = 4
emb1b = torch.randn(B, graph_out)
emb2b = torch.randn(B, graph_out)
gf1b = torch.randn(B, GLOBAL_DIM)
gf2b = torch.randn(B, GLOBAL_DIM)
fp1b = torch.randn(B, FP_DIM)
fp2b = torch.randn(B, FP_DIM)

# unsqueeze checks (they already are 2D)
if gf1b.dim() == 1:
    gf1b = gf1b.unsqueeze(0)
if gf2b.dim() == 1:
    gf2b = gf2b.unsqueeze(0)
if fp1b.dim() == 1:
    fp1b = fp1b.unsqueeze(0)
if fp2b.dim() == 1:
    fp2b = fp2b.unsqueeze(0)

combined_b = torch.cat([
    emb1b, emb2b,
    (emb1b - emb2b).abs(),
    emb1b * emb2b,
    gf1b, gf2b,
    fp1b, fp2b,
], dim=1)
print('batch combined shape:', combined_b.shape)
