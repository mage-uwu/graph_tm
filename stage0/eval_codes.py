"""Evaluate a code-target model: per-bit accuracy and nearest-code token recovery.
usage: python3 eval_codes.py <test.gtmd> <sums.i32>   (sums from `gtm score --sums`)"""
import sys
import numpy as np
from gtmcore import Dataset

d = Dataset.load(sys.argv[1])
s = np.fromfile(sys.argv[2], dtype="<i4").reshape(d.n_graphs, d.n_outputs)
pred = (s >= 0).astype(np.int32)
codes, tok = np.unique(d.Y, axis=0, return_inverse=True)
nearest = np.abs(pred[:, None, :] - codes[None]).sum(2).argmin(1)
print(f"per-bit acc {(pred == d.Y).mean():.4f} | nearest-code token acc {(nearest == tok.ravel()).mean():.4f} "
      f"(chance {1 / len(codes):.3f}) | all-bits exact {(pred == d.Y).all(1).mean():.4f} "
      f"| predicted-1 rate {pred.mean():.3f} (true {d.Y.mean():.3f})")
