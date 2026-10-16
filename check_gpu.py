"""Run this once (online) to confirm the GPU works and to download TabPFN weights. Then Airgap runs fully offline."""
import sys, time, os
os.environ.setdefault("TABPFN_NO_BROWSER", "1")      # the browser login flow crashes inside .bat on Windows
if not os.environ.get("TABPFN_TOKEN"):
    if os.path.exists("tabpfn_token.txt"):
        os.environ["TABPFN_TOKEN"] = open("tabpfn_token.txt").read().strip()
    elif not os.path.exists(".weights_ready"):
        print("TabPFN needs a one-time licence acceptance to download weights:")
        print("  1. Open https://ux.priorlabs.ai/account/licenses and accept the licence (log in or register)")
        print("  2. Open https://ux.priorlabs.ai/account and copy your API key")
        k = input("Paste API key here: ").strip()
        if k:
            open("tabpfn_token.txt", "w").write(k)
            os.environ["TABPFN_TOKEN"] = k
import numpy as np
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    print("\nFAIL: torch cannot see the GPU.\n - Update the NVIDIA driver (no CUDA Toolkit needed).\n - Make sure install.bat installed the CUDA build of torch (pip list | findstr torch should show +cu128).")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"GPU OK: {p.name}, {p.total_memory/1024**3:.1f} GB")
from tabpfn import TabPFNClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
X, y = make_classification(2000, 25, n_informative=10, random_state=0)
m = TabPFNClassifier(device="cuda")
t = time.time(); m.fit(X[:1500], y[:1500]); pr = m.predict_proba(X[1500:])[:, 1]
print(f"TabPFN on cuda: fit+predict 1500 train / 500 test in {time.time()-t:.1f}s (first call includes model load), AUC {roc_auc_score(y[1500:], pr):.3f}")
t = time.time(); m.fit(X[:1500], y[:1500]); m.predict_proba(X[1500:]); print(f"second call: {time.time()-t:.2f}s")
open(".weights_ready", "w").write("ok")
print("\nWeights are cached. run.bat will now start in offline mode.")
