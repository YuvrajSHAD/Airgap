"""TabPFN engine wrapper. Real TabPFN on CUDA by default; dev fallback only if ALLOW_FALLBACK=1."""
import os
import numpy as np

ALLOW_FALLBACK = os.environ.get("ALLOW_FALLBACK", "0") == "1"
MODEL_PATH = os.environ.get("TABPFN_MODEL_PATH", "auto")
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "20000"))
N_EST = os.environ.get("TABPFN_N_ESTIMATORS", "auto")


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def pick_device():
    forced = os.environ.get("TABPFN_DEVICE", "").strip()
    if forced:
        return forced
    t = _torch()
    return "cuda" if (t and t.cuda.is_available()) else "cpu"


def hardware():
    t = _torch()
    info = {"torch": getattr(t, "__version__", None), "cuda_available": False,
            "gpu": None, "vram_gb": None, "device": pick_device()}
    if t and t.cuda.is_available():
        p = t.cuda.get_device_properties(0)
        info.update(cuda_available=True, gpu=p.name, vram_gb=round(p.total_memory / 1024 ** 3, 1))
    return info


def tabpfn_version():
    try:
        import tabpfn
        return getattr(tabpfn, "__version__", "installed")
    except Exception:
        return None


class Engine:
    def __init__(self):
        self.device = pick_device()
        self.name = "TabPFN"
        self.fallback = False
        try:
            import tabpfn  # noqa: F401
            self._tab = True
        except Exception as e:
            if not ALLOW_FALLBACK:
                raise RuntimeError(f"tabpfn is not installed: {e}")
            self._tab = False
        if self._tab and os.environ.get("FORCE_FALLBACK") == "1" and ALLOW_FALLBACK:
            self._tab = False
        if not self._tab:
            self.name, self.fallback, self.device = "DEV FALLBACK (not TabPFN)", True, "cpu"

    # ---- classification -------------------------------------------------
    def classifier(self, cat_idx, n_train):
        if self.fallback:
            return _SkClf(cat_idx)
        from tabpfn import TabPFNClassifier
        kw = dict(device=self.device, categorical_features_indices=cat_idx or None,
                  ignore_pretraining_limits=n_train > 10000)
        if MODEL_PATH != "auto":
            kw["model_path"] = MODEL_PATH
        if N_EST != "auto":
            kw["n_estimators"] = int(N_EST)
        return _TabClf(TabPFNClassifier(**kw))

    def regressor(self, cat_idx, n_train):
        if self.fallback:
            return _SkReg(cat_idx)
        from tabpfn import TabPFNRegressor
        kw = dict(device=self.device, categorical_features_indices=cat_idx or None,
                  ignore_pretraining_limits=n_train > 10000)
        if MODEL_PATH != "auto":
            kw["model_path"] = MODEL_PATH
        if N_EST != "auto":
            kw["n_estimators"] = int(N_EST)
        return _TabReg(TabPFNRegressor(**kw))

    def cleanup(self):
        t = _torch()
        if t and t.cuda.is_available():
            t.cuda.empty_cache()


class _TabClf:
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X, y); return self
    def proba_pos(self, X):
        p = self.m.predict_proba(X)
        return p[:, list(self.m.classes_).index(1)]


class _TabReg:
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X, lo=0.1, hi=0.9):
        q = self.m.predict(X, output_type="quantiles", quantiles=[lo, 0.5, hi])
        q = np.asarray(q)  # (3, n)
        return q[1], q[0], q[2]


class _SkClf:
    def __init__(self, cat_idx): self.cat = cat_idx
    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=0,
                                                categorical_features=self.cat or None).fit(X, y)
        return self
    def proba_pos(self, X): return self.m.predict_proba(X)[:, list(self.m.classes_).index(1)]


class _SkReg:
    def __init__(self, cat_idx): self.cat = cat_idx
    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        kw = dict(max_iter=150, learning_rate=0.06, random_state=0, categorical_features=self.cat or None)
        self.mid = H(loss="quantile", quantile=0.5, **kw).fit(X, y)
        self.lo = H(loss="quantile", quantile=0.1, **kw).fit(X, y)
        self.hi = H(loss="quantile", quantile=0.9, **kw).fit(X, y)
        return self
    def predict(self, X, lo=0.1, hi=0.9):
        return self.mid.predict(X), self.lo.predict(X), self.hi.predict(X)
