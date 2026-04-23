# =========================
# Helper
# =========================
def create_dummy_face():
    """Creates a valid grayscale face image"""
    return np.zeros((48, 48), dtype=np.uint8)


# =========================
# Model loading
# =========================
def test_get_model_lazy_loading(monkeypatch):
    """Ensure model loads only when needed"""

    import src.inference.predict as p

    # reset state
    p.model = None
