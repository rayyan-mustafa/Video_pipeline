class _G:
    def manual_seed(self, s): return self
def Generator(dev): return _G()
class inference_mode:
    def __enter__(self): return self
    def __exit__(self, *a): return False
