from hypothesis import settings

settings.register_profile("topk_softmax", deadline=None, max_examples=60, derandomize=True)
settings.load_profile("topk_softmax")
