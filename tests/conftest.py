from hypothesis import settings

settings.register_profile("softk", deadline=None, max_examples=60, derandomize=True)
settings.load_profile("softk")
