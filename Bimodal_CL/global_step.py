class GlobalStep:
    step = 0

    @classmethod
    def increment(cls):
        cls.step += 1

    @classmethod
    def set(cls, value):
        cls.step = value

    @classmethod
    def get(cls):
        return cls.step
