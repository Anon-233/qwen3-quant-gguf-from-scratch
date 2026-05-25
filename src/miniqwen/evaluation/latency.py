from time import perf_counter


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *_args):
        self.end = perf_counter()
        self.seconds = self.end - self.start
