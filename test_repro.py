import paddle
import functools

print(f"Paddle version: {paddle.__version__}")

try:
    place = paddle.CPUPlace()
    print(f"Place: {place}, Hash: {hash(place)}")

    @functools.lru_cache(maxsize=128)
    def cached_func(dim, base, device):
        print(f"Computing for {dim}, {base}, {device}")
        return paddle.zeros([dim]).to(device)

    r1 = cached_func(10, 10000.0, place)
    r2 = cached_func(10, 10000.0, place)

    print("Cache info:", cached_func.cache_info())

    assert cached_func.cache_info().hits == 1
    print("LRU Cache with Place works.")

except Exception as e:
    print(f"Caught exception: {e}")
