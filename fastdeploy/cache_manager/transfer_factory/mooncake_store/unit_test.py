import paddle

from fastdeploy.cache_manager.transfer_factory import MooncakeStore


def test_init_and_warmup():
    store = MooncakeStore()
    assert store.store is not None


def test_store_basic_function():
    store = MooncakeStore()
    buffer = paddle.zeros([1024, 1024], dtype=paddle.float32).cpu()
    store.register_buffer(buffer.data_ptr(), 1024 * 1024 * buffer.element_size())

    key = ["test_key_" + str(i) for i in range(2)]
    buffer[0, :] = 1
    buffer[1, :] = 2
    ptrs = [buffer.data_ptr(), buffer.data_ptr() + 1024 * 4]
    sizes = [1024, 1024]

    store.set(key, target_location=ptrs, target_sizes=sizes)
    buffer[0, :] = 3
    buffer[1, :] = 4
    print(buffer[0, 0], buffer[1, 0])

    store.get(key, target_location=ptrs, target_sizes=sizes)
    print("key: ", key)
    print("buffer: ", buffer[0, 0], buffer[1, 0])
    assert buffer[0, 0] == 1
    assert buffer[1, 0] == 2
    keys = ["test_key_0", "non_existent_key"]

    result = store.exists(keys)
    assert isinstance(result, dict)
    assert "test_key_0" in result
    print(result)
    assert result["test_key_0"] == 1
    assert result["non_existent_key"] == 0

    res = store.delete("test_key_0", timeout=10)
    assert res == 0

    new_result = store.exists(["test_key_0"])
    print(new_result)
    assert new_result["test_key_0"] == 0


if __name__ == "__main__":
    import os

    os.environ["MOONCAKE_CONFIG_PATH"] = "./mooncake_config.json"
    test_init_and_warmup()
    test_store_basic_function()
