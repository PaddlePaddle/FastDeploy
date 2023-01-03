import fastdeploy as fd
import os

if __name__ == "__main__":
    input = "Hello"
    cipher, key = fd.encryption.encrypt(input)
    output = fd.encryption.decrypt(cipher, key)
    assert input == output, "test encryption failed"
    print("test encryption success")
