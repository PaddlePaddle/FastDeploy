# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .. import c_lib_wrap as C


def generate_key():
    """generate a key for encryption
    :return: key(str)
    """
    return C.encryption.generate_key()


def encrypt(input, key=None):
    """Encrypt a input string with key.
    :param: input: (str) The input str for encryption
    :param: key: (str,optional) The key for encryption(if not given, generate automatically.)
    :return: pair(str, str) [encrypted string, key]
    """
    if key is None:
        key = generate_key()
    return C.encryption.encrypt(input, key)


def decrypt(cipher, key):
    """Decrypt a input cipher with key.
    :param: cipher: (str) The input str for decryption
    :param: key: (str) The key for decryption
    :return: str(The decrypted str)
    """
    return C.encryption.decrypt(cipher, key)
