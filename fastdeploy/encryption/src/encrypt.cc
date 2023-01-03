//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "fastdeploy/encryption/include/model_code.h"
#include "fastdeploy/encryption/include/encrypt.h"
#include "fastdeploy/encryption/util/include/constant/constant_model.h"
#include "fastdeploy/encryption/util/include/crypto/aes_gcm.h"
#include "fastdeploy/encryption/util/include/crypto/sha256_utils.h"
#include "fastdeploy/encryption/util/include/crypto/base64.h"
#include "fastdeploy/encryption/util/include/system_utils.h"
#include "fastdeploy/encryption/util/include/io_utils.h"
#include "fastdeploy/encryption/util/include/log.h"

namespace fastdeploy {
    
std::string GenerateRandomKey() {
    std::string tmp = util::SystemUtils::random_key_iv(AES_GCM_KEY_LENGTH);
    // return util::crypto::Base64Utils::encode(tmp);
    return baidu::base::base64::base64_encode(tmp);
}

int EncryptStream(std::istream& in_stream, std::ostream& out_stream,
                   const std::string &keydata) {
    std::string key_str = baidu::base::base64::base64_decode(keydata);
    if (key_str.length() != 32) {
        return CODE_KEY_LENGTH_ABNORMAL;
    }

    in_stream.seekg(0, std::ios::beg);
    in_stream.seekg(0, std::ios::end);
    size_t plain_len = in_stream.tellg();
    in_stream.seekg(0, std::ios::beg);

    std::unique_ptr<unsigned char[]> plain(new unsigned char[plain_len]);
    in_stream.read(reinterpret_cast<char *>(plain.get()), plain_len);

    std::string aes_key = key_str.substr(0, AES_GCM_KEY_LENGTH);
    std::string aes_iv = key_str.substr(16, AES_GCM_IV_LENGTH);

    std::unique_ptr<unsigned char[]> cipher(
            new unsigned char[plain_len + AES_GCM_TAG_LENGTH]);
    size_t cipher_len = 0;
    int ret_encrypt = util::crypto::AesGcm::encrypt_aes_gcm(
                    plain.get(),
                    plain_len,
                    reinterpret_cast<const unsigned char*>(aes_key.c_str()),
                    reinterpret_cast<const unsigned char*>(aes_iv.c_str()),
                    cipher.get(),
                    reinterpret_cast<int&>(cipher_len));
    if (ret_encrypt != CODE_OK) {
        LOGD("[M]aes encrypt ret code: %d", ret_encrypt);
        return CODE_AES_GCM_ENCRYPT_FIALED;
    }

    std::string randstr = util::SystemUtils::random_str(constant::TAG_LEN);
    std::string aes_key_iv(key_str);
    std::string sha256_key_iv =
            util::crypto::SHA256Utils::sha256_string(aes_key_iv);
    for (int i = 0; i < 64; ++i) {
        randstr[i] = sha256_key_iv[i];
    }

    size_t header_len = constant::MAGIC_NUMBER_LEN +
                            constant::VERSION_LEN + constant::TAG_LEN;
    out_stream.write(constant::MAGIC_NUMBER.c_str(),
                    constant::MAGIC_NUMBER_LEN);
    out_stream.write(constant::VERSION.c_str(), constant::VERSION_LEN);
    out_stream.write(randstr.c_str(), constant::TAG_LEN);
    out_stream.write(reinterpret_cast<char *>(cipher.get()), cipher_len);

    return CODE_OK;
}

std::vector<std::string> Encrypt(const std::string& input,
                  const std::string& key) {
  
  std::istringstream isst(input);
  std::ostringstream osst;
  int ret =  EncryptStream(isst, osst, key);
  if (ret != 0) {
    FDERROR << ret << ", Failed encrypt " << std::endl;
    return {"", ""};
  }
  
  return {baidu::base::base64::base64_encode(osst.str()), key};
}

}  // namespace fastdeploy