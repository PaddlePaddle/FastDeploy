# tokenizer

## Description

The **Tokenizer** subcommand provides encoding and decoding functionality between text and token sequences. It also allows viewing or exporting model vocabulary information. Both text and multimodal models are supported.

## Usage

```
fastdeploy tokenizer --model MODEL (--encode TEXT | --decode TOKENS | --vocab-size | --info)
```

## Parameters

| Parameter                     | Description                                                                    | Default |
| ----------------------------- | ------------------------------------------------------------------------------ | ------- |
| --model, -m                   | Model path or name                                                             | None    |
| --encode, -e                  | Encode text into a list of tokens                                              | None    |
| --decode, -d                  | Decode a list of tokens back into text                                         | None    |
| --vocab-size, -vs             | Display the vocabulary size                                                    | None    |
| --info, -i                    | Display detailed tokenizer information (special tokens, IDs, max length, etc.) | None    |
| --vocab-export FILE, -ve FILE | Export the vocabulary to a file                                                | None    |

## Examples

```
# 1. Encode text into tokens
# Convert input text into a token sequence recognizable by the model
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --encode "Hello, world!"

# 2. Decode tokens into text
# Convert a token sequence back into readable text
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --decode "[1, 2, 3]"

# 3. View vocabulary size
# Output the total number of tokens in the model’s vocabulary
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --vocab-size

# 4. View tokenizer details
# Includes special symbols, ID mappings, max token length, etc.
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --info

# 5. Export vocabulary to a file
# Save the tokenizer’s vocabulary to a local file
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --vocab-export ./vocab.txt

# 6. Support for multimodal models
# Decode tokens for a multimodal model
fastdeploy tokenizer --model baidu/EB-VL-Lite-d --decode "[5300, 96382]"

# 7. Combine multiple functions
# Encode, decode, view vocabulary, and export vocabulary in a single command
fastdeploy tokenizer \
    -m baidu/ERNIE-4.5-0.3B-PT \
    -e "你好哇" \
    -d "[5300, 96382]" \
    -i \
    -vs \
    -ve vocab.json
```
