import io
import json
from argparse import ArgumentParser, Namespace

import pytest
from PIL import Image

import fastdeploy.benchmarks.datasets as bd


class DummyTokenizer:
    vocab_size = 100

    def num_special_tokens_to_add(self):
        return 1

    def decode(self, ids):
        return "dummy_text"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def make_temp_json(tmp_path, content):
    fpath = tmp_path / "data.json"
    with open(fpath, "w", encoding="utf-8") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")
    return str(fpath)


def test_is_valid_sequence_variants():
    assert bd.is_valid_sequence(10, 10)
    assert not bd.is_valid_sequence(1, 10)  # prompt too short
    assert not bd.is_valid_sequence(10, 1)  # output too short
    assert not bd.is_valid_sequence(2000, 10, max_prompt_len=100)
    assert not bd.is_valid_sequence(2000, 100, max_total_len=200)
    # skip min output len
    assert bd.is_valid_sequence(10, 1, skip_min_output_len_check=True)


def test_process_image_with_pil_and_str(tmp_path):
    # dict input with raw bytes
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_dict = {"bytes": buf.getvalue()}
    out = bd.process_image(raw_dict)
    assert "image_url" in out

    # PIL image input
    out2 = bd.process_image(img)
    assert out2["type"] == "image_url"
    assert out2["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # str input
    out3 = bd.process_image("path/to/file")
    assert out3["image_url"]["url"].startswith("file://")

    out4 = bd.process_image("http://abc.com/img.png")
    assert out4["image_url"]["url"].startswith("http://")

    # invalid input
    with pytest.raises(ValueError):
        bd.process_image(123)


def test_maybe_oversample_requests(caplog):
    dataset = bd.RandomDataset()
    requests = [bd.SampleRequest(1, "a", [], None, 10, 20)]
    dataset.maybe_oversample_requests(requests, 3)
    assert len(requests) >= 3

    def test_EBDataset_and_EBChatDataset(tmp_path):
        eb_content = [
            {
                "text": "hello",
                "temperature": 0.7,
                "penalty_score": 1.0,
                "frequency_score": 1.0,
                "presence_score": 1.0,
                "topp": 0.9,
                "input_token_num": 5,
                "max_dec_len": 10,
            }
        ]
        eb_file = make_temp_json(tmp_path, eb_content)
        eb = bd.EBDataset(dataset_path=eb_file, shuffle=True)
        samples = eb.sample(2)
        assert all(isinstance(s, bd.SampleRequest) for s in samples)
        assert all(s.json_data is not None for s in samples)

        chat_content = [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}]
        chat_file = make_temp_json(tmp_path, chat_content)
        chat = bd.EBChatDataset(dataset_path=chat_file, shuffle=True)
        samples2 = chat.sample(2, enable_multimodal_chat=False)
        assert all(isinstance(s, bd.SampleRequest) for s in samples2)
        assert all(s.json_data is not None for s in samples2)


def test_RandomDataset_sample():
    tok = DummyTokenizer()
    dataset = bd.RandomDataset(random_seed=123)
    samples = dataset.sample(tok, 2, prefix_len=2, range_ratio=0.1)
    assert len(samples) == 2
    assert all(isinstance(s, bd.SampleRequest) for s in samples)

    # range_ratio >= 1 should raise
    with pytest.raises(AssertionError):
        dataset.sample(tok, 1, range_ratio=1.0)


def test__ValidateDatasetArgs_and_get_samples(tmp_path):
    parser = ArgumentParser()
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", action=bd._ValidateDatasetArgs)

    # invalid: random + dataset-path
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-path", "abc.json"])

    # test get_samples with EBChat
    chat_content = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you?"},
            ],
            "max_tokens": 10,
        }
    ]
    chat_file = make_temp_json(tmp_path, chat_content)
    args = Namespace(
        dataset_name="EBChat", dataset_path=chat_file, seed=0, shuffle=False, num_prompts=1, sharegpt_output_len=10
    )
    out = bd.get_samples(args)
    assert isinstance(out, list)

    # unknown dataset
    args.dataset_name = "unknown"
    with pytest.raises(ValueError):
        bd.get_samples(args)


def test_add_dataset_parser():
    parser = bd.FlexibleArgumentParser()
    bd.add_dataset_parser(parser)
    args = parser.parse_args([])
    assert hasattr(args, "seed")
    assert hasattr(args, "num_prompts")
