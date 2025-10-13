import unittest

from pydantic import ValidationError

from fastdeploy.entrypoints.openai.protocol import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


class TestBatchRequestModels(unittest.TestCase):

    def test_batch_request_input_with_dict_body(self):
        body_dict = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "default",
        }
        obj = BatchRequestInput(
            custom_id="test",
            method="POST",
            url="/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertIsInstance(obj.body, ChatCompletionRequest)
        self.assertEqual(obj.body.model_dump()["messages"], body_dict["messages"])
        self.assertEqual(obj.body.model, "default")

    def test_batch_request_input_with_model_body(self):
        body_model = ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}], model="gpt-test")
        obj = BatchRequestInput(
            custom_id="456",
            method="POST",
            url="/v1/chat/completions",
            body=body_model,
        )
        self.assertIsInstance(obj.body, ChatCompletionRequest)
        self.assertEqual(obj.body.model, "gpt-test")

    def test_batch_request_input_with_other_url(self):
        obj = BatchRequestInput(
            custom_id="789",
            method="POST",
            url="/v1/other/endpoint",
            body={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertIsInstance(obj.body, ChatCompletionRequest)
        self.assertEqual(obj.body.messages[0]["content"], "hi")

    def test_batch_response_data(self):
        usage = UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1)
        chat_msg = ChatMessage(role="assistant", content="ok")
        choice = ChatCompletionResponseChoice(index=0, message=chat_msg, finish_reason="stop")

        resp = ChatCompletionResponse(id="r1", model="gpt-test", choices=[choice], usage=usage)

        data = BatchResponseData(
            status_code=200,
            request_id="req-1",
            body=resp,
        )
        self.assertEqual(data.status_code, 200)
        self.assertEqual(data.body.id, "r1")
        self.assertEqual(data.body.choices[0].message.content, "ok")

    def test_batch_request_output(self):
        response = BatchResponseData(status_code=200, request_id="req-2", body=None)
        out = BatchRequestOutput(id="out-1", custom_id="cid-1", response=response, error=None)
        self.assertEqual(out.id, "out-1")
        self.assertEqual(out.response.request_id, "req-2")
        self.assertIsNone(out.error)

    def test_invalid_batch_request_input(self):
        with self.assertRaises(ValidationError):
            BatchRequestInput(
                custom_id="id",
                method="POST",
                url="/v1/chat/completions",
                body={"model": "gpt-test"},
            )


if __name__ == "__main__":
    unittest.main()
