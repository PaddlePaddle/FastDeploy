import os
import unittest

from paddleocr import PaddleOCRVL

# os.environ["MACA_VISIBLE_DEVICES"] = "0"
# os.environ["FD_MOE_BACKEND"] = "cutlass"
# os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
# os.environ["FLAGS_weight_only_linear_arch"] = "80"
# os.environ["FD_METAX_KVCACHE_MEM"] = "8"
# os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
# os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"
# os.environ["FD_SAMPLING_CLASS"] = "rejection"
# os.environ["PADDLE_PDX_DISABLE_DEV_MODEL_WL"] = "true"


class TestPaddleOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()

        # ===============================
        # 基于 FastDeploy Server 推理
        # ===============================
        cls.pipeline = PaddleOCRVL(
            vl_rec_backend="fastdeploy-server",
            vl_rec_server_url="http://127.0.0.1:8118/v1",
            # device=os.getenv("PADDLE_XCCL_BACKEND", "metax_gpu"),
            layout_detection_model_name="PP-DocLayoutV3",
            layout_detection_model_dir=cls.model_root_path + "/PP-DocLayoutV3/",
            vl_rec_model_name="PaddleOCR-VL-1.5-0.9B",
            vl_rec_model_dir=cls.model_root_path + "/PaddleOCR-VL-1.5/",
        )

        # ===============================
        # 可选功能（按需开启）
        # ===============================

        # 是否启用文档方向分类模型
        # pipeline = PaddleOCRVL(use_doc_orientation_classify=True)

        # 是否启用文本图像矫正模块
        # pipeline = PaddleOCRVL(use_doc_unwarping=True)

        # 是否关闭版面区域检测排序模块
        # pipeline = PaddleOCRVL(use_layout_detection=False)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""

        cls.model_root_path = os.getenv(
            "MODEL_ROOT_PATH", "/workspace/models/modelscope.hub.metax-tech.com/models/PaddlePaddle"
        )
        cls.input_image_path = os.getenv("INPUT_IMAGE_PATH", "paddleocr_vl_demo.png")
        cls.ocr_outputs_path = os.getenv("OCR_OUTPUTS_PATH", "./paddle_ocr_outputs")

        cls.image_content_keywords = [
            "助力双方交往 搭建友谊桥梁",
            "学好中文，我们的未来不是梦",
            "在中国学习的经历让我看到更广阔的世界",
            "共同向世界展示非洲和亚洲的灿烂文明",
            "中厄两国人文交流不断深化，互利合作的民意基础日益深厚",
            "这是我人生的重要一步，自此我拥有了一双坚固的鞋子，赋予我穿越荆棘的力量",
            "学习彼此的语言和文化，将帮助厄中两国人民更好地理解彼此，助力双方交往，搭建友谊桥梁",
            "中国文化博大精深，我希望我的学生们能够通过中文歌曲更好地理解中国文化",
        ]

    def test_image_parse(self):
        # ===============================
        # 执行预测
        # ===============================
        outputs = self.pipeline.predict(self.input_image_path)
        image_all_content = ""

        # def extract_contents(contents: list[str]) -> str:
        #     text=""
        #     for content in contents:
        #         text = text + content.split("content:")[1].split("#")[0]
        #     return text

        # ===============================
        # 保存和打印结果
        # ===============================
        for res in outputs:
            # res.print()  # 打印结构化输出
            # image_all_content=extract_contents(str(res["parsing_res_list"]))
            image_all_content = str(res["parsing_res_list"])
            # res.save_to_json(save_path=self.ocr_outputs_path)       # 保存 JSON 结果
            res.save_to_markdown(save_path=self.ocr_outputs_path)  # 保存 Markdown 结果

        assert all(keyword in image_all_content for keyword in self.image_content_keywords)


if __name__ == "__main__":
    unittest.main()

    # model_root_path=os.getenv("MODEL_ROOT_PATH", "/workspace/models/modelscope.hub.metax-tech.com/models/PaddlePaddle")
    # input_image_path=os.getenv("INPUT_IMAGE_PATH", "paddleocr_vl_demo.png")
    # ocr_outputs_path=os.getenv("OCR_OUTPUTS_PATH", "./paddle_ocr_outputs")

    # pipeline = PaddleOCRVL(
    #     vl_rec_backend="fastdeploy-server",
    #     vl_rec_server_url="http://127.0.0.1:8118/v1",
    #     device="metax_gpu",

    #     layout_detection_model_name="PP-DocLayoutV3",
    #     layout_detection_model_dir=model_root_path+"/PP-DocLayoutV3/",

    #     vl_rec_model_name="PaddleOCR-VL-1.5-0.9B",
    #     vl_rec_model_dir=model_root_path+"/PaddleOCR-VL-1.5/",
    # )

    # # ===============================
    # # 可选功能（按需开启）
    # # ===============================

    # # 是否启用文档方向分类模型
    # # pipeline = PaddleOCRVL(use_doc_orientation_classify=True)

    # # 是否启用文本图像矫正模块
    # # pipeline = PaddleOCRVL(use_doc_unwarping=True)

    # # 是否关闭版面区域检测排序模块
    # # pipeline = PaddleOCRVL(use_layout_detection=False)

    # # ===============================
    # # 执行预测
    # # ===============================

    # outputs = pipeline.predict(input_image_path)

    # # ===============================
    # # 保存和打印结果
    # # ===============================

    # for res in outputs:
    #     # res.print()  # 打印结构化输出
    #     # res.save_to_json(save_path=ocr_outputs_path)       # 保存 JSON 结果
    #     res.save_to_markdown(save_path=ocr_outputs_path)   # 保存 Markdown 结果
