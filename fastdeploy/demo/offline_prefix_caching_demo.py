"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""

from fastdeploy import LLM, SamplingParams

common_prefix = (
    "北京，中华人民共和国的首都，是一座融合了厚重历史与现代活力的超大城市。作为国家的政治中心、文化中心、国际交往中心和科技创新中心，北京承载着国家最高权力机关和众多国际机构。\n"
    "北京的历史可追溯至三千年前。它是元、明、清三朝古都，拥有众多举世闻名的文化遗产：世界上规模最大、保存最完整的古代宫殿建筑群——故宫，历经六百余年风雨；被誉为世界建筑奇迹的万里长城，"
    "其精华段蜿蜒于北京北部群山；庄严肃穆的天坛，是古代帝王祭天的圣地；贯穿城市南北、体现传统规划智慧的中轴线，串联起众多历史地标。\n"
    "步入现代，北京展现出蓬勃的活力。鸟巢（国家体育场）和水立方（国家游泳中心）是2008年奥运会的标志性遗产。中央商务区（CBD） 摩天大楼林立，彰显着经济实力。"
    "同时，传统的胡同和四合院依然散发着独特的生活气息，北京烤鸭等美食吸引着世界各地的游客。北京，这座古老而年轻的城市，正以其兼容并蓄的魅力，续写着辉煌篇章。\n"
    "阅读以上文字，回答下列问题"
)


prompts = [
    "北京作为中国的首都，主要承担着哪几个方面的中心职能？",
    "文中提到了哪两处最具代表性的古代皇家祭祀与居所建筑？",
    "文章分别列举了哪些具体实例来展现北京的“厚重历史”与“现代活力”？请各举两例",
]

generating_prompts = [common_prefix + prompt for prompt in prompts]


sampling_params = SamplingParams(temperature=1, top_p=0.0)

model = "baidu/ERNIE-4.5-21B-A3B-Paddle"

prefix_cached_llm = LLM(
    model=model,
    quantization="wint4",
    enable_prefix_caching=True,
)


prefix_outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

# 输出结果
for output in prefix_outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print("generated_text", generated_text)
    print("-" * 50)
