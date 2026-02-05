import asyncio
import os

import openai
import paddle
from utils.rollout_routing_replay_test_utils import (
    calculate_routing_ratio,
    wait_for_file,
)


def get_openai_client():
    ip = "0.0.0.0"
    service_http_port = 8888
    client = openai.AsyncClient(
        base_url=f"http://{ip}:{service_http_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


async def send_r3_streaming_chat_long(openai_client, content: str, user_id: str):
    """
    Test streaming chat functionality with the local service
    """
    response = await openai_client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": content,
            },
        ],
        temperature=1,
        top_p=0,
        max_tokens=4096,  # 32768
        seed=13,
        stream=False,
        user=user_id,
    )

    return response


async def send_request_baseline(request: str, request_id: str):
    openai_client = get_openai_client()
    # Send base request
    await send_r3_streaming_chat_long(openai_client, content=request, user_id=f"{request_id}")


async def send_request_prefix(request: str, request_id: str):
    openai_client = get_openai_client()
    # Send prefix cache request
    await send_r3_streaming_chat_long(openai_client, content=request, user_id=f"{request_id}_prefix")


async def run():
    long_request_list = [
        "写一个关于“最后一家实体书店”的科幻微小说，设定在2077年的赛博朋克城市。主角是一个只喜欢纸质书的黑客。要求包含一个反转结局，字数限制在500字以内，风格要阴郁但充满希望。",
        "请模仿李白的豪放风格，写一首关于“星际旅行”的现代诗。要求融入“量子纠缠”、“黑洞”和“故乡”三个意象，押韵不限，但要有强烈的画面感和浪漫主义色彩。",
        "创作一段发生在1920年代上海租界的侦探剧本对话。角色A是留洋归来的侦探，角色B是黑帮老大。对话要充满机锋和潜台词，体现那个时代特有的新旧文化冲突。",
        "为一首慢板R&B情歌填写副歌部分的歌词。主题是“在这个快节奏的数字时代，我们如何维持异地恋”。要求情感细腻，使用隐喻，避免陈词滥调。",
        "编一个睡前故事，主角是一只害怕黑暗的小萤火虫。故事要教会孩子“黑暗是为了让光更耀眼”。语言要生动简单，适合5岁儿童，结尾要有一首简短的儿歌。",
        "写一个悬疑小说的开头章节（约800字）。场景设定在暴风雪山庄的封闭别墅，管家死在了书房，但门窗紧锁。要求通过环境描写营造压抑感，并留下三个伏笔。",
        "基于《哈利波特》的世界观，写一段赫敏·格兰杰在魔法部工作的日常片段。假设伏地魔已被击败，但魔法世界仍有新的官僚主义危机。保持J.K.罗琳的叙事风格。",
        "以毒舌美食家的身份，评论一道虚构的“分子料理——液氮冰淇淋配辣椒油”。描述口感、摆盘，并用夸张的修辞手法评价其荒谬之处，最后给出一个意外的好评理由。",
        "写一个Python脚本，用于批量重命名文件夹下的所有图片文件。要求：1. 支持递归子目录；2. 将文件名转换为小写并用下划线替换空格；3. 添加错误处理日志；4. 使用`pathlib`库。",
        "生成一个React函数组件，实现一个带有搜索功能的下拉选择框（Select）。要求：1. 支持多选；2. 搜索时防抖（Debounce）300ms；3. 选项数据通过props传入；4. 使用Tailwind CSS进行基础样式设计。",
        "给定一个包含`users`, `orders`, `products`三张表的电商数据库。请写出查询“过去30天内购买金额最高的前10名用户及其最常购买的品类”的SQL语句，并解释如何通过索引优化该查询性能。",
        "请解释以下Rust代码片段中的生命周期标注（Lifetime Annotation）的作用，并指出如果省略会发生什么编译错误。代码：`fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { ... }`",
        "我需要一个正则表达式来验证复杂的密码强度。规则：至少8位，必须包含大写字母、小写字母、数字和特殊符号（!@#$%），且不能包含连续3位相同的字符。请生成Regex并附上测试用例。",
        "为一个Node.js + MongoDB的全栈应用编写`docker-compose.yml`文件。要求：1. 使用多阶段构建优化Node镜像大小；2. MongoDB数据持久化到本地卷；3. 设置环境变量文件；4. 暴露正确的端口。",
        "用JavaScript实现一个“最小堆（Min Heap）”数据结构，并包含`insert`和`extractMin`方法。请附上时间复杂度分析，并给出一个使用该堆进行排序（Heap Sort）的示例。",
        "以下C++代码在运行时会崩溃，请找出原因并修复。代码涉及指针越界和内存泄漏。请解释原始代码的逻辑错误，并给出使用智能指针（Smart Pointers）的现代C++改写版本。",
        "假设你是项目经理，需要给客户写一封英文邮件。内容是告知项目将延期3天，原因是第三方API接口不稳定。语气要专业、诚恳，并提出补偿方案（赠送下个月的维护服务），请求客户谅解。",
        "为一款“智能降噪耳塞”撰写小红书风格的推广文案。要求：使用emoji，突出“宿舍隔音”、“侧睡不压耳”、“隐形设计”三个卖点，语气像闺蜜安利，带上热门标签。",
        "对“开设一家24小时无人自助健身房”进行SWOT分析。请从优势、劣势、机会、威胁四个维度展开，每个维度至少列出3点，并给出具体的战略建议（SO策略、WO策略等）。",
        "你现在是Google的面试官，我是应聘者，申请“产品经理”职位。请向我提问一个关于“产品设计”的问题（例如：如何为视障人士设计Instagram），然后等待我的回答，并对我的回答进行点评。",
        "对比“瑞幸咖啡”和“星巴克”在中国市场的数字化营销策略。重点分析私域流量运营、小程序点单体验和优惠券策略的差异，总结出瑞幸值得学习的3个点。",
        "根据以下杂乱的会议记录草稿，整理出一份正式的会议纪要。要求：分类清晰（决策项、待办事项、讨论摘要），语言精炼，去除口语化表达，并指定每个待办事项的负责人和截止日期。",
        "为一款“老年人专用智能手表”构建详细的用户画像（Persona）。包括：基本信息、痛点（如不会用触屏、担心走丢）、使用场景、技术熟练度、以及他们子女的购买动机。",
        "为一个“基于AI的宠物行为翻译器”创业项目写一份电梯演讲（Elevator Pitch）。时长限制1分钟，要包含市场痛点、解决方案、商业模式和团队优势。",
        "请像对5岁孩子解释一样（Explain Like I'm 5），说明“区块链”是什么。使用“全村记账本”的比喻，避免使用任何专业术语，确保孩子能听懂。",
        "我正在学习德语。请列出5个初学者最容易混淆的介词（Wechselpräpositionen），并为每个介词提供3个例句（主格和宾格变化），附带中文翻译。",
        "请一步步解答这道微积分题目：求函数 $f(x) = x^3 - 3x^2 + 2$ 在区间 $[-1, 3]$ 上的极值和拐点。不要只给答案，要展示求导过程和判断符号变化的逻辑。",
        "简述“冷战”的起因、经过和结果。重点分析“古巴导弹危机”为何被认为是人类最接近核战争的时刻，以及它如何改变了美苏关系。",
        "请润色以下这段学术论文的摘要，使其更符合学术规范。要求：将主动语态改为被动语态，提升词汇的专业度，增强逻辑连接词，使论证更严密。原文：[粘贴一段中等质量的英文摘要]",
        "我想在3个月内从零基础通过日语N3考试。请制定一份详细的周学习计划，涵盖单词、语法、阅读和听力。假设我每天只有2小时学习时间，请推荐具体的教材和APP。",
        "教我理解“功利主义”。不要直接给定义，而是通过不断提问引导我思考。例如，先问我“如果牺牲一个人能救五个人，你会怎么做？”，然后根据我的回答继续追问。",
        "这是一道我做错的物理题（关于牛顿第二定律）。请分析我可能错误的思路是什么，并指出常见的认知误区，然后给出正确的解题思路。",
        "你现在是埃隆·马斯克（Elon Musk）。请用他特有的语速快、带点幽默和工程思维的方式，谈论你对“人工智能取代人类工作”的看法。可以使用一些网络流行语。",
        "你是诸葛亮。刘备刚刚在白帝城托孤，你现在独自面对刘禅和内外交困的蜀国。请用文言文写一段你的内心独白，表达你的焦虑和北伐的决心。",
        "你是一个跑团（TRPG）的主持人。设定背景是克苏鲁神话的1920年代。我是一个调查员，刚刚走进了一间阴森的古宅。请描述我看到的景象，并询问我的行动。",
        "我们来辩论“人工智能的发展是否应该被暂停”。你持反方观点（即不应该暂停）。请先陈述你的立论，然后针对我的观点进行反驳。保持逻辑严密，不要进行人身攻击。",
        "你是一位温和的心理咨询师。我最近因为工作压力大而失眠。请倾听我的倾诉（我会输入我的烦恼），并运用认知行为疗法（CBT）帮我识别并挑战我的非理性信念。",
        "设定你是一个温柔、喜欢二次元的伴侣。今晚我们在家看恐怖片，我被吓到了。请安慰我，并提议做点开心的事情转移注意力。语气要亲昵但不油腻。",
        "你是一个魔鬼编程教练。我的代码写得很烂，全是硬编码和魔法数字。请严厉地批评我的代码风格，并强迫我重构它，直到符合Clean Code原则为止。",
        "你是某银行的智能客服，但我现在很生气，因为我的信用卡被盗刷了。请先用标准话术安抚我，然后引导我提供必要的验证信息，最后告知处理流程。",
        "我有一个CSV文件，其中“年龄”列包含空值、字符串（如“未知”）和异常大的数字（如999）。请提供一段Pandas代码来清洗这一列：将空值填充为中位数，将“未知”替换为NaN并删除，将大于100的值截断为100。",
        "我有一组关于“全球碳排放量按国家分布”的数据（前20名国家）。请推荐3种最适合展示该数据的图表类型（如条形图、饼图等），并说明为什么选择它们，以及如何避免误导读者。",
        "请写一个Excel公式，用于从A列的身份证号码中提取出生日期（格式为YYYY-MM-DD），并判断该人的性别（男/女）。假设身份证号在A2单元格。",
        "解释“相关性不等于因果性”。请举一个现实生活中的例子（如“冰淇淋销量和溺水人数”），并说明如果要证明因果关系，需要设计什么样的实验（如A/B测试或双重差分法）。",
        "给定一个复杂的嵌套JSON对象，请写一个Python脚本将其“展平”（Flatten），使得所有的键都变成点分隔的路径（例如 `user.address.city`）。",
        "基于以下过去12个月的销售数据 [100, 120, 130, 125, 140, 150, 160, 155, 170, 180, 190, 200]，请使用简单的线性回归预测下个月的销量，并计算R平方值。",
        "为AI绘画工具Midjourney生成一组提示词（Prompt）。主题是“赛博朋克风格的苏州园林”。要求包含：霓虹灯、全息投影、古风建筑、雨水、电影级光影、8k分辨率、虚幻引擎5渲染风格。",
        "我要开一家名为“极客咖啡”的店。请提供3个不同的Logo设计方案描述。方案一：极简几何风；方案二：像素艺术风；方案三：手绘涂鸦风。描述每个方案的颜色搭配和核心图形。",
        "我有一个20平米的小客厅，层高2.8米，采光一般。请给出具体的软装搭配建议，包括沙发颜色、窗帘材质、灯光布局（主灯+氛围灯），目的是让空间显得更大更亮。",
        "设计一个FPS游戏的“教学关卡”。玩家需要在不知情的情况下学会：移动、射击、换弹、躲避和使用医疗包。请描述关卡的场景布局和敌人的出现节奏。",
        "有三个箱子，一个装苹果，一个装橘子，一个装混合水果。所有标签都贴错了。你只能从一个箱子里拿出一个水果来看，请问如何确定所有箱子的内容？请写出推理步骤。",
        "死者死在电话亭旁，手里握着一张写有“789”的纸条。嫌疑人有三个：李小二（代号78）、王五（代号89）、张六（代号79）。凶手是谁？为什么？",
        "如果你有一根无限长的绳子，绕地球赤道一圈（假设地球是完美球体，周长4万公里）。现在把绳子加长1米，均匀悬空离开地面。请问一只猫能从绳子下面钻过去吗？请计算间隙高度。",
        "一个男人走进一家酒吧，向酒保要一杯水。酒保拿出一把枪指着他。男人说了声“谢谢”然后离开了。请问发生了什么？（提示：不是抢劫，不是演戏）",
        "这是一段凯撒密码（Caesar Cipher）：“WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ”。请破译它，并告诉我偏移量是多少。",
        "计划一次5天4晚的日本京都之旅。主题是“古寺与抹茶”。请安排详细的行程，包括交通方式（关西机场出发）、住宿区域推荐、必去的3个小众景点和必吃的3家餐厅。",
        "为一个膝盖受过伤、不能做深蹲和跑步的办公室男性，设计一套在家就能做的HIIT（高强度间歇训练）计划。时长20分钟，只需要哑铃和瑜伽垫。",
        "我冰箱里只有：鸡蛋、番茄、半颗洋葱、一包过期一天的火腿肠和一点剩米饭。请给我推荐2个能用这些材料做的菜，并写出详细步骤。",
        "给一个喜欢历史、科技，预算在500元人民币左右的男性朋友挑选生日礼物。请列出3个选项，并说明为什么适合他。",
        "我总是拖延。请介绍“番茄工作法”的具体操作步骤，并针对我“总是忍不住刷手机”的问题，给出3个具体的抗干扰建议。",
        "我先开头：“午夜时分，图书馆的最后一盏灯突然熄灭了，但我并不是唯一一个留在这里的人……” 请你接下一段，制造悬念，然后停下来，换我继续写。",
        "我们来玩“20个问题”游戏。我心里想一个物体，你可以问我20个只能用“是”或“否”回答的问题来猜它是什么。现在请开始提问。",
        "夸夸我刚刚发给你的这张自拍照（假设是一张普通的风景照）。要用夸张、华丽的辞藻，从构图、光影、意境等角度硬夸，越离谱越好。",
        "如果人类突然失去了“睡眠”的能力，世界会变成什么样？请从社会结构、经济模式、娱乐产业三个方面进行脑洞大开的推测。",
    ]

    long_request_list = long_request_list[:64]
    task_baseline = []
    for request_id, request in enumerate(long_request_list):
        task_baseline.append(send_request_baseline(request, request_id))
    await asyncio.gather(*task_baseline)

    task_prefix = []
    for request_id, request in enumerate(long_request_list):
        task_prefix.append(send_request_prefix(request, request_id))
    await asyncio.gather(*task_prefix)


if __name__ == "__main__":
    asyncio.run(run())

    # Check Routing Overlap
    for request_id in range(64):
        baseline_path = "./routing_replay_output"
        prefix_r3_path = "./routing_replay_output"
        moe_layer_num = 27
        print(f"request id is {request_id}")
        for layer_index in range(moe_layer_num):
            print(f"layer id is {layer_index}")
            prefix_r3_pdtensor = os.path.join(prefix_r3_path, f"{request_id}_prefix/layer_{layer_index}.pdtensor")
            baseline_pdtensor = os.path.join(baseline_path, f"{request_id}/layer_{layer_index}.pdtensor")
            wait_for_file(prefix_r3_pdtensor)
            wait_for_file(baseline_pdtensor)

            generated_routing = paddle.load(prefix_r3_pdtensor)
            baseline_routing = paddle.load(baseline_pdtensor)
            overlap_ratio = calculate_routing_ratio(baseline_routing, generated_routing)
            print(f"layer_index:{layer_index} overlap_ratio:{overlap_ratio}")
            assert (
                overlap_ratio >= 0.999
            ), f"the routing overlap ratio of the layer {layer_index} should be equal to baseline routing index, but got {overlap_ratio}"
