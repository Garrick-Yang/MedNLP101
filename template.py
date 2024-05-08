
CMeIE_templates = [
    "[INPUT_TEXT]\n这个句子里面具有一定医学关系的实体组有哪些？\n三元组关系选项[LIST_LABELS]",
    "给出句子中的[LIST_LABELS]等关系类型的实体对：[INPUT_TEXT]\n答：",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]等关系类型三元组是什么：",
    "给出句子中的[LIST_LABELS]的实体对：[INPUT_TEXT]\n答：",
    "找出句子中的[LIST_LABELS]的头尾实体对：\n[INPUT_TEXT]\n答："
]
CTC_templates = [
    "判断临床试验筛选标准的类型：\n[INPUT_TEXT]\n选项：[LIST_LABELS]\n答：",
    "确定试验筛选标准的类型：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n这句话是什么临床试验筛选标准类型？\n类型选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n是什么临床试验筛选标准类型？\n选项：[LIST_LABELS]\n答：",
    "请问是什么类型？\n[INPUT_TEXT]\n临床试验筛选标准选项：[LIST_LABELS]\n答："
]
QQR_templates = [
    "判断两个查询所表述的主题的匹配程度：\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n选项：[LIST_LABELS]\\n答：",
    "我想知道下面两个搜索词的意思有多相同。\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n选项：[LIST_LABELS]\\n答：",
    "下面两个句子的语义关系是？\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n选项: [LIST_LABELS]\\n答：",
    "“[INPUT_TEXT_1]”和“[INPUT_TEXT_2]”表述的主题完全一致吗？\\n选项：[LIST_LABELS]\\n答：",
    "“[INPUT_TEXT_1]”和“[INPUT_TEXT_2]”的意思有多相似？\\n选项：[LIST_LABELS]\\n答：",
    "“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n这两句是一样的意思吗？\\n选项：[LIST_LABELS]\\n答：",
    "“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n这两句的语义关系是？\\n选项：[LIST_LABELS]\\n答："
]
QIC_templates = [
    "[INPUT_TEXT]\n这个医疗搜索词是什么意图分类？\n选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n这个搜索是什么意图？\n类型选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n患者想要咨询什么信息？\n选项：[LIST_LABELS]\n答："
]
CDN_templates = [
    "给出下面诊断原词的标准化：\\n[INPUT_TEXT]\\n候选集：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "找出归一后的标准词：\\n[INPUT_TEXT]\\n选项：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "诊断归一化：\\n[INPUT_TEXT]\\n选项：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "诊断实体的语义标准化：\\n[INPUT_TEXT]\\n实体选项：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "给出诊断的归一化：\\n[INPUT_TEXT]\\n医学实体选项：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "[INPUT_TEXT]\\n归一化后的标准词是？\\n实体选项：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答：",
    "实体归一化：\\n[INPUT_TEXT]\\n实体候选：[LIST_LABELS]\\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\\n答："
]
STS_templates = [
    "以下两句话的意思相同的吗？\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”\\n选项：是的，不是\\n答：",
    "我想知道下面两句话的意思是否相同。\\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n选项：是的，不是\\n答：",
    "我是否可以用以下的句子：“[INPUT_TEXT_1]”，来替换这个句子：“[INPUT_TEXT_2]”，并且它们有相同的意思？\\n选项：是的，不是\\n答：",
    "“[INPUT_TEXT_1]”和“[INPUT_TEXT_2]”是同一个意思吗？\\n选项：是的，不是\\n答：",
    "“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\\n这两句是一样的意思吗？\\n选项：是的，不是\\n答："
  ]
CMeEE_templates = [
    "找出指定的实体：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]\n答：",
    "找出指定的实体：\n[INPUT_TEXT]\n实体类型选项：[LIST_LABELS]\n答：",
    "找出句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]实体是什么？\n答：",
    "生成句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
    "下面句子中的[LIST_LABELS]实体有哪些？\n[INPUT_TEXT]\n答：",
    "实体抽取：\n[INPUT_TEXT]\n选项：[LIST_LABELS]\n答：",
    "医学实体识别：\n[INPUT_TEXT]\n实体选项：[LIST_LABELS]\n答："
]