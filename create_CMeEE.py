import json
import random

# 读取JSON文件
with open('CMeEE-V2_test.json', 'r') as json_file:
    data = json.load(json_file)

# 随机选择input模板
input_templates = [
    "找出指定的实体：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]\n答：",
    "找出指定的实体：\n[INPUT_TEXT]\n实体类型选项：[LIST_LABELS]\n答：",
    "找出句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]实体是什么？\n答：",
    "生成句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
    "下面句子中的[LIST_LABELS]实体有哪些？\n[INPUT_TEXT]\n答：",
    "实体抽取：\n[INPUT_TEXT]\n选项：[LIST_LABELS]\n答：",
    "医学实体识别：\n[INPUT_TEXT]\n实体选项：[LIST_LABELS]\n答："
]

# 新的映射字典
# 九大类，包括：疾病(dis)，临床表现(sym)，药物(dru)，医疗设备(equ)，医疗程序(pro)，身体(bod)，医学检验项目(ite)，微生物类(mic)，科室(dep)
new_category_mapping = {
    "dis":"疾病",
    "ite":"医学检验项目",
    "dep":"医院科室",
    "bod":"身体部位",
    "mic":"微生物类",
    "sym":"临床表现",
    "dru":"药物",
    "equ":"医疗设备",
    "pro":"医疗程序"
}

# 存储所有格式化数据为一个文件，每行一个 JSON 对象

formatted_data_list = []
index=0
for item in data:
    input_template = random.choice(input_templates)
    entities_list = []
    
    for key in new_category_mapping:
        for entity in item["entities"]:
            if entity["type"] == key:
                entities_list.append(new_category_mapping[key]+"实体："+entity["entity"])
    index=index+1
    formatted_item = {
        "input": input_template.replace("[INPUT_TEXT]", item["text"]).replace("[LIST_LABELS]", ",".join(new_category_mapping.values())),
        "target": "上述句子中的实体包含：\n"+'\n'.join(entities_list),
        "answer_choices": list(new_category_mapping.values()),
        "task_type": "ner",
        "task_dataset": "CMeEE-V2",
        "sample_id": index
    }
    formatted_data_list.append(formatted_item)


# 将格式化后的数据以一行一个 JSON 对象的格式写入文件
with open('CMeEE_formatted_test.json', 'w') as output_file:
    output_file.write('[')
    for formatted_item in formatted_data_list:
        json.dump(formatted_item, output_file, ensure_ascii=False)
        output_file.write(',\n')  # 在 JSON 对象之间换行分隔
    #删除一个多余的逗号
    output_file.seek(output_file.tell() - 3)
    output_file.truncate()
    #写入一个'}'
    output_file.write('}')
    #写入一个']'
    output_file.write(']')