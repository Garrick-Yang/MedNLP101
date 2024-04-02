import json
import random

# 读取JSON文件
with open('CHIP-STS_test.json', 'r') as json_file:
    data = json.load(json_file)

# 随机选择input模板
input_templates = [
    "以下两句话的意思相同的吗？\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”\n选项：是的，不是\n答：",
    "我想知道下面两句话的意思是否相同。\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\n选项：是的，不是\n答：",
    "我是否可以用以下的句子：“[INPUT_TEXT_1]”，来替换这个句子：“[INPUT_TEXT_2]”，并且它们有相同的意思？\n选项：是的，不是\n答：",
    "下面两个句子语义是“相同”或“不同”？\n“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\n选项：相同，不同\n答：",
    "“[INPUT_TEXT_1]”和“[INPUT_TEXT_2]”是同一个意思吗？\n选项：是的，不是\n答：",
    "“[INPUT_TEXT_1]”，“[INPUT_TEXT_2]”。\n这两句是一样的意思吗？\n选项：是的，不是\n答："
]

# 新的映射字典
new_category_mapping = {
    "0": "不同",
    "1": "相同",
    "":""
}

# 存储所有格式化数据为一个文件，每行一个 JSON 对象
formatted_data_list = []
for item in data:
    input_template = random.choice(input_templates)
    formatted_item = {
        "input": input_template.replace("[INPUT_TEXT_1]", item["text1"]).replace("[INPUT_TEXT_2]", item["text2"]),
        "target": new_category_mapping[item["label"]],
        "answer_choices": ["相同","不同"],
        "task_type": "cls",
        "task_dataset": "CHIP-STS",
        "sample_id": item["id"]
    }
    formatted_data_list.append(formatted_item)

# 将格式化后的数据以一行一个 JSON 对象的格式写入文件
with open('STS_formatted_test.json', 'w') as output_file:
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