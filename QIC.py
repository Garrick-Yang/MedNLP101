import json
import random

with open('KUAKE-QIC_dev.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

input_templates = [
    "[INPUT_TEXT]\n这个医疗搜索词是什么意图分类？\n选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n这个搜索是什么意图？\n类型选项：[LIST_LABELS]\n答：",
    "[INPUT_TEXT]\n患者想要咨询什么信息？\n选项：[LIST_LABELS]\n答："
]

categories = {
    "diagnosis": "病情诊断",
    "cause": "病因分析",
    "method": "治疗方案",
    "advice": "就医建议",
    "metric_explain": "指标解读",
    "Ndisease_express": "疾病描述",
    "result": "后果表述",
    "attention": "注意事项",
    "effect": "功效作用",
    "price": "医疗费用",
    "other": "其他",
}

formatted_data_list = []
for item in data:
    input_template = random.choice(input_templates)
    formatted_item = {
        "input": input_template.replace("[INPUT_TEXT]", item["query"]).replace("[LIST_LABELS]", ",".join(categories.values())),
        "target": item["label"],
        "answer_choices": list(categories.values()),
        "task_type": "cls",
        "task_dataset": "CKUAKE-QIC",
        "sample_id": item["id"]
    }
    formatted_data_list.append(formatted_item)

with open('formatted_QIC_dev.json', 'w') as output_file:
    for formatted_item in formatted_data_list:
        json.dump(formatted_item, output_file, ensure_ascii=False)
        output_file.write('\n,')
