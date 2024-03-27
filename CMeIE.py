import json
import random

with open('CMeIE-V2_dev.json', 'r', encoding='utf-8') as json_file:
    data = json_file.readlines()
    # data = json.load(json_file)

input_templates = [
    "[INPUT_TEXT]\n这个句子里面具有一定医学关系的实体组有哪些？\n三元组关系选项[LIST_LABELS]",
    "给出句子中的[LIST_LABELS]等关系类型的实体对：[INPUT_TEXT]\n答：",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]等关系类型三元组是什么：",
    "给出句子中的[LIST_LABELS]的实体对：[INPUT_TEXT]\n答：",
    "找出句子中的[LIST_LABELS]的头尾实体对：\n[INPUT_TEXT]\n答："
]

formatted_data_list = []
count = 1
for item in data:
    item = json.loads(item)
    input_template = random.choice(input_templates)
    pre_list = []
    sub_list = []
    ob_list = []
    target = []
    tar_list = {"pre_list":pre_list, "sub_list":sub_list, "ob_list":ob_list,"target":target}
    for spo in item["spo_list"]:
        tar_list["pre_list"].append(spo["predicate"])
        tar_list["sub_list"].append(spo["subject"])
        tar_list["ob_list"].append(spo["object"]["@value"])
        tar_list['target'].append("具有"+spo["predicate"]+"关系的头尾体如下：头实体为"+spo["subject"]+",尾实体为"+spo["object"]["@value"])
    formatted_item = {
        "input": input_template.replace("[INPUT_TEXT]", item["text"]).replace("[LIST_LABELS]", ",".join(tar_list["pre_list"])),
        "target": "\n".join(tar_list["target"]),
        "answer_choices": tar_list["pre_list"],
        "task_type": "spo_generation",
        "task_dataset": "CMeIE",
        "sample_id": "CMeIE"+str(count)
    }
    formatted_data_list.append(formatted_item)
    count += 1

with open('formatted_CMeIE_dev.json', 'w',encoding='utf-8') as output_file:
    for formatted_item in formatted_data_list:
        json.dump(formatted_item, output_file, ensure_ascii=False)
        output_file.write('\n')
