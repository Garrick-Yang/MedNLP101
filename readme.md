# 代码结构
cblue_baselines:原始cblue的代码  
- 将三个原始任务文件合并为一个，放在run_cblue.py中， 后边脚本(.sh)中的文件也相应改为run_cblue.py

data：模型数据，CBLUE，PromptCBLUE样例数据集，改造数据集样式  
- output: cblue的输出，原始代码是输出到了这里，先不改吧。

output：输出相关内容文件放在这里  

peft：原始PromptCBLUE代码  

src：微调代码和脚本文件  
- scripts所有脚本文件，cblue和promptcblue  
