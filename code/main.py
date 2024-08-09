# !/user/bin/env python3
# -*- coding: utf-8 -*-
import traceback
import time
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd

api_base = ""
api_key = ""

def get_data(n=50):
    db_path = '../data/emotion.csv'
    data = pd.read_csv(db_path, encoding='gbk')
    data = data.head(100)
    return data.sample(n=n)
def get_prompt():
    prompt = """
    **角色身份：**
    作为情感倾向鉴定专家，你利用情感分析技巧来判定文本的情感色彩。

    **任务概述：**
    针对一系列商品评论，你的职责是准确识别每条评论的情感倾向——是积极正面还是消极负面。

    **执行指南：**
    - 针对每段给定的评论文本，进行情感倾向判断。
    - 在心里默默陈列对情感色彩的解释和推理过程要求逻辑合理，不必输出。
    - 若是对于评论在积极与消极情感之间摇摆语句对文本中积极词汇与消极词汇的数量统计加权对比,再句子首尾权重更大，越到句子中间权重越低，例如整句中首尾句权重为1到整句中中间句子权重为0.45 将词汇的位置权重乘1得出单个句子的价值
    例如句子“一二三四五六六五四三二一”我们可以将“一”的加载看作1，将“二”的价值看作0.89，将“二”的价值看作0.78，将“六”的价值看着0.45，我们把“六“看着中间值可以把价值公式写作（1-（中间值到句首的长度-当前位置到中间值的长度）*（0.55/中间值到句首的长度））
    这个时候分别算出积极情绪价值和消极情绪价值分别求和然后对比,不必输出。
    - 最后只选择输出“Positive”或“Negative”，例如`跑步很合适，但是做工不佳，连接性也不是很便利`这段文本虽然前面“跑步很合适”是积极正面的情绪的但是后面的“但是做工不佳”以及“连接性也不是很便利”是负面消极，所以总体而言是负面情绪，所以应该输出Negative。

    **待分析评论样本：**
    {sentenece}

    请依据上述准则，对提供的评论进行情感倾向判断并给出简洁的响应。
            """
    prompt_template = PromptTemplate.from_template(prompt)
    return prompt_template
if __name__ == '__main__':
    start_time = time.time()
    data = get_data()
    prompt = get_prompt()
    batch_size = 1
    pre = []
    for i in range(0,len(data),batch_size):
        print(f'---------当前批次 {i}-{i + batch_size}/{len(data)}---------------------')
        try:
            df = data.iloc[i:i + batch_size, :]
            sentenece = df['sentenece'].to_list()
            print(sentenece)
            res = prompt | ChatOpenAI(
                model_name = 'gpt-3.5-turbo',
                temperature = 1,
                base_url = api_base,
                api_key = api_key,
                # openai_proxy = 'http://localhost:7890'
                )
            response = res.invoke(sentenece)
            print(response.content)
            pre.append(response.content)
        except Exception as e:
            traceback.print_exc()
            pass
    data['llm_res'] = pre
    data['correct'] = data.apply(lambda row:True if row['label'] in row['llm_res'] else False,axis=1)
    data.to_csv(f'../output/data_predit_{start_time}.csv', index=False)
    print('---------------------------------')
    print(f'预测正确率：', round(sum(data['correct']) / len(data) * 100, 2))
    end_time = time.time()
    print(f'用时：{end_time - start_time}s')


