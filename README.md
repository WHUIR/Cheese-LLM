<p align="center">
<img width="500px" alt="Project Cheese-LLM" src="https://github.com/WHUIR/Cheese-ChatBot/blob/96fd23596b6579da96260c3fbdf068ac29a451f1/Cheese.png">
</p>

## What's Cheese?
Cheese is an open-source LLM trained for the Chinese language by Data Intelligence (Dance) Laboratory, School of Cyber Science and Engineering, Wuhan University. 
On the basis of Chinese-Llama, it is built by further going through a series of pre-training, supervised fine-tuning and reinforcement learning from human feedback(RLHF) over different large-scale Chinese datasets that that are web-crawled and high-quality.
Some highlights include a more comprehensive Chinese vocabulary and more than 700K dialogs generated by ChatGPT used through SFT to enhance the capacity of answering diverse information needs. 
The Cheese-LLM is now available on GitHub for use in a wide spectrum of research tasks: question-answering, information search, personal assitant, to name a few.


## Why it's called Cheese?
We chose the name 'Cheese'(芝士) because its Chinese words sounds similarly to 'Knowledge'(知识) and is spelled similar to 'Chinese' in English. This reflects our commitment to developing LLMs that are more knowledgeable in Chinese language, as well as our emphasis on providing a helpful and informative AI assistant.

## Overview
Please be advised that all model weights and data provided here are intended for **research purposes ONLY**. Any commercial use of this information is **strictly prohibited**. We assume **no responsibility or liability** for any consequences resulting from the use of our weights, codes, or data.

## Usage 
### To be updated!
We offer int8 quantizations, which will largely reduce the GPU memory consumption (i.e., only 7GB) and enable deployment in GPUs like 3090, 4090.


## Evaluation and Benchmark
Since our CheeseLLM focuses on Chinese language, we conduct an automatic evaluation by using ChatGPT-3.5 for rating the response. The models in comparison include ChatGPT-3.5, Chinese-Alpaca-Plus-7B, Chinese-Alpaca-13B. We choose to follow the same evaluation setting of [BELLE](https://github.com/LianjiaTech/BELLE/tree/main/eval) and [Phoenix](https://github.com/FreedomIntelligence/LLMZoo). Specifically, there are about 1,000 questions that can be classified into ten categories: Math, Extract, Closed QA, Rewrite, Summarization, Generation, Classification, Brainstorming, Open QA, Code. Each category contains around 100 questions with the predetermined prompts for rating the results. The scores are normalized such that CheeseLLM is 100. The performance comparison is reported as follow:

| Category | ChatGPT-3.5 |  CheeseLLM-v1.0  | Chinese-Alpaca-Plus-7B | Chinese-Alpaca-13B |
| :-------- | :------: | :----------: | :----------------: | :-----------------------: |
| Math | 159.16 | 100 | 70.11 | 33.04 |
| Extract | 111.11 | 100 | 90.94 | 91.99 |
| Closed QA | 104.50 | 100 | 96.20 | 92.05 |
| Rewrite | 110.95 | 100 | 90.92 | 89.38 |
| Summarization | 111.96 | 100 | 95.02 | 98.40 |
| Generation | 102.76 | 100 | 92.58 | 87.69 |
| Classification | 105.30 | 100 | 91.42 | 88.87 |
| Brainstorming | 108.20 | 100 | 92.18 | 88.05 |
| Open QA | 107.76 | 100 | 89.82 | 88.84 |
| Code | 106.81 | 100 | 98.47 | 90.00 |
| Overall | 110.38 | 100 | 90.57 | 85.59 |

### Limitations


## Contributors
Kui Liu, Shirui Hu, Yesheng Liu, Xuekong Xu, Zihao Li, Rui Zhu, Xinyu Zou
### Principle Investigator (PI)
Lixin Zou, Chenliang LI

## Acknowledgement
School of Cyber Science and Engineering, Wuhan University

Aixin Sun, Nanyang Technological University

Qi Zhang, Fudan University

One February night in Beijing: A wonderful dinner at 串亭居酒屋 (a Japanese Izakaya located at Tsinghua Tech. Park) with Kang Liu, Zhiyuan Liu, Xipeng Qiu, Jiajun Zhang, Binyang Li and Xianpei Han
