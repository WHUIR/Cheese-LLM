<p align="center">
<img width="500px" alt="Project Cheese-LLM" src="https://github.com/WHUIR/Cheese-ChatBot/blob/96fd23596b6579da96260c3fbdf068ac29a451f1/Cheese.png">
</p>

## What's Cheese?
Cheese is an open-source LLM trained for the Chinese language by Data Intelligence (Dance) Laboratory, School of Cyber Science and Engineering, Wuhan University. On the basis of Chinese-Llama, it is built by further going through a series of pre-training with [LoRA](https://github.com/microsoft/LoRA) and supervised fine-tuning (wo. LoRA) tasks over different large-scale Chinese datasets that are web-crawled and high-quality. Specifically, , we generate a more comprehensive Chinese vocabulary and more than 700K dialogs generated by ChatGPT are used to enhance the capacity of answering diverse information needs. The Cheese-LLM  is now available on GitHub for use in a wide spectrum of research tasks: question-answering, information search, personal assitant, to name a few.


## Why it's called Cheese?
We chose the name 'Cheese'(芝士) because it sounds similarly to 'Knowledge'(知识) in Chinese and is spelled similar to 'Chinese' in English. This reflects our commitment to developing a chatbot model that is knowledgeable in Chinese language, as well as our emphasis on providing a helpful and informative AI assistant.

## Overview
⚠️ All model weights and data are for **research use ONLY**. Commercial use is **strictly prohibited**. We accept **NO responsibility or liability** for any use of our data, code or weights.

This is the repo for the Cheese project, which aims to build a chinese chat model with LLaMA. This repository contains:
- The [dialogs](data) from 700K questions.
- The [code](finetune.py) for training with DeepSpeed, Lora.
- The [code](generate.py) for chat model demo (forked from [Alpaca-lora](https://github.com/tloen/alpaca-lora)).

### Model Hub
You can download all the above models in 🤗Model Hub, and use [🤗transformers](https://github.com/huggingface/transformers) and [🤗PEFT](https://github.com/huggingface/peft) to call Chinese LLaMA or the Alpaca LoRA model.

| Model              |             MODEL_NAME             |                             Link                             |
| ------------------ | :--------------------------------: | :----------------------------------------------------------: |
| Cheese-LLaMA-7B       | /cheese-llama-lora-7b       | [Model Hub Link]() |


## Performance

To assess the performance of our model, we conducted a evaluation of ten tasks of proposed method on 200 queries. 
Please note that reply generation is random and subject to various factors such as decoding hyperparameters and random seeds. Therefore, the following evaluations are not completely rigorous, and the test results should be used as a reference only. We encourage you to experience our model firsthand. For detailed evaluation results, please refer to .

| Task                           |                     Samples                     |  #   | Alpaca-7B | Chinese-Alpaca-Plus-7B | Cheese-Alpace-7B |
| ------------------------------ | :---------------------------------------------: | :--: | :-------: | :--------: | :------------: |
| **💯 Overall** |                   -                    |  200   |     65.3     |      70.9      |     **👍🏻75.3**     |
| Question Answering |            [QA.md](./examples/QA.md)            |   20   |      66       |       74       |      **👍🏻80**      |
| Open QA |           [OQA.md](./OQA.md)           |   20   |   **👍🏻79**    |       74       |      **👍🏻78**      |
| Computation, Reasoning |     [REASONING.md](./examples/REASONING.md)     |   20   |      31       |    **👍🏻50**    |         45         |
| Poetry, Literature, Philosophy |    [LITERATURE.md](./examples/LITERATURE.md)    |   20   |      68       |       73       |      **👍🏻76**      |
| Music, Sports, Entertainment | [ENTERTAINMENT.md](./examples/ENTERTAINMENT.md) |   20   |      68       |       74       |      **👍🏻79**      |
| Letters and Articles |    [GENERATION.md](./examples/GENERATION.md)    |   20   |      76       |    **👍🏻81**    |      **👍🏻81**      |
| Translation |   [TRANSLATION.md](./examples/TRANSLATION.md)   |   20   |      76       |       78       |      **👍🏻82**      |
| Multi-turn Dialogue |      [DIALOGUE.md](./examples/DIALOGUE.md)      |   20   |   **👍🏻83**    |       73       |      **👍🏻84**      |
| Coding   |          [CODE.md](./examples/CODE.md)          |   20   |      57       |    **👍🏻64**    |         59         |
| Ethics |        [ETHICS.md](./examples/ETHICS.md)        |   20   |      49      |       68       |      **👍🏻89**      |


## CLI and API
Now you can use Cheese with [Fastchat](https://github.com/lm-sys/FastChat) for the CLI and API provided by Fastchat!

First, install the latest version of Fastchat:
```bash
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/lm-sys/FastChat.git
```

Then, merge Baize's LoRA weights into LLaMA. Take 7B checkpoint as an example.
```bash
# Note you have to include "baize" in the target directory so Fastchat can recognize Baize.
python3 -m fastchat.model.apply_lora --base huggyllama/llama-7b --target ./model_weights/baize-7b --lora project-baize/baize-lora-7B
```

Now, run the CLI in your terminal! More options and configs can be found [here](https://github.com/lm-sys/FastChat#inference-with-command-line-interface).
```bash
# Optional: Add `--style rich` for better style.
python -m fastchat.serve.cli --model-path ./model_weights/baize-7b
```
You can use Cheese with OpenAI API or Hugging Face API following the instruction [here](https://github.com/lm-sys/FastChat#api).

### How to Run Locally
First, make sure your Python version is 3.8, and then install the required packages using the command below:

```bash
cd demo
pip install -r requirements.txt
```

You can host the model on your local machine using the following command:

```bash
# We assume you have obtained access to use LLaMA. The following LLaMA weights are from a 3rd party.
base_model=huggyllama/llama-7b
lora_model=project-baize/xxxxxx
python app.py $base_model $lora_model
```
#### GPU VRAM Requirements
|           | Inference (without int8) |
|-----------|--------------------------|
| Cheese-7B  | 16GB                     |

If you have a GPU with smaller VRAM, you can do inference with `int8`, by passing the 8bit argument:

```bash
python app.py $base_model $lora_model 8bit
```

## How to Reproduce

### Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. If `bitsandbytes` doesn't work, [install it from source](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md). Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).


### Training

The fine-tuning code is designed to run on an A100-80G GPU. The `finetune.py` script accepts three parameters: foundation model size (i.e., 7B, 13B, or 30B), batch size, learning rate and datasets. Note the total batch size is fixed to 64 (can be modified [here](https://github.com/project-baize/baize/blob/cbcf39902fcdfab8d935b7ea771a4e7d452a1be0/finetune.py#L24)) and the batch size here is the per device batch size before gradient accumulation. Set it to a smaller value if you are training on a GPU with smaller VRAM.

```bash
# For the 7B model (takes about 9 hours)
python finetune.py 7b 32 0.0002 alpaca,stackoverflow,quora

# For the 13B model (takes about 16 hours)
python finetune.py 13b 16 0.0001 alpaca,stackoverflow,quora

# For the 30B model (takes about 36 hours)
python finetune.py 30b 8 0.00005 alpaca,stackoverflow,quora
```
#### GPU VRAM Consumption
With the settings ABOVE:

|           | Training (with int8) |
|-----------|----------------------|
| Cheese-7B  | 14GB                 |


Got a question? See [this issue]().

### Merge LoRA into LLaMA
Now you can easily merge the trained LoRA weights into a LLaMA model so you can use it with everything that supports standard Hugging Face API!

Here's an example for merging `Cheese-lora-7B` into Chinese-LLaMA-7B.
```bash
python merge_lora.py \
--base huggyllama/llama-7b \
--target ~/model_weights/baize-7b \
--lora project-baize/baize-lora-7B
```
