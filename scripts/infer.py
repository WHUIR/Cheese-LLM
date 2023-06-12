import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import json, os
import gradio as gr
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--with_prompt',action='store_true')
args = parser.parse_args()


prompt_input = (
    "# 问题:\n\n{instruction}\n\n# 回答:\n\n"
)


def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)

    if device==torch.device('cpu'):
        base_model.float()

    base_model.eval()
    def evaluate(
        instruction,
        inputs=None,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=400,
        repetition_penalty=2.0,
        **kwargs,
    ):
        with torch.autocast("cuda"):
            with torch.no_grad():
                raw_input_text = instruction
                print(instruction)
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?

                generation_config = dict(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=False,
                    num_beams=num_beams,
                    repetition_penalty=float(repetition_penalty),
                    bad_words_ids=[[30330]],
                    no_repeat_ngram_size=10,
                    max_new_tokens=max_new_tokens
                )
                generation_output = base_model.generate(
                    input_ids = inputs["input_ids"].to(device), 
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                # print(generation_output.shape)
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("# 回答:")[1].strip()
                else:
                    response = output
                print("Response: ",response)
                print("\n")
                return response


    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.2, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=400, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=0.0, maximum=3, step=0.1, value=2.0, label="Repetition Penalty"
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="CheeseChat",
        description="CheeseChat 是由武汉大学Dance Lab训练的大型自然语言处理模型。它能提供各种自然语言处理服务，例如回答问题、翻译文本、生成文章、聊天对话等。",  # noqa: E501
    ).launch(server_name="0.0.0.0", share=True)
