from prompts.tools_instructions import CLEAN_LIST_SYSTEM, CLEAN_LIST_USER, EXTRACT_BASE_USER, \
    EXTRACT_NAME_SYSTEM, EXTRACT_DESCRIPTION_SYSTEM, EXTRACT_JUSTIFICATION_SYSTEM, \
    EXTRACT_GRADE_SYSTEM, EXTRACT_FINAL_ANSWER_SYSTEM, \
    EXTRACT_CONFIDENCE_SCORE_SYSTEM, EXTRACT_REASONING_STEPS_SYSTEM, EXTRACT_CONCLUSION_SYSTEM

from prompts.chat_template import chat_template

from typing import Tuple
from time import time
from transformers import set_seed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Suppress specific warning from configuration_utils.py
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `temperature` is set to `0.0`")

PATTERN_TO_REMOVE = "<|start_header_id|>assistant<|end_header_id|>"

##################################
#  initialize the model
##################################

#  set seed

seed_value = 23
# Set seed for PyTorch
torch.manual_seed(seed_value)

# Set seed for Hugging Face Transformers
set_seed(seed_value)


#################################


def LoadTokenizerModel(args_dict):
    """
        args_dict is a dictionary containing at least:
            'model_name' (str): name of the model to load
            'token'      (str): API token
            'cache_dir'  (str): cache directory

        return:
            model, tokenizer, and the device to map input
    """

    tokenizer = AutoTokenizer.from_pretrained(args_dict['cache_dir'],  # args_dict['model_name'],
                                              # token=args_dict['token'],
                                              # cache_dir=args_dict['cache_dir']
                                              local_files_only=True,
                                              )
    # Set the pad_token to eos_token if pad_token doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set the chat template in the tokenizer
    tokenizer.chat_template = chat_template

    # Use init_empty_weights to allocate model layers with empty weights
    model = AutoModelForCausalLM.from_pretrained(
        args_dict['cache_dir'],  # args_dict['model_name'],
        # token=args_dict['token'],
        # cache_dir=args_dict['cache_dir']
        local_files_only=True,
        # args_dict['model_name'],
        torch_dtype=torch.float16,  # Use FP16 precision to save memory
        device_map='auto',  # Automatically select device (CPU/GPU)
        # token=args_dict['token'],
        # cache_dir=args_dict['cache_dir']
    )

    print("Model and tokenizer are ready for inference.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    return model, tokenizer, device


##################################

def SendToLLM(messages: list,
              model=None,
              tokenizer=None,
              device=None,
              temperature=1.2,
              nucleus=0.0,
              alternatives=2,
              max_tokens=1024) -> Tuple:
    """

    :param messages:  the list of messages
    :param model:  the Large language model
    :param temperature:  the temperature
    :param max_tokens: the max number of tokens generated

    :return: response_message, messages, generation_probability.item(), generation_time

            NB: the list of messages is updated even if it is not returned. since it is a list. remember this behavior in python
    """
    # print(f"device: {device}")

    # Tokenize Messages
    inputs = tokenizer.apply_chat_template(conversation=messages,
                                           chat_template=chat_template,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_dict=True,
                                           padding=True,
                                           return_tensors="pt",
                                           continue_final_message=False).to(device)

    # print(tokenizer.decode(inputs['input_ids']))
    input_ids = inputs['input_ids']  # Extract the input_ids tensor
    if input_ids.shape[1] > int(max_tokens * 0.9):
        # print(f"max_tokens: {max_tokens} | {input_ids.shape[1]}")
        max_tokens = input_ids.shape[1] + max_tokens

    #
    # print(f"input_ids device: {input_ids.device}")
    # print(f"model device: {next(model.parameters()).device}")

    # print(f"max_tokens: {max_tokens}")
    #
    # # Convert each tensor of token IDs back to a list of integers
    # decoded_inputs = [tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids.tolist()]

    # # Print the decoded inputs
    # for i, decoded_text in enumerate(decoded_inputs):
    #     print(f"Decoded Input {i}: {type(decoded_text)=} {decoded_text=}")
    #

    # Generate response
    if temperature == 0.0 and nucleus == 0.0:
        do_sampling = False
        alternatives = 1
        temperature = None
        nucleus = None
    elif temperature == 0.0 and nucleus != 0.0:
        #  set temperature to 0.1
        temperature = 0.1
        do_sampling = True
    else:
        do_sampling = True

    # print(f"do_sampling: {do_sampling} | {alternatives=} | {temperature=} | {nucleus=}")
    # Ensure max_length does not exceed the model's capacity
    # max_length = min(max_length, model.config.max_position_embeddings)
    start_time = time()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):  # Use mixed precision for better performance
            # Generate text
            # model.eval()
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sampling,
                temperature=temperature,
                top_p=nucleus,
                # top_k=1550,
                num_return_sequences=alternatives,
                return_dict_in_generate=True,
                output_scores=True
            )
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True  # normalize SHOULD be true?
            )
    generation_time = time() - start_time

    # print(f"Time taken to generate: {time() - start_time:.2f}s")
    #  move from cuda to cpu
    transition_scores = transition_scores.cpu()

    # output_length = inputs.input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
    # print(f"output_length: {output_length}")
    # length_penalty = model.generation_config.length_penalty

    probabilities = torch.exp(transition_scores.sum(axis=1) / 1)  # no penality given (output_length, length_penalty)
    #  convert to float
    probabilities = probabilities.float()
    # print(f"probabilities: {probabilities}")

    generated_sequences = [tokenizer.decode(seq, skip_special_tokens=False) for seq in outputs['sequences']]
    # text_prob = zip(generated_sequences, probabilities.tolist())
    # take the highest probability
    high_prob_text = generated_sequences[probabilities.argmax()]
    generation_probability = probabilities.max()

    #  remove input messages from the generated text
    # print(f"high_prob_text: p({generation_probability}) | {high_prob_text}")

    response_message = high_prob_text[high_prob_text.rfind(PATTERN_TO_REMOVE) + len(PATTERN_TO_REMOVE):]
    response_message = response_message.replace("<|eot_id|>", "").strip()

    # print(f"response_message: {response_message}, {generation_probability=}")

    return response_message, messages, generation_probability.item(), generation_time


def update_messages(messages: list,
                    role: str,
                    query: str,
                    ):
    messages.append({"role": role,
                     "content": query})
    return messages


def clean_list(list_string: str,
               model=None,
               tokenizer=None,
               device=None,
               temperature=1.2,
               nucleus=0.0,
               alternatives=2,
               max_tokens=4096) -> str:
    """

    :param list_string: (str) the list to clean
    :param model:  (str)the Large language model
    :param temperature: (float) the temperature
    :param max_tokens: (int) the max number of tokens generated
    :return: (str)the text generated

    """

    messages = list()
    messages.append({"role": "system",
                     "content": CLEAN_LIST_SYSTEM})
    messages.append({"role": "user",
                     "content": CLEAN_LIST_USER.format(text=list_string)})
    response_message, _, _ = SendToLLM(messages=messages,
                                       model=model,
                                       tokenizer=tokenizer,
                                       device=device,
                                       temperature=temperature,
                                       nucleus=nucleus,
                                       alternatives=alternatives,
                                       max_tokens=max_tokens)
    return response_message


def extract_list_items(list_string: str) -> list:
    return [item.strip() for item in list_string.split('\n')]


def extract_base(base_prompt: str,
                 string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=4096,
                 ) -> str:
    """

        base_prompt is the prompt to use to extract the data from

                    Base extract method

    """

    messages = list()
    messages.append({"role": "system",
                     "content": base_prompt})
    messages.append({"role": "user",
                     "content": EXTRACT_BASE_USER.format(text=string)})
    response_message, _, _, _ = SendToLLM(messages=messages,
                                          model=model,
                                          tokenizer=tokenizer,
                                          device=device,
                                          temperature=temperature,
                                          nucleus=nucleus,
                                          max_tokens=max_tokens)
    return response_message


def extract_name(list_string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_NAME_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_description(list_string: str,
                        model=None,
                        tokenizer=None,
                        device=None,
                        temperature=1.2,
                        nucleus=0.0,
                        max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_DESCRIPTION_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_grade(list_string: str,
                  model=None,
                  tokenizer=None,
                  device=None,
                  temperature=1.2,
                  nucleus=0.0,
                  max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_GRADE_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_justification(list_string: str,
                          model=None,
                          tokenizer=None,
                          device=None,
                          temperature=1.2,
                          nucleus=0.0,
                          max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_JUSTIFICATION_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_final_answer(list_string: str,
                         model=None,
                         tokenizer=None,
                         device=None,
                         temperature=1.2,
                         nucleus=0.0,
                         max_tokens=2048) -> str:
    response_message = extract_base(base_prompt=EXTRACT_FINAL_ANSWER_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_confidence_score(list_string: str,
                             model=None,
                             tokenizer=None,
                             device=None,
                             temperature=1.2,
                             nucleus=0.0,
                             max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_CONFIDENCE_SCORE_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def extract_reasoning_steps(list_string: str,
                            model=None,
                            tokenizer=None,
                            device=None,
                            temperature=1.2,
                            nucleus=0.0,
                            max_tokens=2048) -> list:
    return extract_base(base_prompt=EXTRACT_REASONING_STEPS_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )
    # return sent_tokenize(response_message, language='english')
    # return extract_list_items(response_message)


def extract_conclusion(list_string: str,
                       model=None,
                       tokenizer=None,
                       device=None,
                       temperature=1.2,
                       nucleus=0.0,
                       max_tokens=4096) -> str:
    response_message = extract_base(base_prompt=EXTRACT_CONCLUSION_SYSTEM,
                                    string=list_string,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    temperature=temperature,
                                    nucleus=nucleus,
                                    max_tokens=max_tokens,
                                    )
    return response_message


def create_experts_answers_string(query_answers: list, experts: list) -> str:
    experts_answers_string = '[\n'
    for ans, expert in zip(query_answers, experts):
        expert_string = "{\n"
        expert_string += f"\"expert-name\": \"{expert['name']}\",\n"
        expert_string += f"\"expert-field\": \"{expert['field']}\",\n"
        expert_string += f"\"answer\": \"{ans['final_answer']}\",\n"
        expert_string += f"\"grade\": \"{ans['grade']}\",\n"
        expert_string += f"\"confidence-score\": \"{ans['confidence-score']}\",\n"
        expert_string += f"\"justification\": \"{ans['justification']}\",\n"
        expert_string += f"\"reasoning-steps\": \"{ans['reasoning-steps']}\",\n"
        expert_string += f"\"conclusion\": \"{ans['conclusion']}\"\n"
        expert_string += '}\n'

        experts_answers_string += expert_string
    # print(f"{experts_answers_string=}")

    experts_answers_string += '\n]'

    return experts_answers_string
