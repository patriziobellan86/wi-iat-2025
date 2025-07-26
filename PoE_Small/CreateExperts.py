from ModelRequests import SendToLLM, clean_list, extract_list_items, extract_name, extract_description
from FileToolkit import SaveExpertsFieldList
from prompts.project_manager import CREATION_LIST_EXPERTS_SYSTEM, CREATION_LIST_EXPERTS_USER
from prompts.experts import CREATE_EXPERT_IN_A_FIELD_SYSTEM, CREATE_EXPERT_IN_A_FIELD_USER
from prompts.final_decision_maker import CREATION_FINAL_DECISOR_USER, CREATION_FINAL_DECISOR_SYSTEM

import json
from tqdm import tqdm


def get_experts_fields(experts_list):
    """
        return the list of fields of the experts
    """
    fields = list()
    # with open(args_dict['experts-list-filename'], 'r') as f:
    #     experts_list = json.load(f)
    for expert in experts_list:
        print(f"expert: {expert}")
        field = expert['field']
        fields.append(field)
    return fields


def CreateExperts(args_dict,
                  model=None,
                  tokenizer=None,
                  device=None
                  ):
    #####   Create Experts  #####
    messages = [{"role": "system",
                 "content": CREATION_LIST_EXPERTS_SYSTEM.format(
                     projectmanager_description=args_dict['project-manager']['description'])},
                {"role": "user",
                 "content": CREATION_LIST_EXPERTS_USER.format(task=args_dict['task'],
                                                              context=args_dict['context'],
                                                              max_experts_number=args_dict['max_experts_number'])}]

    answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                                                model=args_dict['model'],
                                                                                tokenizer=args_dict['tokenizer'],
                                                                                device=args_dict['device'],
                                                                                temperature=args_dict['temperature'],
                                                                                nucleus=args_dict['nucleus'],
                                                                                alternatives=args_dict['alternatives'],
                                                                                max_tokens=256)

    try:
        tmp_answer = answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')

        experts_list = json.loads(tmp_answer)

    except:
        experts_list = extract_list_items(clean_list(answer,
                                                     model=model,
                                                     tokenizer=tokenizer,
                                                     device=device,
                                                     temperature=0.0,  # temperature,
                                                     nucleus=0.0,  # nucleus,
                                                     alternatives=1,
                                                     max_tokens=256))

    if len(experts_list) > args_dict['max_experts_number']:
        print(f"\n||| \t\t Experts list too long, selecting the first {args_dict['max_experts_number']}")
        experts_list = experts_list[:args_dict['max_experts_number']]

    print(f'\n\tExperts list:\n\t\t{experts_list}')

    #  save the data generated up to here
    experts_field = {'prompt': messages,
                     'generated-data': answer,
                     'generation-probability': probability,
                     'generation-time': generation_time,

                     'list': experts_list,
                     }

    SaveExpertsFieldList(args_dict, experts_field)

    #  create experts
    experts = list()
    for expert_field in tqdm(experts_list, desc="Creating Experts", ascii=True):
        messages = [{"role": "system",
                     "content": CREATE_EXPERT_IN_A_FIELD_SYSTEM.format(
                         psychologist_description=args_dict['psychologist']['description'],
                         description_framework=args_dict['description_framework'])},
                    {"role": "user",
                     "content": CREATE_EXPERT_IN_A_FIELD_USER.format(
                         field=expert_field,
                         description_framework=args_dict['description_framework'])}]

        answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                                   model=args_dict['model'],
                                                                   tokenizer=args_dict['tokenizer'],
                                                                   device=args_dict['device'],
                                                                   temperature=args_dict['temperature'],
                                                                   nucleus=args_dict['nucleus'],
                                                                   max_tokens=1024)

        try:
            tmp_answer = answer.replace("```json\n", "").replace("```", "")
            tmp_dict = json.loads(tmp_answer)
            name = tmp_dict['name']
            description = tmp_dict['description']

        except:
            #  extract data using LLM
            name = extract_name(answer,
                                model=args_dict['model'],
                                tokenizer=args_dict['tokenizer'],
                                device=args_dict['device'],
                                temperature=0,
                                nucleus=0,
                                max_tokens=12)
            description = extract_description(answer,
                                              model=args_dict['model'],
                                              tokenizer=args_dict['tokenizer'],
                                              device=args_dict['device'],
                                              temperature=0,
                                              nucleus=0,
                                              max_tokens=1024)
        print(f"--------------\nCreated Expert: {name} field:{experts_field}\nDescription: {description}\n--------------")
        experts.append({
            'prompt': messages,
            'generated-data': answer,
            'generation-probability': probability,
            'generation-time': generation_time,
            "name": name,
            "description": description,
            'field': expert_field,
        })

    return experts

