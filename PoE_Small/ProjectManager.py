import json

from ModelRequests import SendToLLM, extract_name, extract_description

from prompts.project_manager import CREATION_PROJECTMANAGER_USER, CREATION_PROJECTMANAGER_SYSTEM


def CreateProjectManager(args_dict):
    messages = [{"role": "system",
                 "content": CREATION_PROJECTMANAGER_SYSTEM.format(
                     psychologist_description=args_dict['psychologist']['description'],
                     description_framework=args_dict['description_framework'])},
                {"role": "user",
                 "content": CREATION_PROJECTMANAGER_USER.format(task=args_dict['task'],
                                                                context=args_dict['context'],
                                                                description_framework=args_dict[
                                                                    'description_framework'])}]

    answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                               model=args_dict['model'],
                                                               tokenizer=args_dict['tokenizer'],
                                                               device=args_dict['device'],
                                                               temperature=args_dict['temperature'],
                                                               nucleus=args_dict['nucleus'],
                                                               alternatives=args_dict['alternatives'],
                                                               max_tokens=1024)
    #  if the data is parsable as JSON use it, otherwise, use the extracted data
    try:
        tmp_answer = answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
        tmp_dict = json.loads(tmp_answer)
        name = tmp_dict['name']
        description = tmp_dict['description']

    except:
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
    print(f"--------------\nCreated Project Manager: {name}\nDescription: {description}\n--------------")
    #  save the data generated up to here
    return {'prompt': messages,
            'generated-data': answer,
            'generation-probability': probability,
            'generation-time': generation_time,

            'name': name,
            'description': description,
            }
