from ModelRequests import SendToLLM, extract_description, extract_name
from prompts.psychologist import CREATION_PSYCHOLOGIST_SYSTEM, CREATION_PSYCHOLOGIST_USER
import json


def CreatePsychologist(args_dict):
    messages = [{"role": "system",
                 "content": CREATION_PSYCHOLOGIST_SYSTEM.format(
                     description_framework=args_dict['description_framework'])
                 },
                {"role": "user",
                 "content": CREATION_PSYCHOLOGIST_USER.format(
                     description_framework=args_dict['description_framework'])}]

    answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                               model=args_dict['model'],
                                                               tokenizer=args_dict['tokenizer'],
                                                               device=args_dict['device'],
                                                               temperature=args_dict['temperature'],
                                                               nucleus=args_dict['nucleus'],
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
                            temperature=args_dict['temperature'],
                            nucleus=args_dict['nucleus'],
                            max_tokens=12)
        description = extract_description(answer,
                                          model=args_dict['model'],
                                          tokenizer=args_dict['tokenizer'],
                                          device=args_dict['device'],
                                          temperature=args_dict['temperature'],
                                          nucleus=args_dict['nucleus'],
                                          max_tokens=1024)
    print(f"--------------\nCreated Psychologist: {name}\nDescription: {description}\n--------------")
    #  save the data generated up to here
    return {'prompt': messages,
            'generated-data': answer,
            'generation-probability': probability,
            'generation-time': generation_time,

            'name': name,
            'description': description,
            }
