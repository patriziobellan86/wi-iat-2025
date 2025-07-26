from CreateExperts import get_experts_fields
from ModelRequests import extract_description, extract_name, SendToLLM
from prompts.final_decision_maker import CREATION_FINAL_DECISOR_USER, CREATION_FINAL_DECISOR_SYSTEM


def CreateFinalDecisionMaker(args_dict,
                             experts
                             ):
    experts_fields_list = get_experts_fields(experts)
    messages = list()
    messages.append({"role": "system",
                     "content": CREATION_FINAL_DECISOR_SYSTEM.format(
                         psychologist_description=args_dict['psychologist']['description'],
                         description_framework=args_dict['description_framework'],
                         experts_fields=experts_fields_list,
                         task=args_dict['task'],
                         context=args_dict['context'])
                     })
    messages.append({"role": "user",
                     "content": CREATION_FINAL_DECISOR_USER.format(
                         description_framework=args_dict['description_framework'],
                         # experts_fields=experts_fields_list,
                         task=args_dict['task'],
                         context=args_dict['context'])})

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
        tmp_final_decisor_dict = json.loads(tmp_answer)
        final_decisor_name = tmp_final_decisor_dict['name']
        final_decisor_description = tmp_final_decisor_dict['description']

    except:
        final_decisor_name = extract_name(answer,
                                          model=args_dict['model'],
                                          tokenizer=args_dict['tokenizer'],
                                          device=args_dict['device'],
                                          temperature=0,
                                          nucleus=0,
                                          max_tokens=12)
        final_decisor_description = extract_description(answer,
                                                        model=args_dict['model'],
                                                        tokenizer=args_dict['tokenizer'],
                                                        device=args_dict['device'],
                                                        temperature=0,
                                                        nucleus=0,
                                                        max_tokens=1024)
    print(f"--------------\nCreated Final Decision Maker: {final_decisor_name}\nDescription: {final_decisor_description}\n--------------")
    #  save the data generated up to here
    final_decisor_dict = {'prompt': messages,
                          'generated-data': answer,
                          'generation-probability': probability,
                          'generation-time': generation_time,

                          'name': final_decisor_name,
                          'description': final_decisor_description,
                          }
    #
    # with codecs.open(args_dict['final-decision-maker-filename'], 'w', 'utf-8') as json_file:
    #     json.dump(final_decisor_dict, json_file, indent=4)

    return final_decisor_dict
