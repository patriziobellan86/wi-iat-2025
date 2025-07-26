from ModelRequests import extract_grade, extract_confidence_score, \
    extract_reasoning_steps, extract_conclusion, extract_justification, extract_final_answer
from ModelRequests import SendToLLM, create_experts_answers_string
from prompts.experts import ASK_TO_EXPERT_USER, ASK_TO_EXPERT_SYSTEM
from prompts.final_decision_maker import ASK_FINAL_ANSWER_USER, ASK_FINAL_ANSWER_SYSTEM

import json
from tqdm import tqdm


def AskToExperts(args_dict: dict,
                 experts,
                 query=None):
    """
            This method makes a run of conversation where each expert propose its idea.
            Here there is no critique to the others. Each expert is independent.

            return answers
    """
    answers = list()
    for expert_ID, expert in enumerate(tqdm(experts, desc="Experts processed", ascii=True)):

        messages = [{"role": "system",
                     "content": ASK_TO_EXPERT_SYSTEM.format(expert_description=expert['description'],
                                                            task=args_dict['task'],
                                                            context=args_dict['context'])},
                    {"role": "user",
                     "content": ASK_TO_EXPERT_USER.format(query=query)}]

        raw_answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                                       model=args_dict['model'],
                                                                       tokenizer=args_dict['tokenizer'],
                                                                       device=args_dict['device'],
                                                                       temperature=args_dict['temperature'],
                                                                       nucleus=args_dict['nucleus'],
                                                                       max_tokens=512,
                                                                       )
        try:
            tmp_answer = raw_answer.replace("```json\n", "").replace("```", "")
            tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
            tmp_dict = json.loads(tmp_answer)
            expert_final_answer = tmp_dict['final_answer']
            grade = tmp_dict['grade']
            confidence_score = tmp_dict['confidence_score']
            reasoning_steps = tmp_dict['reasoning_steps']
            justification = tmp_dict['justification']
            conclusion = tmp_dict['conclusion']

        except:
            expert_final_answer = extract_final_answer(raw_answer,
                                                       model=args_dict['model'],
                                                       tokenizer=args_dict['tokenizer'],
                                                       device=args_dict['device'],
                                                       temperature=0,
                                                       nucleus=0,
                                                       max_tokens=128,
                                                       )
            grade = extract_grade(raw_answer,
                                  model=args_dict['model'],
                                  tokenizer=args_dict['tokenizer'],
                                  device=args_dict['device'],
                                  temperature=0,
                                  nucleus=0,
                                  max_tokens=12,
                                  )
            confidence_score = extract_confidence_score(raw_answer,
                                                        model=args_dict['model'],
                                                        tokenizer=args_dict['tokenizer'],
                                                        device=args_dict['device'],
                                                        temperature=0,
                                                        nucleus=0,
                                                        max_tokens=12,
                                                        )
            reasoning_steps = extract_reasoning_steps(raw_answer,
                                                      model=args_dict['model'],
                                                      tokenizer=args_dict['tokenizer'],
                                                      device=args_dict['device'],
                                                      temperature=0,
                                                      nucleus=0,
                                                      max_tokens=256,
                                                      )
            justification = extract_justification(raw_answer,
                                                  model=args_dict['model'],
                                                  tokenizer=args_dict['tokenizer'],
                                                  device=args_dict['device'],
                                                  temperature=0,
                                                  nucleus=0,
                                                  max_tokens=128,
                                                  )

            conclusion = extract_conclusion(raw_answer,
                                            model=args_dict['model'],
                                            tokenizer=args_dict['tokenizer'],
                                            device=args_dict['device'],
                                            temperature=0,
                                            nucleus=0,
                                            max_tokens=128,
                                            )

        expert_answer_dict = {'expert_ID': expert_ID,
                              'generated-data': raw_answer,
                              'generation-probability': probability,
                              'generation-time': generation_time,

                              'final_answer': expert_final_answer,
                              'grade': grade,
                              'confidence-score': confidence_score,
                              'reasoning-steps': reasoning_steps,
                              'justification': justification,
                              'conclusion': conclusion
                              }
        answers.append(expert_answer_dict)
    return answers


def MakeFinalDecision(args_dict: dict,
                      final_decision_maker,
                      experts,
                      # model=None, tokenizer=None, device=None,
                      query="", query_answers=list()):
    experts_answers = create_experts_answers_string(query_answers, experts)
    messages = [{"role": "system",
                 "content": ASK_FINAL_ANSWER_SYSTEM.format(
                     final_decison_maker_description=final_decision_maker['description'],
                     task=args_dict['task'],
                     context=args_dict['context'],
                 )
                 }, {"role": "user", "content": ASK_FINAL_ANSWER_USER.format(query=query,
                                                                             experts_answers=experts_answers,
                                                                             )
                     }]

    raw_answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                                   model=args_dict['model'],
                                                                   tokenizer=args_dict['tokenizer'],
                                                                   device=args_dict['device'],
                                                                   temperature=args_dict['temperature'],
                                                                   nucleus=args_dict['nucleus'],
                                                                   max_tokens=1024)

    try:
        tmp_answer = raw_answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
        tmp_dict = json.loads(tmp_answer)
        reasoning_steps = tmp_dict['reasoning_steps']
        conclusion = tmp_dict['conclusion']
        final_answer = tmp_dict['final_answer']

    except:
        #  using LLM strategy

        reasoning_steps = extract_reasoning_steps(raw_answer,
                                                  model=args_dict['model'],
                                                  tokenizer=args_dict['tokenizer'],
                                                  device=args_dict['device'],
                                                  temperature=0,
                                                  nucleus=0,
                                                  max_tokens=512,
                                                  )

        conclusion = extract_conclusion(raw_answer,
                                        model=args_dict['model'],
                                        tokenizer=args_dict['tokenizer'],
                                        device=args_dict['device'],
                                        temperature=0,
                                        nucleus=0,
                                        max_tokens=256,
                                        )
        final_answer = extract_final_answer(raw_answer,
                                            model=args_dict['model'],
                                            tokenizer=args_dict['tokenizer'],
                                            device=args_dict['device'],
                                            temperature=0,
                                            nucleus=0,
                                            max_tokens=256,
                                            )

    return final_answer, reasoning_steps, conclusion, raw_answer, messages, probability, generation_time


def Baseline(args_dict, query=""):
    messages = list()
    messages.append({"role": "user", "content": query})

    answer, messages, probability, generation_time = SendToLLM(messages=messages,
                                                               model=args_dict['model'],
                                                               tokenizer=args_dict['tokenizer'],
                                                               device=args_dict['device'],
                                                               temperature=args_dict['temperature'],
                                                               nucleus=args_dict['nucleus'],
                                                               max_tokens=1024)
    return answer, messages, probability, generation_time
