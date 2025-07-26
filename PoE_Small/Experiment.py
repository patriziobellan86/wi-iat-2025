import codecs
import json

from ExpertsConversation import AskToExperts, MakeFinalDecision, Baseline

from tqdm import tqdm

from FileToolkit import SaveQueriesAnswers, LoadExperts, LoadFinalDecisionMaker


def get_queries_last_index(queries_answers):
    #  get the query number of the last inserted item
    try:
        return len(queries_answers)
    except:
        return 0


def RunExperiment(args_dict):
    #  read the queries
    queries = list()
    with codecs.open(args_dict['input'], 'r', 'utf-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            queries.append(line)

    #  try to open queries answers file to resume, if exist
    queries_last_index = 0
    try:
        with open(args_dict['queries_answers_filename'], 'r') as json_file:
            queries_answers = json.load(json_file)  # Load the JSON file into a Python dictionary
        if args_dict['resume']:
            queries_last_index = get_queries_last_index(queries_answers)

    except FileNotFoundError:
        queries_answers = dict()

    if not args_dict['baseline']:
        experts = LoadExperts(args_dict)
        final_decision_maker = LoadFinalDecisionMaker(args_dict)

    for n_query, query in enumerate(tqdm(queries[queries_last_index:], desc="Queries processed", ascii=True)):
        n_query += queries_last_index

        if args_dict['baseline']:
            #  run baseline by explicitly asking for the final answer to the bare model
            baseline_answer, _, baseline_generation_probability, generation_time = Baseline(args_dict,
                                                                           query=query)

            #  save the query answers
            queries_answers[n_query] = {"query": query,

                                        'final-decison-maker-generation-probability': '',
                                        "final-decison-maker-generated-data": '',
                                        'final-decison-maker-generation-time': '',

                                        'final-decison-maker-final_answer': '',
                                        'final-decison-maker-reasoning-steps': '',
                                        'final-decison-maker-conclusion': '',

                                        "query_answers": list(),

                                        "baseline-answer": baseline_answer,
                                        "baseline-generation-probability": baseline_generation_probability,
                                        "baseline-generation-time": generation_time,
                                        }
        else:

            query_answers = AskToExperts(args_dict=args_dict,
                                         experts=experts,
                                         query=query)

            final_answer, reasoning_steps, conclusion, raw_answer, messages, probability, generation_time = MakeFinalDecision(
                args_dict,
                final_decision_maker=final_decision_maker,
                experts=experts,
                query=query,
                query_answers=query_answers)

            queries_answers[n_query] = {"query": query,
                                        'final-decison-maker-generation-probability': probability,
                                        'final-decison-maker-generation-time': generation_time,
                                        "final-decison-maker-generated-data": raw_answer,

                                        'final-decison-maker-answer': final_answer,
                                        'final-decison-maker-reasoning-steps': reasoning_steps,
                                        'final-decison-maker-conclusion': conclusion,

                                        "query_answers": query_answers,

                                        "baseline-answer": '',
                                        "baseline-generation-probability": '',
                                        "baseline-generation-time": '',
                                        }
            SaveQueriesAnswers(args_dict, queries_answers)
