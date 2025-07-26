import codecs
import json
import os
from pathlib import Path


def check_output_dir(args_dict):
    """
        Check the ouput directory exist, otherwise create it
    """
    os.makedirs(args_dict['output_dir'], exist_ok=True)


def check_input_file(args_dict):
    """
        Check input file exist, otherwise throw a FileNotFoundError
    """
    if not os.path.isfile(args_dict['input']):
        raise FileNotFoundError(f"{args_dict['input']} is not a file")

def update_args_dict(args_dict: dict):
    """
        Add filenames to args_dict
    """
    #  update output_dir
    output_dir = list(Path(args_dict['output_dir']).parts)

    experiment_name = "-".join([str(args_dict[x]).strip() for x in ['description_framework', 'temperature', 'nucleus', 'alternatives']])
    experiment_name += f"-{args_dict['model_name'].split('/')[-1]}"

    output_dir.append(experiment_name)

    args_dict['output_dir'] = Path().joinpath(*output_dir).absolute().__str__()

    print(f"output_dir: {args_dict['output_dir']}")

    #  psychologist-creator-filename
    args_dict['psychologist-filename'] = Path().joinpath(args_dict['output_dir'],
                                                              "psychologist.json").absolute().__str__()
    #  project-manager-creator-filename
    args_dict['project-manager-filename']= Path().joinpath(args_dict['output_dir'],
                                                              "project-manager.json").absolute().__str__()

    #  experts-list-filename
    args_dict['experts-list-filename'] = Path().joinpath(args_dict['output_dir'],
                                                            "experts-list.json").absolute().__str__()

    #  experts-list-filename
    args_dict['experts-filename'] = Path().joinpath(args_dict['output_dir'],
                                                         "experts.json").absolute().__str__()

    #  final-decisor-filename
    args_dict['final-decision-maker-filename'] = Path().joinpath(args_dict['output_dir'],
                                                          "final_decisor.json").absolute().__str__()


    #  update queries_answers_filename
    args_dict['queries_answers_filename'] = Path().joinpath(args_dict['output_dir'],
                                                            "queries_answers.json").absolute().__str__()

    print(f"args_dict: {args_dict}")

def save_args_dict(args_dict):
    """
        Save the args_dict into a JSON

        Exclude the model and the tokenizer
    """
    #  filter out model and tokenizer from args_dict
    args_dict = {k: v for k, v in args_dict.items() if k not in ['model', 'tokenizer']}

    #  save args_dict
    with open(f"{args_dict['output_dir']}/args_dict.json", 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"args_dict saved to {args_dict['output_dir']}/args_dict.json")


def read_args_dict(args_dict_filename):
    """
        load args_dict from disk
    """
    with open(args_dict_filename, 'r') as f:
        args_dict = json.load(f)

    return args_dict

def LoadPsychologist(args_dict):
    """
        Load psychologist from disk
    """
    with open(args_dict['psychologist-filename'], 'r') as f:
        psychologist = json.load(f)

    return psychologist

def SavePsychologist(args_dict, psychologist):
    with codecs.open(args_dict['psychologist-filename'], 'w', 'utf-8') as json_file:
        json.dump(psychologist, json_file, indent=4)

def LoadProjectManager(args_dict):
    """
        Load Project Manager from disk
    """
    with open(args_dict['project-manager-filename'], 'r') as f:
        projectmanager = json.load(f)

    return projectmanager

def SaveProjectManager(args_dict, project_manager):
    with codecs.open(args_dict['project-manager-filename'], 'w', 'utf-8') as json_file:
        json.dump(project_manager, json_file, indent=4)

def check_psychologist_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'psychologist-filename' in args_dict:
        if os.path.exists(args_dict['psychologist-filename']):
            return True
        else:
            return False


def check_project_manager_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'project-manager-filename' in args_dict:
        if os.path.exists(args_dict['project-manager-filename']):
            return True
        else:
            return False

def check_experts_list_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'experts-list-filename' in args_dict:
        if os.path.exists(args_dict['experts-list-filename']):
            return True
        else:
            return False


def check_final_decisor_exist(args_dict):
    """
        check if the final decisor exist
    """

    if 'final-decision-maker-filename' in args_dict:
        if os.path.exists(args_dict['final-decision-maker-filename']):
            return True
        else:
            return False


def SaveExpertsFieldList(args_dict, experts_field):
    with codecs.open(args_dict['experts-list-filename'], 'w', 'utf-8') as json_file:
        json.dump(experts_field, json_file, indent=4)
def SaveExperts(args_dict, experts):
    #  save the data generated up to here
    with codecs.open(args_dict['experts-filename'], 'w', 'utf-8') as json_file:
        json.dump(experts, json_file, indent=4)
def check_experts_exist(args_dict):
    return os.path.exists(args_dict['experts-filename'])


def LoadExperts(args_dict):
    """
        Load Project Manager from disk
    """
    with open(args_dict['experts-filename'], 'r') as f:
        experts = json.load(f)

    return experts

def check_final_decision_maker_exist(args_dict):
    return os.path.exists(args_dict['final-decision-maker-filename'])

def SaveFinalDecisionMaker(args_dict, final_decision_maker):
    with codecs.open(args_dict['final-decision-maker-filename'], 'w', 'utf-8') as json_file:
        json.dump(final_decision_maker, json_file, indent=4)
def LoadFinalDecisionMaker(args_dict):
    """
        Load Project Manager from disk
    """
    with open(args_dict['final-decision-maker-filename'], 'r') as f:
        return json.load(f)

def SaveQueriesAnswers(args_dict, queries_answers):
    #  save after each query.
    with codecs.open(args_dict['queries_answers_filename'], 'w', 'utf-8') as json_file:
        json.dump(queries_answers, json_file, indent=4)


