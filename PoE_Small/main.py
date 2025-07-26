from CreateExperts import CreateExperts
from FinalDecisionMaker import CreateFinalDecisionMaker
from Experiment import  RunExperiment
from ModelRequests import LoadTokenizerModel
from FileToolkit import check_input_file, update_args_dict, check_output_dir, save_args_dict, check_psychologist_exist, \
    LoadPsychologist, SavePsychologist, LoadProjectManager, SaveProjectManager, check_project_manager_exist, \
    SaveExperts, LoadExperts, check_experts_exist, check_final_decision_maker_exist, SaveFinalDecisionMaker, \
    LoadFinalDecisionMaker
from ProjectManager import CreateProjectManager
from Psychologist import CreatePsychologist


def RunFramework(args_dict):

    # check input file
    check_input_file(args_dict)
    #  update args_dict
    update_args_dict(args_dict)
    #  check output-dir does not exist.
    check_output_dir(args_dict)
    #  save args_dict
    save_args_dict(args_dict)

    print("loading model and tokenizer")
    #  load model and tokenizer
    args_dict['model'], args_dict['tokenizer'], args_dict['device'] = LoadTokenizerModel(args_dict)

    if 'baseline' in args_dict and args_dict['baseline']:
        print("Running Baseline ONLY")

    else:
        ##################################################
        ####   PSYCHOLOGIST ##############################
        #  Psychologist, used to describe agents
        if check_psychologist_exist(args_dict):
            #  load it
            args_dict['psychologist'] = LoadPsychologist(args_dict)
        else:
            print('\n\t\t\tCreating Psychologist')
            args_dict['psychologist'] = CreatePsychologist(args_dict)
            SavePsychologist(args_dict, args_dict['psychologist'] )
        save_args_dict(args_dict)

        ##################################################
        ####   Project Manager ###########################
        #  Project Manager, used to select expertise fields'
        if check_project_manager_exist(args_dict):
            #  load it
            args_dict['project-manager'] = LoadProjectManager(args_dict)
        else:
            print('\n\t\t\tCreating Project Manager')
            args_dict['project-manager'] = CreateProjectManager(args_dict)
            SaveProjectManager(args_dict, args_dict['project-manager'] )
        save_args_dict(args_dict)

        ##################################################
        ####   The  Experts    ###########################
        #  Experts, used to describe agents

        #  create experts if they do not exist
        if check_experts_exist(args_dict):
            #  load it
            experts = LoadExperts(args_dict)
            print('\n\t\t\tExperts already exist')
        else:
            print('\n\t\t\tCreating experts')
            experts = CreateExperts(args_dict)
            SaveExperts(args_dict, experts)
        save_args_dict(args_dict)

        #  create final decisor if it does not exist
        if check_final_decision_maker_exist(args_dict):
            print('\n\n\t\tFinal decision maker already exist\n\n')
            # final_decision_maker = LoadFinalDecisionMaker(args_dict)
        else:
            print('\n\n\t\tCreating final decision maker\n\n')
            final_decision_maker = CreateFinalDecisionMaker(args_dict, experts)
            SaveFinalDecisionMaker(args_dict, final_decision_maker)
        save_args_dict(args_dict)
    #  run the experiment
    RunExperiment(args_dict)

    print("Experiment completed")
