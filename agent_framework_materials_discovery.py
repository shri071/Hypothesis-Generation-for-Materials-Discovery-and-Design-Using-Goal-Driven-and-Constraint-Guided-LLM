import os
import pandas as pd
import json
import openai
from openai import OpenAI
from typing import List
import re

client = OpenAI(api_key='sk-svcacct-oEyJONnuUBgTc-g35W18i1Jxu7WOHPQz_TzWRjaTS9bKqv-CT-KEjwFzqwIZhhMYH5fT3BlbkFJ43pHN8gA85m8CdI3uNKaqOuiOM_yXF-0YaCqvJGDdaX0HT6aH86HmZ4GbBnWkwcLi8QA')

#=======================================================================================Prompt Construction Functions=======================================================================================

def construct_prompt_for_hypotheses_generator(goal_statement:str, constraint_list:str):
    prompt = f"""{goal_statement} \n\n Constraints:- \n{constraint_list}.\n
Provide me 20 innovative suggestions that will help achieve the above goal while satisfying all of the above mentioned constraints strictly. 
Provide reason for each suggestion. The suggestions must be in the below mentioned format in a JSON object. For example:\n
{{Suggestion_1: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning:
    ,
Suggestion_20: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning: }}"""
    return prompt

def construct_critic_prompt(goal_statement:str, constraint_list:str, chat_history:str):
    critic_prompt = f"""{goal_statement}\n\nConstraints:-\n{constraint_list}\n\nSuggestions:\n{chat_history}Given the above goal statement, constraints and suggestions about materials design and discovery, evaluate each suggestion and generate detailed feedback which will help the suggestion generation process to generate suggestions such that they help achieve goal statement and satisfy all the constraints strictly. The detailed feedback should be in the below JSON format strictly:
    {{"Feedback_for_suggestion_1":
    Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
    Reasoning:" ",
    "Feedback_for_suggestion_20":
    Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
    Reasoning:" ",
    "Overall_Feedback_for_improvement_for_future_suggestion_generation": " " ]]
    }}
    """
    return critic_prompt

def construct_feedback_prompt(feedback):
    feedback_prompt = f"""Below provided is the feedback you gave for each of the initial suggestions generated and an overall feedback for the improvement of future suggestion generations\n{feedback}.Refine your suggestions based on the feedback accordingly to meet the goal statement and satisfy all the constraints strictly. The suggestions must be in the below mentioned format in a JSON object. For example:\n
{{Suggestion_1: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning:
    ,
Suggestion_20: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning:}}"""  
    return feedback_prompt

def construct_feedback_prompt_for_refined_hypotheses(feedback_history,chat_history):
    feedback_prompt = f"""Below provided is the feedback you gave for the initial suggestions\n{feedback_history}. Below are the refined suggestions based on the feedback\n{chat_history}. Now evaluate each refined suggestion and provide detailed feedback which will help the suggestion generation process to generate suggestions such that they help achieve goal statement and satisfy all the constraints strictly. The detailed feedback should be in the below JSON format strictly:
    {{"Feedback_for_suggestion_1":
    Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
    Reasoning:" ",
    "Feedback_for_suggestion_20":
    Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
    Reasoning:" ",
    "Overall_Feedback_for_improvement": " " ]]
    }}
    """
    return feedback_prompt

#=================================================================================================================================================================================================================

#=======================================================================================GPT-4 Response Processing Functions=======================================================================================

def json_to_text(json_obj):
    output_text = ""
    for key, value in json_obj.items():
        suggestion_details = (
            f"{key.replace('_', ' ')}:\n"
            f"Materials:{''.join(value['Materials'])}\n"
            f"Methods_to_develop_the_materials_suggested:{''.join(value['Methods_to_develop_the_materials_suggested'])}\n"
            f"Reasoning:{value[f'Reasoning']}\n\n"
        )
        output_text += suggestion_details
    return output_text

def process_feedback_extract_final_answer(feedback):
    suggestions_with_no = 0
    processed_feedback = ""
    final_answer = "Yes"
    for key, value in feedback.items():
        if key.startswith("Feedback_for_suggestion"):
            suggestion_num = key.split('_')[-1]
            Meets_the_goal_statement_and_satisfies_all_constraints_strictly = value.get('Meets_the_goal_statement_and_satisfies_all_constraints_strictly', 'N/A')
            if Meets_the_goal_statement_and_satisfies_all_constraints_strictly == 'NO':
                suggestions_with_no += 1
                final_answer = "NO"
            Reasoning = value.get('Reasoning', 'N/A')
            processed_feedback += f"Feedback_for_suggestion_{suggestion_num}:\nMeets_the_goal_statement_and_satisfies_all_constraints_strictly:{Meets_the_goal_statement_and_satisfies_all_constraints_strictly}.\nReasoning: {Reasoning}\n\n"
        elif key.startswith("Overall_Feedback_for_improvement"):
            processed_feedback += f"Overall Feedback_for_future_suggestion_improvement: {value}\n"
    return processed_feedback, final_answer, suggestions_with_no

#=================================================================================================================================================================================================================

#=======================================================================================Agent Functions===========================================================================================================

def expert_list_generator(goal_statement):
    completion = client.chat.completions.create(
        model = 'gpt-4o',
        temperature = 0.7,
        messages = [
            {
                'role': 'system',
                'content': f'You are an helpful assistant'
            },
            {
                'role': 'user',
                'content': f'Generate a list of experts required to achieve the below mentioned goal:\n{goal_statement}. Just list the top 5 experts in the format "Expert_1, Expert_2, Expert_3, Expert_4, Expert_5"'
                }
              ]
    )
    return completion.choices[0].message.content



def hypothesis_generator(expert_list,prompt,feedback=None,chat_history=None):
    if feedback == None and chat_history == None:
        completion = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an innovative {expert_list} capable of doing impactful materials discovery and design'
                },
                {
                    'role': 'user',
                    'content': prompt
                },
            ],
            response_format = {"type": "json_object"}
        )
    else:
        completion = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an innovative {expert_list} capable of doing impactful materials discovery and design'
                },
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': chat_history
                },
                {
                    'role': 'user',
                    'content': feedback
                }
            ],
            response_format = {"type": "json_object"}
        )
    return completion.choices[0].message.content

def critic_1(expert_list,critic_prompt,feedback_history,refined_feedback_prompt):
    if feedback_history==None and refined_feedback_prompt==None:
        completion = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly. '
                },
                {
                    'role': 'user',
                    'content': critic_prompt
                }
            ],
            response_format = {"type": "json_object"}
        )
    else: 
        completion = client.chat.completions.create(
        model = 'gpt-4o',
        temperature = 0.7,
        messages = [
            {
                'role': 'system',
                'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly.'
            },
            {
                'role': 'user',
                'content': critic_prompt
            },
            {
                'role': 'assistant',
                'content': feedback_history
            },
            {
                'role': 'user',
                'content': refined_feedback_prompt
            }
        ],
        response_format = {"type": "json_object"}
    )
    return completion.choices[0].message.content

#=================================================================================================================================================================================================================

#=======================================================================================Hypothesis Generation and Refining Process===========================================================================================================

goal_statement = " A self-healing hydrogel that exhibits exceptionally rapid healing.The hydrogel should have an ideal balance between properties such as softness, deformability, ionic and electrical conductivity, self-adhesiveness, response and recovery times, durability, overshoot behavior, and resistanceto nonaxial deformations such as twisting, bending, and pressing"

constraint_list = """ 1) The material must exhibit rapid self-healing, with a recovery time of less than 0.12 seconds, to ensure timely restoration of both mechanical and electrical functions.
 2) The hydrogel must possess ultralow electrical hysteresis (less than 0.64%) under cyclic strains up to 500%, ensuring minimal energy dissipation during repetitive movements.
 3) The material should be highly deformable, with the ability to stretch over 10,000%, while maintaining mechanical integrity in complex, nonaxial deformations such as twisting, bending, and pressing.
 4) The hydrogel must have high ionic and electrical conductivity (greater than 0.074 S m−1) and exhibit strong self-adhesiveness to human skin for effective use in wearable applications.
 5) The material must maintain durability and functionality over long-term use, suitable for monitoring physiological activities such as facial expressions, joint movements, and electrophysiological signals (ECG, EMG, EOG)."""
 
chat_history = []
feedback_history = []
initial_feedback = None
final_answer = ""
feedback_prompt = None
refined_feedback_prompt = None
text_from_feedback = None
suggestions_no_list = []
expert_list = expert_list_generator(goal_statement)

prompt = construct_prompt_for_hypotheses_generator(goal_statement, constraint_list)
print(f'prompt----->')
generated_hypotheses = hypothesis_generator(expert_list,prompt,feedback_prompt)
generated_hypotheses = json.loads(generated_hypotheses)
# generated_hypotheses = json_to_text(generated_hypotheses)
print(f'generated_hypotheses----->')

if len(generated_hypotheses.keys())==20:
    generated_hypotheses = json_to_text(generated_hypotheses)
    chat_history.append(generated_hypotheses)
    critic_prompt = construct_critic_prompt(goal_statement, constraint_list, chat_history[-1])
    feedback_from_critic_1 = critic_1(expert_list,critic_prompt,initial_feedback,refined_feedback_prompt)
    feedback_from_critic_1 = json.loads(feedback_from_critic_1)
    feedback_from_critic_1,final_answer,suggestions_with_no = process_feedback_extract_final_answer(feedback_from_critic_1)
    feedback_history.append(feedback_from_critic_1)
    print('suggestions_with_no----->',suggestions_with_no)
    print(f'final_answer-----> {final_answer}')
    attempts = 0       
    while final_answer!="Yes" and attempts<5:      
        feedback_prompt = construct_feedback_prompt(feedback_from_critic_1)
        print('==============>Constructing feedback prompt for refined hypotheses')
        refined_hypotheses = hypothesis_generator(expert_list, prompt, feedback_prompt, chat_history[-1])
        refined_hypotheses = json.loads(refined_hypotheses)
        print('===================>refined hypothesis generated')
        chat_history.append(json_to_text(refined_hypotheses))
        feedback_prompt_for_refined_hypothesis = construct_feedback_prompt_for_refined_hypotheses(feedback_history[-1], chat_history[-1])
        print('===================>Constructing feedback prompt for refined hypotheses')
        refined_feedback = critic_1(expert_list,critic_prompt,feedback_history[-1],feedback_prompt_for_refined_hypothesis)
        refined_feedback = json.loads(refined_feedback)
        print('===================>feedback received for refined hypotheses')
        feedback_from_critic_1,final_answer,suggestions_with_no = process_feedback_extract_final_answer(refined_feedback)
        feedback_history.append(feedback_from_critic_1)
        suggestions_no_list.append(suggestions_with_no)
        print(f'suggestions_with_no-----> {suggestions_with_no}')
        attempts += 1 
        print(f'attempts-----> {attempts}')
    if final_answer=="Yes":
        print("Suggestions are generated properly")
else:
    print("Suggestions are not generated properly. Please try again")

average_suggestions_with_no = sum(suggestions_no_list) / len(suggestions_no_list)
print(f'average_suggestions_with_no-----> {average_suggestions_with_no}')
# put the chat history and feedback history in a json records format
with open('/data/data/shri/Materials_Discovery/Rapid_Self_Healing_Hydrogel_chat_history_5.json', 'w') as f:
    json.dump(chat_history, f)
with open('/data/data/shri/Materials_Discovery/Rapid_Self_Healing_Hydrogel_feedback_history_5.json', 'w') as f:
    json.dump(feedback_history, f)