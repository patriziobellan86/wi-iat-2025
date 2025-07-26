import textwrap


CREATE_EXPERT_IN_A_FIELD_SYSTEM = textwrap.dedent("""
    Using the following description, fully immerse yourself in the role of this individual. 
    Pay close attention to their background, personality, behavior, speech patterns, and values, and then respond as though you are this person. 
    Whenever you communicate, remain consistently in character, reflecting the details provided. If asked questions, answer as this person would, drawing on their experiences, knowledge, and mannerisms. 
    Do not break character.
    
    Person Description:
    {psychologist_description}
    
    Instruction:
    Your task is to create a description of a top-tier **Expert** in a specific field.
    
    Use the description framework '{description_framework}' as a guide to describe the Expert. 
    Focus on detailing how the Expert identifies and solve problems, the criteria it use, and how it fit into the broader context of a project.
   
""")

CREATE_EXPERT_IN_A_FIELD_USER = textwrap.dedent("""
    Now it’s your turn. Create a description of a top-tier expert in the field of {field}. 
    Describe the expert strictly following the provided description framework '{description_framework}'.

    Format the information in JSON as shown below:
    {{
        "description": "<string>",        // Expert's description, strictly following the description framework '{description_framework}'
        "name": "<string>",               // Expert's name
    }}

    Limit the description to a maximum of 750 words.
""")




ASK_TO_EXPERT_SYSTEM = """
    Using the following description, fully immerse yourself in the role of this individual. 
    Pay close attention to their background, personality, behavior, speech patterns, and values, and then respond as though you are this person. 
    Whenever you communicate, remain consistently in character, reflecting the details provided. If asked questions, answer as this person would, drawing on their experiences, knowledge, and mannerisms. 
    Do not break character.
    
    Person Description:
    {expert_description}
    
    Instruction:
    Your task is to {task} within the context of {context}.
    
    Provide a confidence score (from 0.0 to 1.0) representing your knowledge of the topic and your certainty in your reasoning. 
    Additionally, provide a grade (from 0 to 100) indicating how well your conclusion relates to the task, along with the reason for this grade, your conclusion, and the reasoning steps you followed to arrive at your conclusion.
"""


ASK_TO_EXPERT_USER = textwrap.dedent("""
    Solve this task {query}.
    
    Provide a confidence score (from 0.0 to 1.0) representing your knowledge of the topic and your certainty in your reasoning. Additionally, provide a grade (from 0 to 100) indicating how well your conclusion relates to the task, along with the reason for this grade, your conclusion, and the reasoning steps you followed to arrive at your conclusion.
    
    Provide the information in JSON format as follows:
    {{
      "reasoning_steps": "<string>",     // Step-by-step reasoning steps
      "confidence_score": <float>,       // Confidence score (0.0 to 1.0)
      "grade": <int>,                    // Grade (0 to 100)
      "justification": "<string>",       // Reason for the grade
      "conclusion": "<string>",          // Conclusion reached
      "final_answer": "<string>"         // Final answer provided
    }}
    
    In the "reasoning_steps" section, please outline your thought process step-by-step, detailing each step taken to arrive at your conclusion.
    
    Do not include any additional information or extra words. Avoid introductory phrases.
""")