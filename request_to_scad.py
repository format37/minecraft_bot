from pydantic import BaseModel
import os
import openai
from openai import OpenAI
import base64
import logging
import json

logging.basicConfig(level=logging.INFO)

class ScadResponse(BaseModel):
    task: str
    scad_script: str
    next_step_plan: str
    need_to_continue: bool
    completion_percentage: int
    design_description: str

def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
os.environ["OPENAI_API_KEY"] = load_config('keys.json')['OPENAI_API_KEY']

def generate_scad_script(request = "Please, build a village house, using OpenScad syntax.", base64_image = None, steps = 5):
    client = OpenAI()

    # system_content = f"You are a professional CAD designer and Minecraft architect. Your task is building 3D objects for Minecraft using Scad syntax. Your script will be rendered in plotly as voxels before moving to Minecraft. You need to account the Target size that corresponds to the Maximum block side to choose the design detalization level. Don't forget to provide inner space for players if required. You should use your knowledge and experience in OpenScad syntax to create the best possible result. You have to build the object step by step. You have maximum {steps} steps to finish your work. You need to provide complete scad script containing all functions after each generation."
    # Read system_content from the system_prompt.txt file
    with open('builder_system_prompt.txt', 'r') as file:
        system_content = file.read()
    system_content = system_content.replace('{steps}', str(steps))
    system_message = {"role": "system", "content": system_content}

    if base64_image is None:
        user_message = {"role": "user", "content": request}
    else:
        # print("Using image as a context")
        logging.info("Using image as a context")
        user_message = {"role": "user", "content": [
                    {"type": "text", "text": request},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}}
                ]}
    completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                system_message,
                user_message,
            ],
            response_format=ScadResponse,
            temperature=1.0,
            max_tokens=16384,
        )

    message = completion.choices[0].message
    
    if message.parsed:
        # print("task: ", message.parsed.task)
        # print("next_step_plan: ", message.parsed.next_step_plan)
        # print("scad_script: ", message.parsed.scad_script)
        # print("need_to_continue: ", message.parsed.need_to_continue)
        # print("completion_percentage: ", message.parsed.completion_percentage)
        # print("design_description: ", message.parsed.design_description)
        logging.info(f"task: {message.parsed.task}")
        logging.info(f"next_step_plan: {message.parsed.next_step_plan}")
        logging.info(f"scad_script: {message.parsed.scad_script}")
        logging.info(f"need_to_continue: {message.parsed.need_to_continue}")
        logging.info(f"completion_percentage: {message.parsed.completion_percentage}")
        logging.info(f"design_description: {message.parsed.design_description}")

        return message.parsed.task, \
            message.parsed.scad_script, \
            message.parsed.next_step_plan, \
            message.parsed.need_to_continue, \
            message.parsed.design_description
    else:
        # print(message.refusal)
        # result = message.refusal
        # print(f"Unable to parse the LLM response: {message}")
        logging.info(f"Unable to parse the LLM response: {message}")
        return '{task: "'+request+'", step: 0, scad_script:"", next_step_plan: ""}'
    # return result

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    user_request = "Village house"
    step = 0
    request = '{task: "'+user_request+'", step: '+str(step)+', scad_script:"", next_step_plan: ""}'
    task, scad, plan, need_to_continue, design_description = generate_scad_script(request)
    base64_image = encode_image('combined.png')
    step += 1
    request = '{task: "'+task+'", step: '+str(step)+', scad_script:"'+scad+'", next_step_plan: "'+plan+'"}'
    task, scad, plan, need_to_continue, design_description = generate_scad_script(request, base64_image)
    

if __name__ == "__main__":
    main()
