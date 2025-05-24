import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
import sys
from pydantic import BaseModel
import random
import os
import json
import re

allowable_models = ["meditron:latest", "jyan1/paligemma-mix-224:latest", "qwen2.5:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b",
          "deepseek-r1:7b", "deepseek-r1:70b", "qordmlwls/llama3.1-medical:latest", "mixtral:latest","llava:latest","llava:34b","llava:13b","llama3.2-vision:latest"]


# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

import json

def list_png_files(directory):
    """Returns a list of PNG files in the given directory."""
    return [file for file in os.listdir(directory) if file.lower().endswith('.png')]

def fix_json(json_input):
    """
    Ensures the input is a JSON string or a dictionary and always returns a dictionary.
    If input is a dictionary, return it as-is.
    If input is a valid JSON string, return parsed JSON as a dictionary.
    If input is an invalid JSON string, attempts to fix it by trimming trailing characters.
    """
    # If input is already a dictionary, return it directly
    if isinstance(json_input, dict):
        return json_input  

    # Ensure input is a string (or bytes), otherwise return error JSON
    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"Image": "NA", "Breast Composition": "NA", "Findings": "NA", "BIRADS": "NA"}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"Image": "NA", "Breast Composition": "NA", "Findings": "NA", "BIRADS": "NA"}
    except json.JSONDecodeError:
        pass  # If invalid, proceed with fixing

    # Try trimming trailing characters
    for i in range(len(json_input), 0, -1):  
        try:
            parsed_json = json.loads(json_input[:i])  # Try parsing progressively shorter substrings
            if isinstance(parsed_json, dict):
                return parsed_json  # Ensure it's a dictionary
        except Exception as e:
            print(f"Unexpected error: {e}")  # Catch other unforeseen errors
            continue  # Keep trimming

    # If all attempts fail, return error JSON
    return {"Image": "NA", "Breast Composition": "NA", "Findings": "NA", "BIRADS": "NA"}
