
import os
import logging
from dotenv import load_dotenv
import openai
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from groq import Groq

load_dotenv()

class GroqAdapter:
    def __init__(
        self,
        api_key=None,
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=800
    ):

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def generate(self, prompt):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self.chat(messages)

    def chat(self, messages):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content


class GroqDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def get_model_name(self):
        return "llama-3.1-8b-instant"

    def generate(self, prompt: str):

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content