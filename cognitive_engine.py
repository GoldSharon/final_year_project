import re
import json
from typing import Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


class CognitiveEngine:
    """
    CognitiveEngine manages interaction with LLMs (e.g., DeepSeek Coder)
    for automated data preprocessing, feature engineering, and reasoning.
    """

    def __init__(self, 
                 model_name: str = 'dolphin-llama3:8b', 
                 temperature: float = 0.2):
        """
        Initialize the CognitiveEngine with LLM configuration.

        Args:
            model_name: Ollama model name
            temperature: LLM temperature (lower = more deterministic)
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=16_000,
            format="json"
        )
    
    def chat_llm(self, system_instruction: str, query: str) -> Dict:
        """
        Send query to LLM with system instruction and return parsed JSON response.

        Args:
            system_instruction: A guiding system message for the LLM
            query: The actual user input query

        Returns:
            Dict: Parsed JSON dictionary response from the LLM
        """
        system_msg = SystemMessage(content=system_instruction)
        user_msg = HumanMessage(content=query)

        # print("📥 Input:", query)

        # Call the LLM
        ai_msg = self.llm.invoke([system_msg, user_msg])
        response = ai_msg.content

        print("📤 Raw Output:", response)

        # ✅ Remove unwanted wrappers (e.g., <think>...</think>)
        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # ✅ Parse JSON safely
        try:
            parsed = json.loads(clean_response)
            print("✅ Parsed JSON successfully")
            return parsed
        except json.JSONDecodeError as e:
            print(f"⚠ JSON Parse Error: {e}")
            print(f"⚠ Raw response snippet: {clean_response[:500]}")
            return {}  # fallback empty dict
