from openai import OpenAI
import unicodedata
import os

class GroqCaller:
    """
    A wrapper for the Groq API to generate content.
    """
    def __init__(self, api_key=None, model='openai/gpt-oss-120b'):
        """
        Initializes the GroqCaller.

        Args:
            api_key (str): Your Groq API key. If None, uses GROQ_API_KEY from environment.
            model (str): The model to use for generation.
        """
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    def call_groq(self, prompt):
        """
        Calls the Groq API with a given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated text from the model.
        """
        try:
            response = self.client.responses.create(
                input=prompt,
                model=self.model,
            )
            text = (unicodedata.normalize("NFKC", response.output_text).replace('\u2011', '-').encode('ascii', errors='ignore').decode())

            return text
        except Exception as e:
            print(f"An error occurred with the Groq API: {e}")
            return "" # Return empty string on error
        
if __name__ == '__main__':
    g = GroqCaller()
    print(g.call_groq("What is the height of Mt. Everest"))