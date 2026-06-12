from google import genai

class GeminiCaller:
    """
    A wrapper for the Gemini API to generate content.
    """
    def __init__(self, api_key, model='gemini-2.5-flash'):
        """
        Initializes the GeminiCaller.

        Args:
            api_key (str): Your Google AI API key.
            model (str): The model to use for generation.
        """

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def call_gemini(self, prompt):
        """
        Calls the Gemini API with a given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated text from the model.
        """
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            return response.text
        except Exception as e:
            print(f"An error occurred with the Gemini API: {e}")
            return "" # Return empty string on error
