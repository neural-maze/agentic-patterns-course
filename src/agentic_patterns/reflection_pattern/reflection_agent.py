from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import FixedFirstChatHistory
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()


BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""


class ReflectionAgent:
    """
    A class that implements a Reflection Agent, which generates responses and reflects
    on them using the LLM to iteratively improve the interaction. The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.

    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (Groq): An instance of the Groq client to interact with the language model.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model = model

    def _request_completion(
        self,
        history: list,
        verbose: int = 0,
        log_title: str = "COMPLETION",
        log_color: str = "",
    ):
        """
        A private method to request a completion from the Groq model.

        Args:
            history (list): A list of messages forming the conversation or reflection history.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).

        Returns:
            str: The model-generated response.
        """
        output = completions_create(self.client, history, self.model)

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        """
        Generates a response based on the provided generation history using the model.

        Args:
            generation_history (list): A list of messages forming the conversation or generation history.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: The generated response.
        """
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        """
        Reflects on the generation history by generating a critique or feedback.

        Args:
            reflection_history (list): A list of messages forming the reflection history, typically based on
                                       the previous generation or interaction.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: The critique or reflection response from the model.
        """
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 10,
        verbose: int = 0,
    ) -> str:
        """
        Runs the ReflectionAgent over multiple steps, alternating between generating a response
        and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message or query that initiates the interaction.
            generation_system_prompt (str, optional): The system prompt for guiding the generation process.
            reflection_system_prompt (str, optional): The system prompt for guiding the reflection process.
            n_steps (int, optional): The number of generate-reflect cycles to perform. Defaults to 3.
            verbose (int, optional): The verbosity level controlling printed output. Defaults to 0.

        Returns:
            str: The final generated response after all cycles are completed.
        """
        print(f"Starting ReflectionAgent.run() with user_msg: {{user_msg}}")
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        # Given the iterative nature of the Reflection Pattern, we might exhaust the LLM context (or
        # make it really slow). That's the reason I'm limitting the chat history to three messages.
        # The `FixedFirstChatHistory` is a very simple class, that creates a Queue that always keeps
        # fixeed the first message. I thought this would be useful for maintaining the system prompt
        # in the chat history.
        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,
        )

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

            # Generate the response
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            # Reflect and critique the generation
            critique = self.reflect(reflection_history, verbose=verbose)

            if "<OK>" in critique:
                # If no additional suggestions are made, stop the loop
                print(
                    Fore.RED,
                    "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                )
                break

            update_chat_history(generation_history, critique, "user")
            update_chat_history(reflection_history, critique, "assistant")

        return generation
