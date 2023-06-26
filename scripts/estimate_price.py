from enum import Enum, auto
from flags import Templates, ModelChoices

""" approximate number of tokens including both the prompt and the completion """
AVG_TOKENS = {
    Templates.ASDOT: 160,
    Templates.JSON: 300
}

""" pricings ($ per 1K tokens) as stated in https://openai.com/pricing"""
MODEL_PRICINGS = {
    ModelChoices.G3P5: 0.0200,  # text-davinci-003
    ModelChoices.G3P5T: 0.0020,  # gpt-3.5-turbo
    ModelChoices.G4: 0.0300,  # w 8K context
    ModelChoices.G3: 0.2  # davinci (G3P5)
}

if __name__ == "__main__":
    num_examples = 226
    template_version = Templates.ASDOT
    model_choice = ModelChoices.G3

    n_tokens = AVG_TOKENS[template_version]
    price_per_1k = MODEL_PRICINGS[model_choice]

    print(f"{template_version.name}, {model_choice.name}, {num_examples * n_tokens / 1000. * price_per_1k}")
