from enum import Enum, auto
from flags import Templates, ConsistencyTemplateNames, ModelChoices, DatasetChoice

""" approximate number of tokens including both the prompt and the completion """
AVG_TOKENS = {
    Templates.ASDOT: 160,
    Templates.JSON: 300,
    ConsistencyTemplateNames.V4: 320,
}

""" pricings ($ per 1K tokens) as stated in https://openai.com/pricing"""
MODEL_PRICINGS = {
    ModelChoices.G3P5: 0.0200,  # text-davinci-003
    ModelChoices.G3P5T: 0.0020,  # gpt-3.5-turbo
    ModelChoices.G4: 0.0300,  # w 8K context
    ModelChoices.G3: 0.02,  # davinci (G3P5)
    ModelChoices.NONE: 0
}

NUM_EXAMPLES = {
    DatasetChoice.REL2TEXT: 226,
    DatasetChoice.WEBNLG: 354,
    DatasetChoice.DART: 1439,
}

"""
Assumptions
 - retry shots use the same model as the zeroth shot
 - If cv_model is not None, we assume all completions from N-shot Gen. are under the PARENT F1 threshold
"""

if __name__ == "__main__":
    # Dataset
    dataset_choice = DatasetChoice.REL2TEXT

    # N-Shot Generator
    template_version = Templates.JSON
    model_choice = ModelChoices.G3
    retry_shots = 5

    # Consistency Validation
    cv_model = ModelChoices.NONE
    cv_template = ConsistencyTemplateNames.V4

    # Calculate for one example (=one input sample)
    n_examples = NUM_EXAMPLES[dataset_choice]
    _one_example_cost_n_shot_gen = AVG_TOKENS[template_version] * MODEL_PRICINGS[model_choice] / 1000.
    print(f"One example N-shot Generator: {_one_example_cost_n_shot_gen:.4f}")
    _one_example_cost_cv = AVG_TOKENS[cv_template] * MODEL_PRICINGS[cv_model] / 1000.
    print(f"One example CV: {_one_example_cost_cv:.4f}")

    # calculate total for all samples and retry shots
    total_cost_n_shot = (retry_shots + 1) * n_examples * _one_example_cost_n_shot_gen
    total_cost_cv = n_examples * _one_example_cost_cv
    total_cost = total_cost_n_shot + total_cost_cv

    print(f"{dataset_choice.name} {template_version.name}_{retry_shots}x{model_choice.name} ({cv_model.name}_{cv_template.name})\n\tNSHOT: {total_cost_n_shot:.4f}\n\tCV: {total_cost_cv:.4f}\n\tTOTAL: {total_cost:.4f}")
