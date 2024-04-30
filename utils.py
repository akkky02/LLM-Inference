from datasets import load_dataset, DatasetDict
from vllm import SamplingParams


def zero_shot_classification(dataset_name, prefix):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Iterate over the dataset splits (train, test, validation)
    modified_dataset_dict = {}
    for split in ["train", "test", "validation"]:
        # Get the texts and labels from the current split
        texts = dataset[split]["text"]
        labels = dataset[split]["label"]

        # Generate the prompts for each Text
        generating_prompts = [prefix + "Text: " + text + "\nResponse: " for text in texts]

        # Set the sampling parameters
        sampling_params = SamplingParams(temperature=0, max_tokens=1)

        # Generate the sentiment labels for each text
        outputs = llm.generate(generating_prompts, sampling_params)
        predicted_label = []
        for output in outputs:
            try:
                predicted_label.append(int(output.outputs[0].text))
            except ValueError:
                predicted_label.append(-1)

        # Add the predicted labels to the dataset
        modified_dataset = dataset[split].add_column("predicted_label", predicted_label)
        modified_dataset_dict[split] = modified_dataset

    # Create a DatasetDict with the modified datasets
    return DatasetDict(modified_dataset_dict)