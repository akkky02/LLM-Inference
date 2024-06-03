#!/usr/bin/env python

"""
LLM Inference with Weave

This script demonstrates how to perform sequence classification using a large language model (LLM) with the Weave framework.
The script allows you to pass the dataset name, prompt, model name, and the number of test samples as command-line arguments.
"""

import os
import argparse
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import weave
from sklearn.metrics import accuracy_score
from vllm import LLM, SamplingParams

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Initialize Weave
weave.init('seq-clf-vllm-inference')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run LLM inference with Weave')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--prompt', type=str, required=True, help='Prompt template')
parser.add_argument('--name', type=str, required=True, help='Name of the model')
parser.add_argument('--test_samples', type=int, default=None, help='Number of test samples to use')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
args = parser.parse_args()

# Update variables with passed arguments
dataset_name = args.dataset
prompt = args.prompt
model_name = args.name
num_test_samples = args.test_samples if args.test_samples is not None else None
batch_size = args.batch_size

# Load the dataset
dataset = load_dataset(dataset_name)

# Initialize the LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    trust_remote_code=True,
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    enable_prefix_caching=True
)

# Define the sequence classification model
class SequenceClassificationModel(weave.Model):
    model_name: str
    prompt_template: str
    llm: LLM

    @weave.op()
    def predict(self, texts: list[str]) -> list[int]:
        sampling_params = SamplingParams(temperature=0, max_tokens=1, truncate_prompt_tokens=8192)
        prompt_texts = [self.prompt_template.format(text=text) for text in texts]
        outputs = self.llm.generate(prompt_texts, sampling_params)
        predicted_labels = []
        for output in outputs:
            try:
                predicted_labels.append(int(output.outputs[0].text))
            except ValueError:
                predicted_labels.append(-1)
        return predicted_labels

# Create the sequence classification model
model = SequenceClassificationModel(
    name=model_name,
    model_name='meta-llama/Meta-Llama-3-70B-Instruct',
    prompt_template=prompt,
    llm=llm
)

# Prepare the test data
test_inputs = dataset['test']
if num_test_samples is None:
    num_test_samples = len(test_inputs['text'])
test_examples = [
    {'id': str(i), 'text': text, 'label': label}
    for i, (text, label) in enumerate(zip(test_inputs['text'][:num_test_samples], test_inputs['label'][:num_test_samples]))
]

# Evaluate the model in batches
@weave.op()
def evaluate_model(model: SequenceClassificationModel, test_examples: list) -> dict:
    total_accuracy = 0.0
    num_batches = (len(test_examples) + batch_size - 1) // batch_size

    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        texts = [ex['text'] for ex in batch]
        y_true = [ex['label'] for ex in batch]
        y_pred = model.predict(texts)
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
        if valid_indices:
            accuracy = accuracy_score([y_true[i] for i in valid_indices], [y_pred[i] for i in valid_indices])
            total_accuracy += accuracy * len(valid_indices)

    return {'accuracy': total_accuracy / len(test_examples)}

results = evaluate_model(model, test_examples)
print("Prompt:\n", prompt)
print("Accuracy:", results['accuracy'])