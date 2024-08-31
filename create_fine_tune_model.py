import argparse
from openai import OpenAI
import os
import json
import time
from datetime import datetime, timedelta

client = OpenAI()


# upload the data
def upload_data(file_path):
    dataset_file = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    return dataset_file.id


# create the fine-tune model
def create_fine_tune_model(dataset_file_id, model, suffix, epochs=None):
    if epochs:
        fine_tune_model = client.fine_tuning.jobs.create(
            training_file=dataset_file_id, model=model, suffix=suffix,
            hyperparameters={
                "n_epochs": epochs
            }
        )
    else:
        fine_tune_model = client.fine_tuning.jobs.create(
            training_file=dataset_file_id, model=model, suffix=suffix
        )
    return fine_tune_model


# List 10 fine-tuning jobs
# client.fine_tuning.jobs.list(limit=10)


# Retrieve the state of a fine-tune
def get_fine_tune_model(fine_tune_model_id):
    fine_tune_model = client.fine_tuning.jobs.retrieve(fine_tune_model_id)
    return fine_tune_model


def get_fine_tune_model_events(fine_tune_model_id):
    fine_tune_model_events = client.fine_tuning.jobs.retrieve(fine_tune_model_id)
    return fine_tune_model_events


# List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")


def save_config(config, config_file_path):
    with open(config_file_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4, ensure_ascii=False)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a fine-tuned model using OpenAI API"
    )
    parser.add_argument(
        "config", help="Name of the config file (without .json extension)"
    )
    args = parser.parse_args()

    # Use the provided config name
    config_name = args.config
    config_file_path = f"./{config_name}.json"

    # Check if the config file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"The config file '{config_file_path}' does not exist.")

    with open(config_file_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    # Check if the file exists
    dataset_file_path = f"./config/{config_name}/{config_name}_dataset.jsonl"
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(
            f"The dataset file '{dataset_file_path}' does not exist."
        )

    # Extract necessary information from the config
    # dataset_file_path = config.get("dataset", "./config/prompt_test5_dataset.jsonl")
    base_model = config.get("base_model", "")
    if not base_model:
        raise ValueError("base_model is not set in the config file.")
    suffix = config.get("suffix", "")
    if not suffix:
        raise ValueError("suffix is not set in the config file.")

    if config.get("dataset_file_id") or config.get("fine_job_id"):
        dataset_file_id = config.get("dataset_file_id")
    else:
        dataset_file_id = upload_data(dataset_file_path)
        config["dataset_file_id"] = dataset_file_id

    if config.get("fine_job_id"):
        fine_tune_model_id = config.get("fine_job_id")
    else:
        fine_tune_model = create_fine_tune_model(dataset_file_id, base_model, suffix, config.get("epochs", None))
        fine_tune_model_id = fine_tune_model.id
        config["fine_job_id"] = fine_tune_model_id

    fine_tune_model_events = get_fine_tune_model_events(fine_tune_model_id)
    print(fine_tune_model_events.status)

    save_config(config, config_file_path)

    while fine_tune_model_events.status == "validating_files":
        print("Fine-tuning job is still validating files...")
        time.sleep(10)
        fine_tune_model_events = get_fine_tune_model_events(fine_tune_model_id)

    config["ft_created_at"] = datetime.fromtimestamp(
        fine_tune_model_events.created_at
    ).strftime("%Y-%m-%d %H:%M:%S")
    if fine_tune_model_events.estimated_finish:
        config["ft_estimated_finish"] = datetime.fromtimestamp(
            fine_tune_model_events.estimated_finish
        ).strftime("%Y-%m-%d %H:%M:%S")

    save_config(config, config_file_path)

    while fine_tune_model_events.status == "running":
        current_time = datetime.now()
        if fine_tune_model_events.estimated_finish:
            estimated_finish = datetime.fromtimestamp(
                fine_tune_model_events.estimated_finish
            )
            time_difference = estimated_finish - current_time
        else:
            time_difference = timedelta(minutes=5)

        if time_difference > timedelta(minutes=5):
            print(
                f"Estimated finish time is more than 5 minutes away ({estimated_finish}). Exiting program."
            )
            return

        print("Fine-tuning job is still running...")
        time.sleep(10)  # Wait for 10 seconds before checking again
        fine_tune_model_events = get_fine_tune_model_events(fine_tune_model_id)

    print("Fine-tuning job has completed.")
    if fine_tune_model_events.status == "succeeded":
        config["ft_model"] = fine_tune_model_events.fine_tuned_model
        config["used_tokens"] = fine_tune_model_events.trained_tokens
        print(fine_tune_model_events.fine_tuned_model)
        print(fine_tune_model_events.trained_tokens)

    save_config(config, config_file_path)


if __name__ == "__main__":
    main()
