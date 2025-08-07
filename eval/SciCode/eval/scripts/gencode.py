import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
from tqdm import tqdm

from scicode.gen.models import extract_python_script, get_model_function
from scicode.parse.parse import (
    extract_function_name,
    get_function_from_code,
    read_from_hf_dataset,
)
from scicode.utils.min_entropy import (
    load_hf_model_for_min_entropy_generation,
    min_entropy_inference,
)

DEFAULT_PROMPT_TEMPLATE = Path(
    "eval", "data", "background_comment_template.txt"
).read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("eval", "data", "multistep_template.txt").read_text()


def extract_model_name(model_path):
    import re

    match = re.search(r"ckpts_verl/(.+?)/actor/huggingface", model_path)
    if match:
        extracted = match.group(1)
        # Replace slashes with hyphens
        return extracted.replace("/", "-")
    return model_path


class Gencode:
    def __init__(
        self,
        model: str,
        output_dir: Path,
        prompt_dir: Path,
        with_background: bool,
        temperature: float,
        hyperparameters: dict = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.temperature = temperature
        self.previous_llm_code = []
        self.hyperparameters = hyperparameters

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def save_prompt_with_steps(
        self,
        prob_data: dict,
        prompt: str,
        num_steps: int,
        mode: str = "normal",
        iteration=0,
    ) -> None:
        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]

        if mode == "min_entropy":
            extracted_model_name = (
                extracted_model_name
                + "_min_entropy"
                + f"_kl_{self.hyperparameters['kl_weight']}_lr_{self.hyperparameters['learning_rate']}_steps_{self.hyperparameters['n_grad_steps']}_threshold_{self.hyperparameters['threshold']}"
            )
            output_dir = Path(
                self.prompt_dir, extracted_model_name, self._get_background_dir()
            )
        elif mode == "self_refinement":
            extracted_model_name = extracted_model_name + "_self_refinement"
            output_dir = Path(
                self.prompt_dir,
                extracted_model_name,
                self._get_background_dir(),
                f"iter_{iteration}",
            )
        elif mode == "temp_decrease":
            extracted_model_name = (
                extracted_model_name
                + "_temp_decrease"
                + f"_ratio_{self.hyperparameters['target_ratio']}_threshold_{self.hyperparameters['target_threshold']}"
            )
            output_dir = Path(
                self.prompt_dir, extracted_model_name, self._get_background_dir()
            )
        else:
            output_dir = Path(
                self.prompt_dir, extracted_model_name, self._get_background_dir()
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def save_response_with_steps(
        self,
        prob_data: dict,
        response: str,
        previous_code: str,
        num_steps: int,
        mode: str = "normal",
        iteration=0,
    ) -> None:
        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]

        if mode == "min_entropy":
            extracted_model_name = (
                extracted_model_name
                + "_min_entropy"
                + f"_kl_{self.hyperparameters['kl_weight']}_lr_{self.hyperparameters['learning_rate']}_steps_{self.hyperparameters['n_grad_steps']}_threshold_{self.hyperparameters['threshold']}"
            )
            output_dir = Path(
                self.output_dir, extracted_model_name, self._get_background_dir()
            )
        elif mode == "self_refinement":
            extracted_model_name = extracted_model_name + "_self_refinement"
            output_dir = Path(
                self.output_dir,
                extracted_model_name,
                self._get_background_dir(),
                f"iter_{iteration}",
            )
        elif mode == "temp_decrease":
            extracted_model_name = (
                extracted_model_name
                + "_temp_decrease"
                + f"_ratio_{self.hyperparameters['target_ratio']}_threshold_{self.hyperparameters['target_threshold']}"
            )
            output_dir = Path(
                self.output_dir, extracted_model_name, self._get_background_dir()
            )
        else:
            output_dir = Path(
                self.output_dir, extracted_model_name, self._get_background_dir()
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f"{previous_code}\n{python_code}", encoding="utf-8")

    def generate_min_ent_response_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        model,
        tokenizer,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        *,
        save: bool = True,
    ) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]

        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]
            extracted_model_name = (
                extracted_model_name
                + "_min_entropy"
                + f"_kl_{self.hyperparameters['kl_weight']}_lr_{self.hyperparameters['learning_rate']}_steps_{self.hyperparameters['n_grad_steps']}_threshold_{self.hyperparameters['threshold']}"
            )

        output_file_path = (
            self.output_dir
            / extracted_model_name
            / self._get_background_dir()
            / f"{prob_id}.{num_steps}.py"
        )
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5)
                        or (prob_id == "62" and prev_step == 0)
                        or (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = Path(
                            "eval", "data", f"{prob_id}.{prev_step + 1}.txt"
                        )
                    else:
                        prev_file_path = (
                            self.output_dir
                            / extracted_model_name
                            / self._get_background_dir()
                            / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding="utf-8")
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(
                            f"Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}."
                        )

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data, num_steps, prompt_template
        )
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps, "min_entropy")

        model_kwargs = {}
        if "claude" in self.model:
            model_kwargs["max_tokens"] = 4096
        model_kwargs["temperature"] = self.temperature

        # write the response to a file if it doesn't exist
        # hyperparameters = {
        #     "threshold": 0.7,
        #     "kl_weight": 0.001,
        #     "learning_rate": 0.1,
        #     "n_grad_steps": 5,
        # }
        response_from_llm = min_entropy_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            hyperparameters=self.hyperparameters,
            max_new_tokens=4096,
        )

        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(
            prob_data, response_from_llm, previous_code, num_steps, "min_entropy"
        )

    def generate_self_refinement_response_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        model="gpt-4o",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        iteration=0,
        *,
        save: bool = True,
    ) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]

        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]
            extracted_model_name = extracted_model_name + "_self_refinement"

        output_file_path = (
            self.output_dir
            / extracted_model_name
            / self._get_background_dir()
            / f"iter_{iteration}"
            / f"{prob_id}.{num_steps}.py"
        )
        previous_iter_code_path = (
            self.output_dir
            / extracted_model_name
            / self._get_background_dir()
            / f"iter_{iteration - 1}"
            / f"{prob_id}.{num_steps}.py"
        )
        previous_iter_prompt_path = (
            self.prompt_dir
            / extracted_model_name
            / self._get_background_dir()
            / f"iter_{iteration - 1}"
            / f"{prob_id}.{num_steps}.txt"
        )

        # load previous iteration's code and prompt
        if previous_iter_code_path.exists():
            with open(previous_iter_code_path, "r", encoding="utf-8") as file:
                previous_code_string = file.read()
        else:
            previous_code_string = ""
            print(f"in iteration 0, no previous code")

        if previous_iter_prompt_path.exists():
            with open(previous_iter_prompt_path, "r", encoding="utf-8") as file:
                previous_prompt_string = file.read()
        else:
            previous_prompt_string = ""
            print(f"in iteration 0, no previous prompt")

        if iteration == 0:
            current_prompt_prefix = ""
        else:
            current_prompt_prefix = (
                previous_prompt_string + "\n\n" + previous_code_string + "\n\n"
            )

        print(
            f"previous iteration's code path: {previous_iter_code_path} | previous iteration's prompt path: {previous_iter_prompt_path} | current function saving path: {output_file_path}"
        )

        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5)
                        or (prob_id == "62" and prev_step == 0)
                        or (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = Path(
                            "eval", "data", f"{prob_id}.{prev_step + 1}.txt"
                        )
                    else:
                        prev_file_path = (
                            self.output_dir
                            / extracted_model_name
                            / self._get_background_dir()
                            / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding="utf-8")
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(
                            f"Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}."
                        )

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data, num_steps, prompt_template
        )
        prompt = current_prompt_prefix + prompt
        if save:
            self.save_prompt_with_steps(
                prob_data=prob_data,
                prompt=prompt,
                num_steps=num_steps,
                mode="self_refinement",
                iteration=iteration,
            )

        model_kwargs = {}
        if "claude" in model:
            model_kwargs["max_tokens"] = 4096
        model_kwargs["temperature"] = self.temperature
        # write the response to a file if it doesn't exist
        model_fct = get_model_function(model, **model_kwargs)
        response_from_llm = model_fct(prompt)

        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(
            prob_data=prob_data,
            response=response_from_llm,
            previous_code=previous_code,
            num_steps=num_steps,
            mode="self_refinement",
            iteration=iteration,
        )

    def generate_temp_decrease_response_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        model="gpt-4o",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        *,
        save: bool = True,
    ) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]

        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]

        extracted_model_name = (
            extracted_model_name
            + "_temp_decrease"
            + f"_ratio_{self.hyperparameters['target_ratio']}_threshold_{self.hyperparameters['target_threshold']}"
        )

        output_file_path = Path(
            self.output_dir,
            extracted_model_name,
            self._get_background_dir(),
            f"{prob_id}.{num_steps}.py",
        )
        # output_file_path = (
        #     self.output_dir
        #     / extracted_model_name
        #     / self._get_background_dir()
        #     / f"{prob_id}.{num_steps}.py"
        # )
        # print(f"current function saving path: {output_file_path}")
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5)
                        or (prob_id == "62" and prev_step == 0)
                        or (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = Path(
                            "eval", "data", f"{prob_id}.{prev_step + 1}.txt"
                        )
                    else:
                        prev_file_path = (
                            self.output_dir
                            / extracted_model_name
                            / self._get_background_dir()
                            / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding="utf-8")
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(
                            f"Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}."
                        )

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data, num_steps, prompt_template
        )
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps, "temp_decrease")

        model_kwargs = {}
        if "claude" in model:
            model_kwargs["max_tokens"] = 4096
        model_kwargs["temperature"] = self.temperature
        model_kwargs["is_temp_decrease"] = True
        model_kwargs["hyperparameters"] = self.hyperparameters
        # write the response to a file if it doesn't exist
        model_fct = get_model_function(model, **model_kwargs)
        response_from_llm = model_fct(prompt)

        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(
            prob_data, response_from_llm, previous_code, num_steps, "temp_decrease"
        )

    def generate_response_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        model="gpt-4o",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        *,
        save: bool = True,
    ) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]

        if "ckpts_verl" in self.model:
            extracted_model_name = extract_model_name(self.model)
        else:
            extracted_model_name = Path(self.model).parts[-1]
        output_file_path = (
            self.output_dir
            / extracted_model_name
            / self._get_background_dir()
            / f"{prob_id}.{num_steps}.py"
        )
        # print(f"current function saving path: {output_file_path}")
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5)
                        or (prob_id == "62" and prev_step == 0)
                        or (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = Path(
                            "eval", "data", f"{prob_id}.{prev_step + 1}.txt"
                        )
                    else:
                        prev_file_path = (
                            self.output_dir
                            / extracted_model_name
                            / self._get_background_dir()
                            / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding="utf-8")
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(
                            f"Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}."
                        )

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data, num_steps, prompt_template
        )
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps)

        model_kwargs = {}
        if "claude" in model:
            model_kwargs["max_tokens"] = 4096
        model_kwargs["temperature"] = self.temperature
        # write the response to a file if it doesn't exist
        model_fct = get_model_function(model, **model_kwargs)
        response_from_llm = model_fct(prompt)

        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(
            prob_data, response_from_llm, previous_code, num_steps
        )

    @staticmethod
    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data["sub_steps"][num_steps - 1]["function_header"]
        return_str = prob_data["sub_steps"][num_steps - 1]["return_line"]
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(self, problem_data: dict, num_steps: int):
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(
                problem_data["sub_steps"][i]["step_description_prompt"]
                + "\n"
                + problem_data["sub_steps"][i]["step_background"]
                if self.with_background
                else problem_data["sub_steps"][i]["step_description_prompt"]
            )
            output_lines.append(self.previous_llm_code[i])
            previous_code.append(self.previous_llm_code[i])
            output_lines.append("------")

        next_step.append(
            problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
            + "\n"
            + problem_data["sub_steps"][num_steps - 1]["step_background"]
            if self.with_background
            else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
        )
        next_step.append(self.process_problem_code(problem_data, num_steps))
        # print(output_lines)
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str

    def generate_prompt_with_steps(
        self, prob_data: dict, num_steps: int, prompt_template=DEFAULT_PROMPT_TEMPLATE
    ):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = (
            self.process_problem_steps(prob_data, num_steps)
        )
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f"{dependencies}\n{previous_code_str}\n"


def subprocess_job(
    chunk,
    prompt_template,
    model,
    output_dir,
    prompt_dir,
    with_background,
    temperature,
    worker_id,
    hyperparameters,
):
    gcode = Gencode(
        model=model,
        output_dir=output_dir,
        prompt_dir=prompt_dir,
        with_background=with_background,
        temperature=temperature,
        hyperparameters=hyperparameters,
    )

    model, tokenizer = load_hf_model_for_min_entropy_generation(
        model,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    for problem in tqdm(
        chunk,
        desc=f"Processing Problems on Worker {worker_id} (process {os.getpid()})",
        position=worker_id,
        leave=True,
    ):
        prob_id = problem["problem_id"]
        steps = len(problem["sub_steps"])

        for i in range(steps):
            if (
                (prob_id == "13" and i == 5)
                or (prob_id == "62" and i == 0)
                or (prob_id == "76" and i == 2)
            ):
                continue
            gcode.generate_min_ent_response_with_steps(
                prob_data=problem,
                num_steps=i + 1,
                tot_steps=steps,
                model=model,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
            )


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split manner",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Output directory",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("eval_results", "prompt"),
        help="Prompt directory",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Generation temperature",
    )
    parser.add_argument("--mode", type=str, default="normal", help="Model name")
    parser.add_argument("--current_iter", type=int, default=3)
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default="{'tmax': 1, 'tmin': 0.01, 'max_iter': 100, 'tol': 0.1, 'target_ratio': 0.85, 'target_threshold': 1e-05}",
        help="Model name",
    )
    return parser


def main(
    model: str,
    split: str,
    output_dir: Path,
    prompt_dir: Path,
    with_background: bool,
    temperature: float,
    mode: bool = "normal",
    current_iter: int = 0,
    hyperparameters: str = "",
) -> None:
    print(f"with_background: {with_background}")
    prompt_template = (
        BACKGOUND_PROMPT_TEMPLATE if with_background else DEFAULT_PROMPT_TEMPLATE
    )
    data = read_from_hf_dataset(split)

    if mode == "min_entropy":
        print("Generating with entropy minimization inference mode")
        ################################# Data Parallelism Inference setup #################################
        try:
            mp.set_start_method("spawn")  # Important for CUDA safety!
        except RuntimeError:
            pass
        num_processes = 14
        model = model.replace("openai/", "")
        ####################################################################################################

        # --- 1. Prepare and Chunkify prompts ---
        data = [item for item in data]
        k, m = divmod(len(data), num_processes)
        problem_chunks = [
            data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_processes)
        ]

        # --- 2. Create and Start Processes ---
        print(
            f"\nStarting inference on {len(data)} prompts across {num_processes} processes..."
        )

        hyperparameters = json.loads(hyperparameters)
        print(f"Hyperparameters: {hyperparameters}")

        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(
                subprocess_job,
                [
                    (
                        chunk,
                        prompt_template,
                        model,
                        output_dir,
                        prompt_dir,
                        with_background,
                        temperature,
                        worker_id,
                        hyperparameters,
                    )
                    for worker_id, chunk in enumerate(problem_chunks)
                ],
            )
    elif mode == "self_refinement":
        print(
            f"Generating with self-refinement inference mode: iteration {current_iter}"
        )
        gcode = Gencode(
            model=model,
            output_dir=output_dir,
            prompt_dir=prompt_dir,
            with_background=with_background,
            temperature=temperature,
        )
        for problem in data:
            prob_id = problem["problem_id"]
            steps = len(problem["sub_steps"])
            print(f"Generating {prob_id}...")
            for i in range(steps):
                if (
                    (prob_id == "13" and i == 5)
                    or (prob_id == "62" and i == 0)
                    or (prob_id == "76" and i == 2)
                ):
                    continue
                gcode.generate_self_refinement_response_with_steps(
                    problem,
                    i + 1,
                    steps,
                    model,
                    prompt_template,
                    iteration=current_iter,
                )
    elif mode == "temp_decrease":
        hyperparameters = json.loads(hyperparameters)
        print(f"Generating with temperature decrease inference mode: {hyperparameters}")
        gcode = Gencode(
            model=model,
            output_dir=output_dir,
            prompt_dir=prompt_dir,
            with_background=with_background,
            temperature=temperature,
            hyperparameters=hyperparameters,
        )
        for problem in data:
            prob_id = problem["problem_id"]
            steps = len(problem["sub_steps"])
            print(f"Generating {prob_id}...")
            for i in range(steps):
                if (
                    (prob_id == "13" and i == 5)
                    or (prob_id == "62" and i == 0)
                    or (prob_id == "76" and i == 2)
                ):
                    continue
                gcode.generate_temp_decrease_response_with_steps(
                    problem, i + 1, steps, model, prompt_template
                )
    else:
        gcode = Gencode(
            model=model,
            output_dir=output_dir,
            prompt_dir=prompt_dir,
            with_background=with_background,
            temperature=temperature,
        )
        for problem in data:
            prob_id = problem["problem_id"]
            steps = len(problem["sub_steps"])
            print(f"Generating {prob_id}...")
            for i in range(steps):
                if (
                    (prob_id == "13" and i == 5)
                    or (prob_id == "62" and i == 0)
                    or (prob_id == "76" and i == 2)
                ):
                    continue
                gcode.generate_response_with_steps(
                    problem, i + 1, steps, model, prompt_template
                )


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
