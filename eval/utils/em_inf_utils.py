import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from transformers.cache_utils import Cache
from transformers.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
)

from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)


class AdaptiveTemperatureProcessor:
    def __init__(
        self,
        tmax=1,
        tmin=0.01,
        max_iter=100,
        tol=0.1,
        target_ratio=0.2,
        target_threshold=1e-6,
    ):
        # self.tokenizer = tokenizer
        # self.mask = None
        self.t_max = tmax
        self.t_min = tmin
        self.max_iter = max_iter
        self.tol = tol
        self.target_ratio = target_ratio
        self.target_threshold = target_threshold

    def entropy(self, p):
        return -(p * torch.log(p + 1e-9)).sum()

    def __call__(self, input_ids, scores):
        """Modify the logits before softmax"""
        t_min, t_max = self.t_min, self.t_max
        p = torch.softmax(scores, dim=-1)
        H = self.entropy(p)
        # print(f'initial entropy: {H}')
        target_entropy = max(self.target_threshold, self.target_ratio * H)
        # print(f'target_entropy: {target_entropy}') #0.5546875
        final_tempertature = 1
        if H > self.target_threshold:
            for _ in range(self.max_iter):
                T_mid = (t_max + t_min) / 2
                p = torch.softmax(scores / T_mid, dim=-1)
                H = self.entropy(p)
                # print('H', H) # 0.0225
                # print('T_mid', T_mid)
                if abs(H - target_entropy) < self.tol:
                    final_tempertature = T_mid
                    break
                elif H < target_entropy:
                    t_min = T_mid
                else:
                    t_max = T_mid
                final_tempertature = T_mid

        # print('final_tempertature', final_tempertature)
        # temp = T_mid  # best estimate
        scores_processed = scores / final_tempertature

        # input('press enter to continue')
        return scores_processed


def generate_with_ent_minimization(
    model, tokenizer, prompt, max_gen_steps=4096, n_grad_steps=20, learning_rate=0.1
):
    model.eval()

    tokenized_input = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        model.device
    )
    input_ids = tokenized_input.clone()
    prompt_id_len = input_ids.shape[1]

    for step in range(max_gen_steps):
        # Get the logits from the model
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[:, -1, :].detach().clone().requires_grad_(True)

        # Make logits trainable
        logits_param = torch.nn.Parameter(logits)
        optimizer = torch.optim.Adam([logits_param], lr=learning_rate)

        # Compute and minimize entropy
        for grad_step in range(n_grad_steps):
            optimizer.zero_grad()
            probs = F.softmax(logits_param, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))
            # pd = F.softmax(logits, dim=-1)
            # log_sum_exp = torch.logsumexp(logits, dim=-1)
            # sum_pd_logits = torch.sum(pd * logits, dim=-1)
            # entropy = log_sum_exp - sum_pd_logits

            entropy.backward()
            optimizer.step()
        # print(f"Entropy after: {entropy.item()}")

        # Use the updated logits to sample a token
        with torch.no_grad():
            probs = F.softmax(logits_param, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Attach the token and continue generation
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode and print the result
    generated_text = tokenizer.decode(
        input_ids[0][prompt_id_len:], skip_special_tokens=False
    )

    return generated_text


def ent_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    ### Parameters for entropy minimization
    learning_rate = getattr(generation_config, "learning_rate", 0.1)
    n_grad_steps = getattr(generation_config, "n_grad_steps", 5)
    kl_weight = getattr(generation_config, "kl_weight", 0.001)
    threshold = getattr(generation_config, "threshold", 0.3)
    # print(
    #     f"Entropy minimization parameters: learning_rate={learning_rate}, n_grad_steps={n_grad_steps}, kl_weight={kl_weight}"
    # )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = (
            model_kwargs["past_key_values"].is_compileable
            and self._supports_static_cache
        )
        is_compileable = is_compileable and not self.generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda"
            or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        with torch.no_grad():
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # next_token_logits = outputs.logits[:, -1, :].clone().float()

        next_token_logits = outputs.logits[:, -1, :].clone().requires_grad_(True)
        next_token_logits = next_token_logits.to(torch.float32)

        next_token_logits_new = next_token_logits.clone()
        original_probs = F.softmax(next_token_logits_new, dim=-1).detach()

        # Make logits trainable
        logits_param = torch.nn.Parameter(next_token_logits)
        optimizer = torch.optim.Adam([logits_param], lr=learning_rate)

        with torch.enable_grad():
            for grad_step in range(n_grad_steps):
                optimizer.zero_grad()

                probs = F.softmax(logits_param, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                if entropy < threshold:
                    break
                # print(
                #     f"Step {grad_step} - Entropy: {entropy.item()} | min: {entropy.min().item()} | max: {entropy.max().item()}"
                # )

                kl_div = F.kl_div(
                    input=probs.log(), target=original_probs, reduction="batchmean"
                )
                loss = kl_weight * kl_div + entropy
                loss.backward()
                optimizer.step()
            # print(f"Entropy after: {entropy.item()}")

            # with torch.no_grad():
            # probs = F.softmax(logits_param, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token_logits = logits_param.detach().clone()
            del logits_param, optimizer

            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            probs = torch.clamp(probs, min=1e-9, max=1)
            # print(f"probs: {probs.mean()}")
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def load_hf_model_for_min_entropy_generation(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    trust_remote_code=True,
):
    """
    Load hf model for entropy minimization.
    """
    if "qwen2" in model_name.lower():
        from transformers import Qwen2ForCausalLM

        try:
            original_generate = Qwen2ForCausalLM.generate.__wrapped__

            # Create a new version without the no_grad decorator
            def new_generate(self, *args, **kwargs):
                # This calls the original implementation but without the no_grad decorator
                return original_generate(self, *args, **kwargs)

            Qwen2ForCausalLM.generate = new_generate
        except:
            pass

        Qwen2ForCausalLM._sample = ent_sample

        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model.eval()
        return model, tokenizer
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM

        try:
            original_generate = LlamaForCausalLM.generate.__wrapped__

            # Create a new version without the no_grad decorator
            def new_generate(self, *args, **kwargs):
                # This calls the original implementation but without the no_grad decorator
                return original_generate(self, *args, **kwargs)

            LlamaForCausalLM.generate = new_generate
        except:
            pass

        LlamaForCausalLM._sample = ent_sample

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model.eval()
        return model, tokenizer
    else:
        raise ValueError("Model not supported for entropy minimization.")


def min_entropy_inference(
    model_name,
    prompt_chunk,
    worker_id,
    temperature=0.1,
    hyperparameters={
        "threshold": 0.3,
        "learning_rate": 0.1,
        "n_grad_steps": 5,
        "kl_weight": 0.001,
    },
    max_new_tokens=4096,
):
    model, tokenizer = load_hf_model_for_min_entropy_generation(
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    generation_config = model.generation_config

    generation_config.learning_rate = hyperparameters["learning_rate"]
    generation_config.n_grad_steps = hyperparameters["n_grad_steps"]
    generation_config.kl_weight = hyperparameters["kl_weight"]
    generation_config.threshold = hyperparameters["threshold"]

    if worker_id == 0:
        print(
            f"threshold: {generation_config.threshold} | learning_rate: {generation_config.learning_rate} | n_grad_steps: {generation_config.n_grad_steps} | kl_weight: {generation_config.kl_weight} | max_new_tokens: {max_new_tokens} | temperature: {temperature}"
        )

    outputs = []
    for i, prompt in tqdm(
        enumerate(prompt_chunk),
        total=len(prompt_chunk),
        desc=f"Processing prompts on Worker {worker_id} (process {os.getpid()})",
        position=worker_id,
        leave=True,
    ):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)
        prompt_ids_len = input_ids["input_ids"].shape[-1]

        output_ids = model.generate(
            **input_ids,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.decode(
            output_ids[0][prompt_ids_len:],
            skip_special_tokens=True,
        )
        outputs.append(output)

    return outputs


if __name__ == "__main__":
    model_name = "allenai/Llama-3.1-Tulu-3-8B"
    prompt1 = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": "Let $a$ and $b$ be the two real values of $x$ for which\\[\\sqrt[3]{x} + \\sqrt[3]{20 - x} = 2\\]The smaller of the two values can be expressed as $p - \\sqrt{q}$, where $p$ and $q$ are integers. Compute $p + q$.",
        },
    ]
    prompt2 = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": "What is the sum of the first 100 positive integers?",
        },
    ]
    prompt3 = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": "What is the fourth smallest prime integer?",
        },
    ]
    prompt4 = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": "What is the cubic root of 10?",
        },
    ]
    prompts = [prompt1, prompt2, prompt3, prompt4]

    # model, tokenizer = load_hf_model_for_min_entropy_generation(
    #     model_name,
    #     device_map="auto",
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    # )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
    conversations = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    print(f"Conversations: {conversations}")

    # tokenized_prompt = tokenizer(
    #     conversations[0], return_tensors="pt", padding=True, padding_side="left"
    # ).to(model.device)

    # prompt_id_len = tokenized_prompt["input_ids"].shape[-1]

    # print(tokenized_prompt)

    # print(
    #     f"input_ids: {tokenized_prompt['input_ids'].shape} | attention_mask: {tokenized_prompt['attention_mask'].shape}"
    # )
    output = min_entropy_inference(
        "Qwen/Qwen2.5-Math-7B",
        conversations[0],
        0,
        {"kl_weight": 0.001, "learning_rate": 0.1, "n_grad_steps": 5},
    )

    # generated_text = model.generate(
    #     input_ids=tokenized_prompt["input_ids"],
    #     attention_mask=tokenized_prompt["attention_mask"],
    #     max_new_tokens=2048,
    #     pad_token_id=tokenizer.eos_token_id,
    # )
    # generated_text = tokenizer.batch_decode(
    #     generated_text[:, prompt_id_len:], skip_special_tokens=True
    # )
    print(output)
