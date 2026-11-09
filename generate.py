"""Generate a TinyStories continuation from a saved checkpoint."""

import tiktoken
import torch

from train import load_checkpoint_model, select_device


def generate_text(settings, checkpoint_path, prompt=None):
    generation = settings["generation"]
    prompt = generation["prompt"] if prompt is None else prompt
    if not prompt:
        raise ValueError("Generation prompt must not be empty")
    device = select_device(settings["training"]["device"])
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    torch.manual_seed(generation.get("seed", checkpoint["seed"]))
    saved_config = checkpoint["run_config"]
    tokenizer_name = (saved_config["tokenizer"]["name"] if "tokenizer" in saved_config
                      else "gpt2")
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    input_ids = torch.tensor([tokenizer.encode_ordinary(prompt)],
                             dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, generation["max_new_tokens"],
                                temperature=generation["temperature"],
                                top_k=generation["top_k"], eos_token_id=tokenizer.eot_token)
    ids = output_ids[0].tolist()
    if ids[-1] == tokenizer.eot_token:
        ids.pop()
    output = tokenizer.decode(ids)
    print(output)
    return output


if __name__ == "__main__":
    from run import main
    import sys
    main(["generate", *sys.argv[1:]])
