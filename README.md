# NeuralAssimilator

NeuralAssimilator is a Rust crate for fine-tuning Language Learning Models (LLMs) from unstructured text.

## Features
- Generate prompts based on specified use cases
- Create instruction-response pairs for fine-tuning
- Output results in JSONL format
- Perform training on your LLM provider with the generated dataset
 
## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
neuralassimilator = "0.1.0"
```

## Usage

### Command-line Interface

NeuralAssimilator can be used via its command-line interface:

```bash
neuralassimilator --input ./input_folder --output ./output_folder --chunk-size 10000 --model gpt-4o-mini-2024-07-18 --use-case "Creative writing"
```

### Arguments

- `--input` or `-i`: Input directory path (default: "./input")
- `--output` or `-o`: Output file or directory path (optional)
- `--chunk-size`: Size of text chunks to process (default: 10000)
- `--model`: LLM model to use (default: "gpt-4o-mini-2024-07-18")
- `--use-case`: Specific use case for prompt generation (default: "Creative writing")

## How it Works

1. **Input Processing**: The crate reads input files from the specified directory and chunks them into manageable sizes.
2. **Prompt Tuning**: Based on the given use case, it generates appropriate prompts for the LLM.
3. **Instruction Generation**: For each chunk-prompt pair, it generates instruction-response pairs using the specified LLM.
4. **Output**: The resulting pairs are written to a JSONL file in the specified output location.
5. **Fine-tuning**: The generated dataset can then be used to fine-tune the LLM.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).