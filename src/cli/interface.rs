use anyhow::{Context, Result};
use crate::adapters::input::*;
use tokio::sync::Mutex;
use std::{
    fs::File,
    io:: BufWriter,
    sync::Arc,
    path::PathBuf,
};
use tokio;
use crate::utils::lib::*;
use clap::Parser;
use crate::adapters::openai::*;
use crate::core::assimilator::*;

pub async fn run_cli_interface() -> Result<()> {
    let args = Args::parse();
    let output_path = get_output_file_path(args.output);
    let writer = create_writer(&output_path)
        .context("Failed to create writer")?;
    let llm = OpenAI::new(args.model, 1.0, 16000);
    let assimilator = Assimilator::new(llm, writer);
    
    let prompts = assimilator.tune_prompt(&args.use_case).await
        .context("Failed to tune prompt")?;
    
    let chunk_prompt_pairs = prepare_chunk_prompt_pairs(args.input, args.chunk_size, &prompts)
        .context("Failed to prepare chunk-prompt pairs")?;
    
    assimilator.harvest(chunk_prompt_pairs).await
        .context("Failed to harvest chunk-prompt pairs")?;
    
    Ok(())
}

fn prepare_chunk_prompt_pairs(
    input_path: PathBuf,
    chunk_size: usize,
    prompts: &[String]
) -> Result<Vec<(String, String)>> {
    let input_adapter = FileInputAdapter::new(input_path, chunk_size);
    let chunks = input_adapter.fetch_chunks()
        .context("Failed to fetch chunks from input")?;
    let chunk_prompt_pairs = create_chunk_prompt_pairs(&chunks, prompts);
    Ok(chunk_prompt_pairs)
}

fn create_writer(output_path: &PathBuf) -> Result<Arc<Mutex<BufWriter<File>>>> {
    let file = File::create(output_path)
        .context("Failed to create output file")?;
    let writer: Arc<Mutex<BufWriter<File>>> = Arc::new(Mutex::new(BufWriter::new(file)));
    Ok(writer)
}