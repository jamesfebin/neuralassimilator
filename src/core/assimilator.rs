use anyhow::{Context, Result};
use std::sync::Arc;
use std::fs::File;
use futures::stream::{self, StreamExt};
use std::io::BufWriter;
use tokio::sync::Mutex;
use crate::adapters::output::*;
use crate::adapters::llm::*;
use crate::core::learn::*;
use crate::core::prompts::*;
use crate::utils::lib::*;

pub struct Assimilator<T: LLMProvider> {
    llm: T,
    writer: Arc<Mutex<BufWriter<File>>>
}

impl<T: LLMProvider> Assimilator<T> {
    pub fn new(llm: T, writer: Arc<Mutex<BufWriter<File>>>) -> Assimilator<T> {
        Self { llm, writer }
    }

    pub async fn tune_prompt(&self, use_case: &str) -> Result<Vec<String>> {
        let prompt = ROOT_GENERATION_PROMPTS.replace("{{USE CASE}}", use_case);
        let system_prompt = "You are a highly skilled and experienced LLM finetuning expert. You are provided with a use case and some examples and you need to generate prompts for that use case";
        let response: TunedPrompts = self.llm.send_request(system_prompt, &prompt)
            .await
            .context("Failed to send request to LLM for prompt tuning")?;
        Ok(response.prompts)
    }

    pub async fn harvest(&self, chunk_prompt_pairs: Vec<(String, String)>) -> Result<()> {
        let total_pairs = chunk_prompt_pairs.len();
        let progress_bar = create_progress_bar(total_pairs as u64)
            .context("Failed to create progress bar")?;

        let results: Vec<Result<()>> = stream::iter(chunk_prompt_pairs)
            .map(|(chunk, prompt)| {
                let progress_bar = progress_bar.clone();
                async move {
                    let instruction = self.form_learning_instruction(&prompt, &chunk)
                        .await
                        .context("Failed to form learning instruction")?;

                    output_jsonl(&self.writer, instruction)
                        .await
                        .context("Failed to write learning instruction to JSONL")?;
                    progress_bar.inc(1);
                    Ok(())
                }
            })
            .buffer_unordered(10)
            .collect()
            .await;

        progress_bar.finish_with_message("Processing complete");
        Ok(())
    }

    pub async fn form_learning_instruction(&self, prompt: &str, chunk: &str) -> Result<Instruction> {
        let system_prompt = "You are a highly skilled finetuning expert. You are provided with a prompt and a text and you need to extract a single instruction-response pair from the text that follows the prompt.";
        let user_prompt = format!("{}\n\n{}", prompt, chunk);
        self.llm.send_request(system_prompt, &user_prompt)
            .await
            .context("Failed to send request to LLM for forming learning instruction")
    }
}