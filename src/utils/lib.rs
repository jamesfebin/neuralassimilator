use indicatif::{ProgressBar, ProgressStyle};
use std::io;
use clap::Parser;
use std::path::PathBuf;
use std::fs::create_dir_all;
use anyhow::{Context, Result};

use chrono::Local;

use crate::adapters::input::FileInputAdapter;
use tokio::time::{sleep, Duration};
use std::future::Future;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(short, long, value_parser, default_value = "./input")]
    pub input: PathBuf,

    #[clap(short, long, value_parser)]
    pub output: Option<PathBuf>,

    #[clap(long, default_value = "10000")]
    pub chunk_size: usize,

    #[clap(long, default_value = "gpt-4o-mini-2024-07-18")]
    pub model: String,

    #[clap(long, default_value = "Creative writing")]
    pub use_case: String,
}


pub trait ToVecString {
    fn to_vec_string<V: FromIterator<String>>(&self) -> V;
}

impl<T: AsRef<str>> ToVecString for &[T] {
    fn to_vec_string<V: FromIterator<String>>(&self) -> V {
        self.iter().map(|s| s.as_ref().to_string()).collect()
    }
}

pub fn get_output_file_path(output_arg: Option<PathBuf>) -> PathBuf {
    match output_arg {
        Some(path) => {
            if path.is_dir() {
                // If it's a directory, create it if it doesn't exist and use a timestamp-based filename
                create_dir_all(&path).expect("Failed to create output directory");
                path.join(Local::now().format("%Y%m%d_%H%M%S.jsonl").to_string())
            } else {
                // If it's not a directory, assume it's a file path
                if let Some(parent) = path.parent() {
                    create_dir_all(parent).expect("Failed to create output directory");
                }
                path
            }
        },
        None => {
            // Default case: use ./output directory with timestamp-based filename
            let output_dir = PathBuf::from("./output");
            create_dir_all(&output_dir).expect("Failed to create output directory");
            output_dir.join(Local::now().format("%Y%m%d_%H%M%S.jsonl").to_string())
        }
    }
}


pub fn create_pairs(input: PathBuf, chunk_size: usize, prompts: &[String]) -> Result<Vec<(String, String)>> {
    let input_adapter = FileInputAdapter::new(input, chunk_size);
    let chunks = input_adapter.fetch_chunks()
        .context("Failed to fetch chunks from input")?;
    Ok(create_chunk_prompt_pairs(&chunks, prompts))
}
pub fn create_progress_bar(total: u64) -> Result<ProgressBar, io::Error> {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

pub fn create_chunk_prompt_pairs(
    chunks: &Vec<String>,
    prompts: &[String]
) -> Vec<(String, String)> {
    chunks
        .into_iter()
        .flat_map(|chunk| {
            prompts
                .iter()
                .map(move |prompt| (chunk.clone(), prompt.to_owned()))
        })
        .collect()
}


pub async fn retry<F, Fut, T>(
    max_retries: u32,
    initial_delay: Duration,
    mut task: F
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut retries = 0;
    let mut delay = initial_delay;

    loop {
        match task().await {
            Ok(result) => return Ok(result),
            Err(e) if retries >= max_retries => {
                return Err(e).context(format!("Task failed after {} retries", max_retries))
            },
            Err(_) => {
                println!("Task failed. Retrying in {:?}...", delay);
                sleep(delay).await;
                retries += 1;
                delay *= 2; // Exponential backoff
            }
        }
    }
}