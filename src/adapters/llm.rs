use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::utils::llm::FromLLMResponse;
use anyhow::Result;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug)]
pub enum OutputFormat {
    String,
    Json,
    StrictJson(Value),
}

pub trait LLMProvider: LLMInterface {
    fn generate_headers(&self) -> Result<HeaderMap>;
    fn generate_request_body(
        &self,
        sys_prompt: &str,
        user_prompt: &str,
        output_format: &OutputFormat,
    ) -> Result<Value>;

}

pub trait LLMInterface {
    fn send_request<T: FromLLMResponse + Send + Sync>(
        &self,
        sys_prompt: &str,
        user_prompt: &str
    ) -> impl std::future::Future<Output = Result<T>>;

    fn upload_file(&self, file_path: PathBuf) -> impl std::future::Future<Output = Result<Value>>;

    fn create_fine_tuning_job(&self, training_file: &str) -> impl std::future::Future<Output = Result<Value>>;

    fn train(&self, file_path: PathBuf) -> impl std::future::Future<Output = Result<()>>;
}


