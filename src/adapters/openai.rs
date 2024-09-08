use super::llm::{LLMProvider,LLMInterface, OutputFormat};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION};
use serde_json::{json, Value};
use std::env;
use crate::utils::llm::FromLLMResponse;
use tokio::time::Duration;
use crate::utils::lib::*;
use anyhow::{Context, Result};
use std::path::PathBuf;
use reqwest::multipart::{Form, Part};
use log::{info, debug, error};


pub struct OpenAI {
    model: String,
    temperature: f32,
    max_tokens: u32,
    max_retries: u32,
    delay: Duration
}

impl OpenAI {
    pub fn new(model: String, temperature: f32, max_tokens: u32) -> Self {
        Self { model, temperature, max_tokens, max_retries: 3, delay: Duration::from_secs(1) }
    }

    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}
 
impl LLMProvider for OpenAI {
    fn generate_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY must be set")?;
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))
            .context("Failed to create Authorization header")?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    fn generate_request_body(&self, sys_prompt: &str, user_prompt: &str, output_format: &OutputFormat) -> Result<Value> {
        let messages = vec![
            json!({"role": "system", "content": sys_prompt}),
            json!({"role": "user", "content": user_prompt}),
        ];

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        });

        match output_format {
            OutputFormat::String => {},
            OutputFormat::Json => {
                body["response_format"] = json!({"type": "json_object"});
            },
            OutputFormat::StrictJson(schema) => {
                body["response_format"] = schema.clone();
            },
        }

        Ok(body)
    }

    
}

impl LLMInterface for OpenAI {
    async fn send_request<T: FromLLMResponse + Send + Sync>(&self, sys_prompt: &str, user_prompt: &str) -> Result<T> {
        retry(self.max_retries, self.delay, || async {
            let client = reqwest::Client::new();
            let headers = self.generate_headers()?;
            let output_format = T::output_format();
            let body = self.generate_request_body(sys_prompt, user_prompt, &output_format)?;
            let response = client.post("https://api.openai.com/v1/chat/completions")
                .headers(headers)
                .json(&body)
                .send()
                .await
                .context("Failed to send request to OpenAI API")?;

            if response.status().is_success() {
                let response_value = response.json::<Value>().await
                    .context("Failed to parse OpenAI API response as JSON")?;
                let content = response_value["choices"][0]["message"]["content"]
                    .as_str()
                    .context("Failed to extract content from OpenAI API response")?
                    .to_string();
                info!("OpenAI API request successful");
                debug!("Response: {:?}", response_value);
                T::from_llm_response(content)
            } else {
                let error_text = response.text().await
                    .context("Failed to get error text from OpenAI API")?;
                error!("OpenAI API request failed: {}", error_text);
                anyhow::bail!("OpenAI API request failed: {}", error_text)
            }
        }).await
    }

    async fn upload_file(&self, file_path: PathBuf) -> Result<Value> {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = reqwest::Client::new();

        let purpose = "fine-tune";

        let content = tokio::fs::read(&file_path)
            .await
            .context("Failed to read file")?;

        let file_name = file_path.file_name()
            .and_then(|n| n.to_str())
            .context("Failed to get file name")?
            .to_string();

        let part = Part::bytes(content)
            .file_name(file_name)
            .mime_str("application/json")
            .context("Failed to set MIME type")?;

        let form = Form::new()
            .part("file", part)
            .text("purpose", purpose.to_string());

        let response = client.post("https://api.openai.com/v1/files")
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await
            .context("Failed to send file upload request")?;

        if response.status().is_success() {
            let response_json: Value = response.json().await
                .context("Failed to parse upload response as JSON")?;
            info!("File uploaded successfully");
            debug!("Response: {:?}", response_json);
            Ok(response_json)
        } else {
            let error_text = response.text().await
                .context("Failed to get error text from upload response")?;
            error!("File upload failed: {}", error_text);
            Err(anyhow::anyhow!("File upload failed: {}", error_text))
        }
    }

     async fn create_fine_tuning_job(&self, training_file: &str) -> Result<Value> {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let client = reqwest::Client::new();

        let body = json!({
            "training_file": training_file,
            "model": self.model,
        });

        let response = client.post("https://api.openai.com/v1/fine_tuning/jobs")
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send fine-tuning job request")?;

        if response.status().is_success() {
            let response_json: Value = response.json().await
                .context("Failed to parse fine-tuning job response as JSON")?;
            info!("Fine-tuning job created successfully");
            debug!("Response: {:?}", response_json);
            Ok(response_json)
        } else {
            let error_text = response.text().await
                .context("Failed to get error text from fine-tuning job response")?;
            error!("Fine-tuning job creation failed: {}", error_text);
            Err(anyhow::anyhow!("Fine-tuning job creation failed: {}", error_text))
        }
    }

    async fn train(
        &self,
        file_path: PathBuf
    ) -> Result<()> {
        let upload_response = self.upload_file(file_path).await
            .context("Failed to upload training file")?;
        
        let training_file_id = upload_response["id"].as_str()
            .context("Failed to get training file ID from upload response")?;
     
        let fine_tuning_response = self.create_fine_tuning_job(
            training_file_id,
        ).await
        .context("Failed to create fine-tuning job")?;

        let status = fine_tuning_response["status"].as_str()
            .context("Failed to get status from fine-tuning job response")?
            .to_string();

        println!("Fine-tuning job created. Status: {}", status);
        debug!("Response: {:?}", fine_tuning_response);
        Ok(())
    }

    

    

}


