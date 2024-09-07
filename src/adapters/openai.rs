use super::llm::{LLMProvider,LLMInterface, OutputFormat};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION};
use serde_json::{json, Value};
use std::env;
use crate::utils::llm::FromLLMResponse;
use tokio::time::Duration;
use crate::utils::lib::*;
use anyhow::{Context, Result};

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
    fn generate_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    fn generate_request_body(&self, sys_prompt: &str, user_prompt: &str, output_format: &OutputFormat) -> Value {
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

        body
    }

    
}

impl LLMInterface for OpenAI {
    async fn send_request<T: FromLLMResponse + Send + Sync>(&self, sys_prompt: &str, user_prompt: &str) -> Result<T> {
        retry(self.max_retries, self.delay, || async {
            let client = reqwest::Client::new();
            let headers = self.generate_headers();
            let output_format = T::output_format();
            let body = self.generate_request_body(sys_prompt, user_prompt, &output_format);
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
                T::from_llm_response(content)
            } else {
                let error_text = response.text().await
                    .context("Failed to get error text from OpenAI API")?;
                anyhow::bail!("OpenAI API request failed: {}", error_text)
            }
        }).await
    }

}


