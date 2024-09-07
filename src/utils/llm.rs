
use crate::adapters::llm::OutputFormat;
use crate::core::prompts::TunedPrompts;
use serde_json::Value;
use serde_json::json;
use once_cell::sync::Lazy;
use crate::core::learn::Instruction;
use anyhow::{Result,Context};

pub trait FromLLMResponse: Sized {
  fn from_llm_response(response: String) -> Result<Self>;
  fn output_format() -> OutputFormat;
}

impl FromLLMResponse for String {
  fn from_llm_response(response: String) -> Result<Self> {
      Ok(response)
  }

  fn output_format() -> OutputFormat {
      OutputFormat::String
  }
}

impl FromLLMResponse for Value {
  fn from_llm_response(response: String) -> Result<Self> {
      serde_json::from_str(&response).context("Failed to parse JSON response")
  }

  fn output_format() -> OutputFormat {
      OutputFormat::Json
  }
}

impl FromLLMResponse for TunedPrompts {
  fn from_llm_response(response: String) -> Result<Self> {
      let response: Value = serde_json::from_str(&response)
          .context("Failed to parse JSON response")?;
      let response = response.as_object()
          .context("Response is not an object")?;
      let prompts = response
          .get("prompts")
          .context("Missing 'prompts' field")?
          .as_array()
          .context("'prompts' field is not an array")?;
      let prompts = prompts
          .iter()
          .map(|prompt| prompt.as_str().unwrap_or_default().to_string())
          .collect();
      Ok(TunedPrompts { prompts })
  }

  fn output_format() -> OutputFormat {
      OutputFormat::StrictJson(TUNED_PROMPTS_RESPONSE_FORMAT.clone())
  }
}

impl FromLLMResponse for Instruction {
  fn from_llm_response(response: String) -> Result<Self> {
      let response: Value = serde_json::from_str(&response)
          .context("Failed to parse JSON response")?;
      let instruction = response.get("instruction")
          .context("Missing 'instruction' field")?
          .as_str()
          .context("'instruction' is not a string")?
          .to_string();
      let response_text = response.get("response")
          .context("Missing 'response' field")?
          .as_str()
          .context("'response' is not a string")?
          .to_string();
      Ok(Instruction { instruction, response: response_text })
  }

  fn output_format() -> OutputFormat {
      OutputFormat::StrictJson(INSTRUCTION_RESPONSE_FORMAT.clone())
  }
}

pub static TUNED_PROMPTS_RESPONSE_FORMAT: Lazy<Value> = Lazy::new(|| {
  json!({
      "type": "json_schema",
      "json_schema": {
          "name": "prompts",
          "strict": true,
          "schema": {
              "type": "object",
              "properties": {
                  "prompts": {
                      "type": "array",
                      "items": {
                          "type": "string"
                      }
                  }
              },
              "required": ["prompts"],
              "additionalProperties": false
          }
      }
  })
});

pub static INSTRUCTION_RESPONSE_FORMAT: Lazy<Value> = Lazy::new(|| {
  json!({
      "type": "json_schema",
      "json_schema": {
          "name": "instruction",
          "strict": true,
          "schema": {
              "type": "object",
              "properties": {
                  "instruction": {
                      "type": "string"
                  },
                  "response": {
                      "type": "string"
                  }
              },
              "required": ["instruction", "response"],
              "additionalProperties": false
          }
      }
  })
});