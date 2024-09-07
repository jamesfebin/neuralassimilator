use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::sync::Arc;
use std::io::Write;
use tokio::sync::Mutex;
use crate::core::learn::Instruction;

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct JsonLine {
    messages: Vec<Message>,
}

pub async fn output_jsonl(
    writer: &Arc<Mutex<std::io::BufWriter<File>>>, 
    instruction: Instruction
) -> Result<()> {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a highly intelligent, creative and helpful AI assistant.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: instruction.instruction,
        },
        Message {
            role: "assistant".to_string(),
            content: instruction.response,
        },
    ];

    let json_line = JsonLine { messages };
    let json = serde_json::to_string(&json_line)
        .context("Failed to serialize JsonLine to string")?;

    let mut writer = writer.lock().await;
    writeln!(writer, "{}", json)
        .context("Failed to write JSON line to file")?;

    Ok(())
}