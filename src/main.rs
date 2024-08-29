use serde_json::json;
use std::{error::Error, fmt::format, fs::File, io::{self, Read, Write}};
use serde_json::Value;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION};
use std::env;
use serde::Serialize;
use std::thread;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::task::JoinSet;
use csv::Writer;
use text_splitter::TextSplitter;
use std::path::Path;


use once_cell::sync::Lazy;
struct Segment {
    content: String,
    start_index: usize,
    end_index: usize,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct JsonLine {
    messages: Vec<Message>,
}


pub static RESPONSE_FORMAT: Lazy<Value> = Lazy::new(|| {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "instruction_response_pair",
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

struct InstructionResponsePair {
    instruction: String,
    response: String,
}

fn read_file(filename: &str) -> io::Result<String> {
    std::fs::read_to_string(format!("./input/{}", filename))
}

fn read_files_in_folder(folder_path: &str) -> io::Result<Vec<String>> {
    let mut contents = Vec::new();
    for entry in std::fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let content = std::fs::read_to_string(path)?;
            contents.push(content);
        }
    }
    Ok(contents)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // let extraction_prompt = r#"
    // Extract a single instruction-response pair from the provided text. The instruction should be clear and concise. The response must be extracted verbatim from the original text, without adding any external information. It should be comprehensive, between 5 paragraphs to a full page in length. Ensure the extracted response demonstrates in-depth knowledge, logical flow, and proper structure. It is mandatory to use only the content from the original text for the response. The goal is to create high-quality training data for fine-tuning a large language model to produce detailed, well-structured responses based solely on given information. Include code examples in the instruction when neccessary to simulate how developers would ask in a real world scenario.
    // "#;
    let extraction_prompt = r#"
Given a body of information containing insights and knowledge on the respective topic: 1. Analyze the key concepts, insights, knowledge, and details presented in the text. 2. Formulate a stackoverflow problem or scenario that could be solved using the information from the text. Don't use the information as template but think of concrete scenario that would be asked by a developer in stackoveflow. The problem should be: - Clear and concise - Problem-solving oriented rather than explanation-focused - Include code examples or technical context when appropriate to simulate real-world developer queries 3. Craft a comprehensive solution to the problem, ensuring it: - Uses only information present in the original text - Demonstrates insight and provides intuition into problem-solving - Shows in-depth knowledge of the subject matter - Follows a logical flow - Has proper structure (e.g. paragraphs, sections) - Is between 5 paragraphs to a full page in length 4. Present the result as an instruction-response pair: - Instruction: The formulated problem/scenario - Response: The crafted solution The goal is to create high-quality training data for fine-tuning a large language model to produce detailed, well-structured responses to problem-solving scenarios based solely on given information.    "#;

    let input_folder = "./input";
    let file_contents = read_files_in_folder(input_folder)?;
    
    let mut all_segments = Vec::new();
    let splitter = TextSplitter::new(10000);
    
    for content in file_contents {
        let segments: Vec<String> = splitter.chunks(&content).map(|s| s.to_string()).collect();
        all_segments.extend(segments);
    }
    
    let file = File::create("output.jsonl")?;
    let mut writer = io::BufWriter::new(file);
    let total_segments = all_segments.len();
    let progress_bar = ProgressBar::new(total_segments as u64);
    progress_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} segments processed ({eta})")
        .unwrap()
        .progress_chars("##-"));

    let mut join_set = JoinSet::new();
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(8));

    for segment in all_segments {
        let content = segment.to_string();
        let extraction_prompt = extraction_prompt.to_string();
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let progress_bar = progress_bar.clone();

        join_set.spawn(async move {
            let _permit = permit;
            let result = request(&extraction_prompt, &content).await;
            progress_bar.inc(1);
            thread::sleep(Duration::from_millis(200));
            result
        });
    }

    let mut responses = Vec::new();
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(response_tuple)) => {
                responses.push(response_tuple);
            }
            Ok(Err(e)) => {
                eprintln!("Error processing segment: {}", e);
            }
            Err(e) => {
                eprintln!("Task join error: {}", e);
            }
        }
    }

    progress_bar.finish_with_message("All segments processed");

    for (user_input, ai_response) in responses {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a highly skilled rust system's programmer, highly skilled at writing composable, performant, elegant and reliable rust code".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: user_input,
            },
            Message {
                role: "assistant".to_string(),
                content: ai_response,
            },
        ];

        let json_line = JsonLine { messages };
        let json = serde_json::to_string(&json_line).unwrap();
        writeln!(writer, "{}", json)?;
    }

    Ok(())
}


async fn request(extraction_prompt: &str, segment: &str) -> Result<(String, String),  Box<dyn std::error::Error + Send>> {
    // Retrieve the OpenAI API key from environment variables
    let api_key = env::var("OPENAI_API_KEY").map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

    // Create the request body
    let request_body = json!({
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {
                "role": "system",
                "content": "You are a highly skilled and experienced LLM finetuning expert"
            },
            {
                "role": "user",
                "content": format!("{} {}", extraction_prompt, segment)
            }
        ],
        "response_format": *RESPONSE_FORMAT
    });
    // println!("{:?}", request_body);

    // Create a header map
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?);
    // println!("{:?}", headers);
    // Send the request
    let client = reqwest::Client::new();
    let response = client.post("https://api.openai.com/v1/chat/completions")
        .headers(headers)
        .json(&request_body)
        .send()
        .await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

    // Check the response status
    if response.status().is_success() {
        let response_json: Value = response.json().await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

        // Extract the desired information
        let content = &response_json["choices"][0]["message"]["content"];
        let extracted: Value = serde_json::from_str(content.as_str().unwrap()).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;
        
        let instruction = extracted["instruction"].as_str().unwrap().to_string();
        let response = extracted["response"].as_str().unwrap().to_string();
        println!("Processing segment: {}...", &segment.chars().take(50).collect::<String>());
        println!("{}", instruction);
        // println!("{}", response);
        
        // Return the instruction and response as a tuple
        Ok((instruction, response))
    } else {
        eprintln!("Error: {}", response.status());
        println!("{:?}", response);
        Err(Box::from("Failed to get a successful response".to_string()) as Box<dyn std::error::Error + Send + Sync>)
    }
}
