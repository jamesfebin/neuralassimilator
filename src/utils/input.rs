use anyhow::{Context, Result};
use std::io::{self, Write};
use crate::core::prompts::generate_prompts;
use crate::core::assimilator::Assimilator;
use crate::adapters::llm::LLMProvider;

pub fn input_use_case_option() -> Result<usize> {
    loop {
        println!("Please select a use case from the following options:");
        println!("1. Creative Writing");
        println!("2. Problem Solving");
        println!("3. Code Problem Solving");
        println!("4. Explanation");
        println!("5. Custom");
        println!("Enter the number of your choice:");

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)
            .context("Failed to read line")?;

        match user_input.trim().parse() {
            Ok(num) if (1..=5).contains(&num) => return Ok(num),
            _ => {
                println!("Invalid option. Please enter a number between 1 and 5.");
            }
        }
    }
}

pub fn input_use_case_custom() -> Result<String> {
    loop {
        print!("Please enter the use case: ");
        io::stdout().flush()
            .context("Failed to flush stdout")?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)
            .context("Failed to read line")?;

        let trimmed = user_input.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        } else {
            println!("Use case cannot be empty. Please try again.");
        }
    }
}

pub async fn fetch_prompts<T: LLMProvider>(assimilator: &Assimilator<T>) -> Result<Vec<String>> {
    let use_case_option = input_use_case_option()
        .context("Failed to get use case option")?;

    let prompts = match use_case_option {
        1..=4 => generate_prompts(use_case_option),
        5 => {
            let use_case = input_use_case_custom()
                .context("Failed to get custom use case")?;
            assimilator.tune_prompt(&use_case).await
                .context("Failed to tune prompt")?
        },
        _ => unreachable!("input_use_case_option() should only return 1-5"),
    };

    Ok(prompts)
}