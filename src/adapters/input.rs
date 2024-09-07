use std::fs;
use text_splitter::TextSplitter;
use std::path::PathBuf;
use anyhow::{Context, Result};

pub struct FileInputAdapter {
    input_folder: PathBuf,
    chunk_size: usize,
}

impl FileInputAdapter {
    pub fn new(input_folder: PathBuf, chunk_size: usize) -> Self {
        Self {
            input_folder: input_folder,
            chunk_size,
        }
    }

    fn read_files_in_folder(&self) -> Result<Vec<String>> {
        let mut contents = Vec::new();
        for entry in fs::read_dir(&self.input_folder)
            .with_context(|| format!("Failed to read directory: {:?}", self.input_folder))?
        {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "txt" || extension == "md" {
                        println!("Reading file: {:?}", path.to_str().unwrap());
                        let content = fs::read_to_string(&path)
                            .with_context(|| format!("Failed to read file: {:?}", path))?;
                        contents.push(content);
                    }
                }
            }
        }
        Ok(contents)
    }

    pub fn fetch_chunks(&self) -> Result<Vec<String>> {
        let file_contents = self.read_files_in_folder()?;
        let mut all_chunks = Vec::new();
        let splitter = TextSplitter::new(self.chunk_size);
        for content in file_contents {
            let chunks: Vec<String> = splitter.chunks(&content).map(|s| s.to_string()).collect();
            all_chunks.extend(chunks);
        }
        Ok(all_chunks)
    }
}

