use std::fs;
use text_splitter::TextSplitter;
use std::path::PathBuf;
use anyhow::{Context, Result};
use log::{info, debug};

pub struct FileInputAdapter {
    input_folder: PathBuf,
    chunk_size: usize,
}

impl FileInputAdapter {
    pub fn new(input_folder: PathBuf, chunk_size: usize) -> Self {
        info!("Creating new FileInputAdapter with input folder: {:?} and chunk size: {}", input_folder, chunk_size);
        Self {
            input_folder: input_folder,
            chunk_size,
        }
    }

    fn read_files_in_folder(&self) -> Result<Vec<String>> {
        info!("Reading files from folder: {:?}", self.input_folder);
        let mut contents = Vec::new();
        for entry in fs::read_dir(&self.input_folder)
            .with_context(|| format!("Failed to read directory: {:?}", self.input_folder))?
        {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "txt" || extension == "md" {
                        debug!("Reading file: {:?}", path.to_str().unwrap());
                        let content = fs::read_to_string(&path)
                            .with_context(|| format!("Failed to read file: {:?}", path))?;
                        contents.push(content);
                    }
                }
            }
        }
        info!("Read {} files from folder", contents.len());
        Ok(contents)
    }

    pub fn fetch_chunks(&self) -> Result<Vec<String>> {
        info!("Fetching chunks with size: {}", self.chunk_size);
        let file_contents = self.read_files_in_folder()?;
        let mut all_chunks = Vec::new();
        let splitter = TextSplitter::new(self.chunk_size);
        for (index, content) in file_contents.iter().enumerate() {
            let chunks: Vec<String> = splitter.chunks(content).map(|s| s.to_string()).collect();
            debug!("Split content {} into {} chunks", index + 1, chunks.len());
            all_chunks.extend(chunks);
        }
        info!("Total chunks fetched: {}", all_chunks.len());
        Ok(all_chunks)
    }
}
