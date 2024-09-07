use uuid::Uuid;

pub struct Trainer {
    file_path: String,
    base_model: String,
}

pub struct Job {
    job_id: String,
    status: String,
    output_model: String,
}

pub trait Trainable {
    fn train(&self) -> Job;
}

impl Trainable for Trainer {
    fn train(&self) -> Job {
        Job {
            job_id: Uuid::new_v4().to_string(),
            status: "pending".to_string(),
            output_model: "".to_string(),
        }
    }
}