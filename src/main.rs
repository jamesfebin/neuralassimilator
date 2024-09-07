use anyhow::Result;
use lib::cli::interface::run_cli_interface;

#[tokio::main]
async fn main() -> Result<()> {
    run_cli_interface().await?;
    Ok(())
}