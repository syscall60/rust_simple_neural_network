/// Returns a string of a progress bar based on the percentage and the size of the bar in parameters
/// # Arguments
/// * `percentage` - the percentage fo the operation
/// * `size` - the size of the progress bar (characters on the terminal) 
fn progress_bar(percentage: f64, size: usize) -> String {
	// the full part of the progress bar
	let repeat = percentage * size as f64 / 100.0;

	let full = "█".repeat(repeat.round() as usize);
	let empty = "▁".repeat(size - (repeat.round() as usize));

	[full, empty].join("")
}



/// Display the current phase of the training (with a progress bar)
/// # Arguments
/// * `current_step` - the current batch
/// * `last` - the number of batches
/// * `cost` - the current cost
pub fn display_progress(current_step: i32,last: usize,cost: f64,epoch : usize,total_epochs : usize) -> (){
	let percentage:f64 = current_step as f64 * 100.0 / last as f64;

	println!("{} {:.2}% EPOCH {epoch}/{total_epochs}",progress_bar(percentage,50),percentage);
	if cost>1000000.0 {
		println!("cost: +1000000");
	} else {
		println!("cost: {:.20}",cost)
	}
}