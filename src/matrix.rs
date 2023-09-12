use rand::{self, Rng};

#[macro_export]
macro_rules! matrix_at {
	($row:expr,$col:expr,$mat:expr) => {
		$mat.values[($row) * ($mat.cols) + ($col)]
	};
}


#[derive(Debug)]
pub struct Matrix<T>{
	pub rows : usize,
	pub cols : usize,
	pub values : Vec<T>
}


impl Matrix<f64> {

	pub fn new(row : usize,col : usize) -> Matrix<f64>{
		Matrix{
			rows: row,
			cols: col,
			values: vec![0.0;row*col],
		}
	}

	pub fn new_radom_gen_range(row : usize,col : usize, min	: f64, max: f64) -> Matrix<f64>{
		let mut values = vec![];
		let mut rng = rand::thread_rng();
		for _ in 0..row*col {
			values.push(rng.gen_range(min..max));
		}

		Matrix{
			rows: row,
			cols: col,
			values,
		}
	}

	pub fn new_dot_result(ma: &Self,mb: &Self) -> Self {
		Matrix { rows: ma.rows, cols: mb.cols, values: vec![0.0;ma.rows*mb.cols] }
	}

	pub fn dot(&self,dest : &mut Matrix<f64>,mb :&Matrix<f64>) {
		assert!(self.cols == mb.rows);
		assert!(dest.cols == mb.cols && dest.rows == self.rows);
		assert!(self.cols!=0 && mb.cols!=0);


		for j in 0..dest.cols {
			for i in 0..dest.rows {
				matrix_at!(i,j,dest)=0.0;
				for k in 0..self.cols {
					matrix_at!(i,j,dest) += matrix_at!(i,k,self) * matrix_at!(k,j,mb);
				}
			}
			
		}
	}

	pub fn add(&self,dest : &mut Matrix<f64>,mb :&Matrix<f64>) {
		assert!(self.cols == mb.cols && self.rows == mb.rows);

		for i in 0..self.rows {
			for j in 0..self.cols {
				matrix_at!(i,j,dest) = matrix_at!(i,j,self)+matrix_at!(i,j,mb);
			}
		}
	}

	/// Multiply two matrices with the The Hadamard product. The result is stored in the caller.
	/// 
	/// # Argument
	/// * `self` - caller Matrix, will be overwirtten 
	/// * `mb` - self will be multiply by this Matrx
	/// 
	/// Matrix should have the same dimensions
	pub fn multiply_by_mut(&mut self, mb : &Matrix<f64>){
			assert!(self.rows == mb.rows && self.cols == mb.cols);

			for (i, elem) in &mut self.values.iter_mut().enumerate() {
				*elem = *elem * mb.values[i];
			}
	}

	pub fn apply<T>(&self, function : fn(&Vec<f64>)->T)->T
	{
		function(&self.values)
	}

	pub fn apply_mut<T>(&mut self, function : fn(&mut Vec<f64>)->T)->T
	{
		function(&mut self.values)
	}

	pub fn dump(&self){
		for i in 0..self.rows {
			for j in 0..self.cols {
				print!("{:.5} ",matrix_at!(i,j,self));
			}
			print!("\n");
		}
	}
}



#[cfg(test)]
mod tests {
	use super::*;

	#[test]
    fn new_dot_result_dim_test() {
		let ma = Matrix::new(2, 2);
		let mb = Matrix::new(2, 3);
		let dot_result = Matrix::new_dot_result(&ma,&mb);

		assert!(dot_result.rows == ma.rows && dot_result.cols == mb.cols && dot_result.values.len() == ma.rows*mb.cols)
	}


    #[test]
    fn dot_product_test() {
		let mut ma = Matrix::new(2, 2);
		let mut mb = Matrix::new(2, 3);
		let mut result = Matrix::new(ma.rows,mb.cols);
		let confirm =vec![4.0,0.0,2.0,2.0,0.0,1.0];

		matrix_at!(0,0,ma) = 2.0;
		matrix_at!(0,1,ma) = 0.0;
		matrix_at!(1,0,ma) = 1.0;
		matrix_at!(1,1,ma) = 0.0;


		matrix_at!(0,0,mb) = 2.0;
		matrix_at!(0,1,mb) = 0.0;
		matrix_at!(0,2,mb) = 1.0;
		matrix_at!(1,0,mb) = 0.0;
		matrix_at!(1,1,mb) = 0.0;
		matrix_at!(1,2,mb) = 4.0;

		ma.dot(&mut result, &mb);

		let precision : f64 = result.values.iter().zip(confirm.iter()).map(|(&a,&b)|(a-b).abs()).sum();
		assert!(precision==0.0);
	}

	#[test]
	#[should_panic]
	fn matrix_dot_wrong_dimension1(){
		let ma = Matrix::new(8, 4);
		let mb = Matrix::new(1, 7);
		let mut result = Matrix::new(10,10);

		ma.dot(&mut result, &mb);
	}

	#[test]
	#[should_panic]
	fn matrix_dot_wrong_dimension2(){
		let ma = Matrix::new(8, 4);
		let mb = Matrix::new(1, 7);
		let mut result = Matrix::new(8,6);

		ma.dot(&mut result, &mb);
	}


	#[test]
	#[should_panic]
	fn matrix_dot_zero_in_dimension(){
		let ma = Matrix::new(0, 0);
		let mb = Matrix::new(0, 7);
		let mut result = Matrix::new(0,7);

		ma.dot(&mut result, &mb);
	}


	#[test]
	#[should_panic]
	fn matrix_dot_zero_in_dimension2(){
		let ma = Matrix::new(4, 0);
		let mb = Matrix::new(0, 7);
		let mut result = Matrix::new(4,7);

		ma.dot(&mut result, &mb);
	}


	#[test]
	fn matrix_add_test(){
		let mut ma = Matrix::new(2, 2);
		let mut mb = Matrix::new(2, 2);
		let mut result = Matrix::new(2,2);
		let confirm =vec![3.0,9.0,3.0,7.0];


		matrix_at!(0,0,ma) = 2.0;
		matrix_at!(0,1,ma) = 8.0;
		matrix_at!(1,0,ma) = 1.0;
		matrix_at!(1,1,ma) = 4.0;

		matrix_at!(0,0,mb) = 1.0;
		matrix_at!(0,1,mb) = 1.0;
		matrix_at!(1,0,mb) = 2.0;
		matrix_at!(1,1,mb) = 3.0;

		ma.add(&mut result, &mb);

		let precision : f64 = result.values.iter().zip(confirm.iter()).map(|(&a,&b)|(a-b).abs()).sum();
		assert!(precision==0.0);
	}


	#[test]
	fn matrix_multiply(){
		let mut ma = Matrix::new(3, 3);
		let mut mb = Matrix::new(3, 3);
		let confirm = vec![
			7.0,
			14.0,
			21.0,
			32.0,
			40.0,
			48.0,
			21.0,
			24.0,
			27.0,
		];

		matrix_at!(0,0,ma) = 7.0;
		matrix_at!(0,1,ma) = 7.0;
		matrix_at!(0,2,ma) = 7.0;
		matrix_at!(1,0,ma) = 8.0;
		matrix_at!(1,1,ma) = 8.0;
		matrix_at!(1,2,ma) = 8.0;
		matrix_at!(2,0,ma) = 3.0;
		matrix_at!(2,1,ma) = 3.0;
		matrix_at!(2,2,ma) = 3.0;


		matrix_at!(0,0,mb) = 1.0;
		matrix_at!(0,1,mb) = 2.0;
		matrix_at!(0,2,mb) = 3.0;
		matrix_at!(1,0,mb) = 4.0;
		matrix_at!(1,1,mb) = 5.0;
		matrix_at!(1,2,mb) = 6.0;
		matrix_at!(2,0,mb) = 7.0;
		matrix_at!(2,1,mb) = 8.0;
		matrix_at!(2,2,mb) = 9.0;

		ma.multiply_by_mut(&mb);


		let precision : f64 = ma.values.iter().zip(confirm.iter()).map(|(&a,&b)|(a-b).abs()).sum();
		assert!(precision==0.0);
		
	}


	#[test]
	#[should_panic]
	fn matrix_multiply_wrong_dimension_1(){
		let mut ma = Matrix::new(2, 3);
		let mb = Matrix::new(3, 3);
		ma.multiply_by_mut(&mb);
	}

	#[test]
	#[should_panic]
	fn matrix_multiply_wrong_dimension_2(){
		let mut ma = Matrix::new(3, 2);
		let mb = Matrix::new(3, 3);
		ma.multiply_by_mut(&mb);
	}

	#[test]
	#[should_panic]
	fn matrix_multiply_wrong_dimension_3(){
		let mut ma = Matrix::new(3, 3);
		let mb = Matrix::new(2, 3);
		ma.multiply_by_mut(&mb);
	}

	#[test]
	#[should_panic]
	fn matrix_multiply_wrong_dimension_4(){
		let mut ma = Matrix::new(3, 3);
		let mb = Matrix::new(3, 2);
		ma.multiply_by_mut(&mb);
	}

}
