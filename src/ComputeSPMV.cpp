//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ExchangeHalo.hpp"

//DEBUG
#include <iomanip>

/*!
 Routine to compute sparse matrix vector product y = Ax where:
 Precondition: First call exchange_externals to get off-processor values of x

 This routine calls the reference SpMV implementation by default, but
 can be replaced by a custom, optimized routine suited for
 the target system.

 @param[in]  A the known system matrix
 @param[in]  x the known vector
 @param[out] y the On exit contains the result: Ax.

 @return returns 0 upon success and non-zero otherwise

 @see ComputeSPMV_ref
 */
int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y,
		bool use_non_blocking_halo_exchange) {

	assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
	assert(y.localLength >= A.localNumberOfRows);

	// Begin halo exchange was called before entering if use_non_blocking_halo_exchange

	const double *const xv = x.values;
	double *const yv = y.values;
	const local_int_t nrow = A.localNumberOfRows;

	// init result
	for (local_int_t i = 0; i < nrow; ++i) {
		yv[i] = 0;
	}

	// local part
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	for (local_int_t i = 0; i < A.local_local_NumberOfColumns; i++) {

		const double *const cur_vals = A.matrixValuesCSC[i];
		const local_int_t *const cur_inds = A.mtxCSCIndL[i];
		const int cur_nnz = A.nonzerosInCol[i];

		for (int j = 0; j < cur_nnz; j++) {
			local_int_t row = cur_inds[j];
#pragma omp atomic
			yv[row] += cur_vals[j] * xv[i];
		}
	}

	if (use_non_blocking_halo_exchange) {
		EndExchangeHaloRecv(A, x);
	}

	// non-local part
	// local part
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	for (local_int_t i = A.local_local_NumberOfColumns;
			i < A.localNumberOfColumns; i++) {

		const double *const cur_vals = A.matrixValuesCSC[i];
		const local_int_t *const cur_inds = A.mtxCSCIndL[i];
		const int cur_nnz = A.nonzerosInCol[i];

		for (int j = 0; j < cur_nnz; j++) {
			local_int_t row = cur_inds[j];
#pragma omp atomic
			yv[row] += cur_vals[j] * xv[i];
		}
	}

	// we need to make shure all sends are completed as well
	// as Send has started before entering this func, the recvs will complete and no deadlock will be present
	// during send, the values to send where collected into a seperate buffer, so no isuses with override
	if (use_non_blocking_halo_exchange) {
		EndExchangeHaloSend(A, x);
	}

	/*
	 //For DEBUGGING
	 Vector yy;
	 InitializeVector(yy, A.localNumberOfRows);
	 ComputeSPMV_ref(A, x, yy);
	 int rank;
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	 //assert matrix is same
	 int differences=0;

	 for (int row = 0; row < A.localNumberOfRows; ++row) {
	 for (int j = 0; j < A.nonzerosInRow[row]; ++j) {
	 int col= A.mtxIndL[row][j];
	 double valA=A.matrixValues[row][j];
	 int i=0;
	 for (i = 0; i < A.nonzerosInCol[col]; ++i) {
	 if (row==A.mtxCSCIndL[col][i])
	 break;
	 }
	 double valB=A.matrixValuesCSC[col][i];
	 if (valA!=valB){
	 //std::cout << "Cmp Val " << row<<","<<col<< "("<<valA<<","<<valB<<")\n";
	 differences++;
	 }
	 assert(valA==valB);
	 }
	 }

	 if (differences!=0){
	 std::cout << "Matrix differences!!!\n";
	 }

	 int vec_differences=0;

	 //std::cout << "My, ref\n";

	 for (int i = 0; i < nrow; ++i) {
	 //std::cout << std::setw(10) << yv[i] << "," <<  std::setw(10) << yy.values[i]<<"\n";
	 if(yv[i] !=yy.values[i]){
	 differences++;
	 }
	 }

	 assert(vec_differences==0);
	 if (vec_differences!=0){
	 std::cout << "different Result!!!\n";
	 }

	 */

	return 0;
}
