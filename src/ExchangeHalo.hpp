
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

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

void ExchangeHalo(const SparseMatrix & A, Vector & x);

// we need inlining, so that our compiler analysis can work properly
// additionally these functions are quite small, so inlining is a good idear either way
//TODO veryfi in godbolt: additionally this should eliminate the unused parameter if compiled without assertion
inline void BeginExchangeHalo(const SparseMatrix &A, Vector &x) {

	assert(A.halo_exchange_vector==&x);
	double *sendBuffer = A.sendBuffer;
	local_int_t totalToBeSent = A.totalToBeSent;
	local_int_t *elementsToSend = A.elementsToSend;

	double *const xv = x.values;
	//
	// Fill up send buffer
	//

	// TODO: Thread this loop
	for (local_int_t i = 0; i < totalToBeSent; i++)
		sendBuffer[i] = xv[elementsToSend[i]];

	// start all MPI communication
	MPI_Startall(A.numberOfSendNeighbors * 2, A.halo_requests);

	return;

}

inline void EndExchangeHalo(const SparseMatrix &A, Vector &x) {
	MPI_Waitall(A.numberOfSendNeighbors * 2, A.halo_requests,
			MPI_STATUSES_IGNORE);

	return;
}

#endif // EXCHANGEHALO_HPP
