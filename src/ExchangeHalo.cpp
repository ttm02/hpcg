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
 @file ExchangeHalo.cpp

 HPCG routine
 */

// Compile this routine only if running with MPI
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>

void BeginExchangeHalo(const SparseMatrix &A, Vector &x) {
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

void EndExchangeHalo(const SparseMatrix &A, Vector &x) {
	MPI_Waitall(A.numberOfSendNeighbors * 2, A.halo_requests,
			MPI_STATUSES_IGNORE);

	return;
}

/*!
 Communicates data that is at the border of the part of the domain assigned to this processor.

 @param[in]    A The known system matrix
 @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix &A, Vector &x) {


	std::cout << "Halo exchange\n";
	// only this Vec can be used for halo exhange
	assert(A.halo_exchange_vector == &x);

	BeginExchangeHalo(A, x);
	EndExchangeHalo(A, x);

}

#endif
// ifndef HPCG_NO_MPI
