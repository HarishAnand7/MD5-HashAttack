#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return EXIT_FAILURE;
    }

    int n = std::atoi(argv[1]);

    // std::cout << n << " ";

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        std::cerr << "This program must be run with exactly 2 processes." << std::endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    float* send_buffer = new float[n];
    float* recv_buffer = new float[n];

    // Initialize send buffer with some data
    for (int i = 0; i < n; ++i) {
        send_buffer[i] = static_cast<float>(i);
    }

    if (rank == 0) {
        // Process 0 sends data and receives data
        double start_time = MPI_Wtime();

        MPI_Send(send_buffer, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_buffer, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double end_time = MPI_Wtime();

        // Send the timing result to process 1
        double elapsed_time = end_time - start_time;
        MPI_Send(&elapsed_time, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // Process 1 receives data and sends data
        double start_time = MPI_Wtime();

        MPI_Recv(recv_buffer, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_buffer, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

        double end_time = MPI_Wtime();

        // Receive the timing result from process 0
        double elapsed_time;
        MPI_Recv(&elapsed_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Print the total time taken by both processes
        std::cout  << (elapsed_time + end_time - start_time) * 1000.0 << std::endl;
    }

    delete[] send_buffer;
    delete[] recv_buffer;

    MPI_Finalize();

    return EXIT_SUCCESS;
}
