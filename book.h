#ifndef __BOOK_H__
#define __BOOK_H__

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
void HandleError( cudaError_t err,
                         const char *file,
                         int line );
#define HANDLE_CUBLAS_ERROR( err ) (HandleCublasError(err,__FILE__,__LINE__ ))
void HandleCublasError( int err,
                         const char *file,
                         int line );

template< typename T >
void swap( T& a, T& b );

void* big_random_block( int size );
int* big_random_block_int( int size );
__device__ double atomicAdd(double* address, double val);
__device__ unsigned char value( float n1, float n2, int hue );
__global__ void float_to_color( unsigned char *optr,
                              const float *outSrc );
__global__ void float_to_color( uchar4 *optr,
                              const float *outSrc );

#if _WIN32
    //Windows threads.
    #include <windows.h>

    typedef HANDLE CUTThread;
    typedef unsigned (WINAPI *CUT_THREADROUTINE)(void *);

    #define CUT_THREADPROC unsigned WINAPI
    #define  CUT_THREADEND return 0

#else
    //POSIX threads.
    #include <pthread.h>

    typedef pthread_t CUTThread;
    typedef void *(*CUT_THREADROUTINE)(void *);

    #define CUT_THREADPROC void
    #define  CUT_THREADEND
#endif

//Create thread.
CUTThread start_thread( CUT_THREADROUTINE, void *data );

//Wait for thread to finish.
void end_thread( CUTThread thread );

//Destroy thread.
void destroy_thread( CUTThread thread );

//Wait for multiple threads.
void wait_for_threads( const CUTThread *threads, int num );

#endif  // __BOOK_H__
