// Basic merge sort implementation from https://www.geeksforgeeks.org/c-program-for-merge-sort/
// I have added the pthread functionality, including:
//      -  splitting the merge_sort into a "thread" call and a recrusive function
//      -  merging the subarrays dynamically based on the number of threads
//      -  added a timer to measure the time spent, and random number generation for the array
//     - function to verify the array is sorted

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 10000
int num_threads;
int n_per_thread;

// list to be sorted
int a[N];

// Merges two subarrays of arr[].
// First subarray is arr[left..mid]
// Second subarray is arr[mid+1..right]
void merge(int arr[], int left, int mid, int right) {

    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    int leftArr[n1], rightArr[n2];

    // Copy data to temporary arrays
    for (i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into a[left..right]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        }
        else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArr[], if any
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightArr[], if any
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void merge_sort(int left, int right) {
    if (left < right) {
        // Calculate the midpoint
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        merge_sort(left, mid);
        merge_sort(mid + 1, right);

        // Merge the sorted halves

        merge(a, left, mid, right);
    
    }
}

// The subarray to be sorted is in the index range [left-right]
void *merge_sort_thread(void *arg) {

    // slice based on the number of threads
    int thread_num = *(int*)arg;
    free(arg);

    int left = thread_num * n_per_thread;
    int right = (thread_num + 1) * n_per_thread - 1;
  

     // add remainder to last thread
    if (thread_num == num_threads - 1) {
        right += N % num_threads;
    }

    if (left < right) {
      
        merge_sort(left, right);
    }

    return NULL;
}

// the # of subarrays depends on (and is equal to) the number of threads, we need to dynamically merge based on this number
void merge_subarrays(int arr[], int num_sections, int offset) {

    int left = 0;
    int right = 0;
    int mid = 0;
    for (int i = 0; i < num_sections; i += 2) {
         left = i * n_per_thread * offset;
         right = (i + 2) * n_per_thread * offset - 1;

        // Ensure the boundaries are valid
        if (right >= (N)){
            right = (N) - 1;
        }

        mid = left + n_per_thread * offset - 1;

        merge(arr, left, mid, right);
    }

    // if there is an odd number of sections merge the last into the second to last to even out

    if (num_sections % 2 != 0) {
        
        mid = right;
        right = N - 1;

        merge(arr, left, mid, right);
    }

    // Stop recursion if there are no more sections to merge
    if (num_sections >= 2) {
            merge_subarrays(arr, num_sections / 2, offset * 2);
    }
}


void verify_sorted() {
    for (int i = 0; i < N - 1; i++) {
        if (a[i] > a[i + 1]) {
            printf("Array not sorted\n");
            printf("a[%d] = %d, a[%d] = %d\n", i, a[i], i + 1, a[i + 1]);
            return;
        }   
    }
    printf("Array sorted\n");
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: ./problem2 <number of threads>\n");
        return -1;
    }

    num_threads = atoi(argv[1]);
    n_per_thread = N / num_threads;

    if (N < num_threads) {
        printf("Number of threads must be less than the array size\n");
        return -1;
    }

    srand(time(0));
   
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 1000;
    }

    clock_t start = clock();

    // Create threads
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        int* arg = malloc(sizeof(int));
        *arg = i;
        int ret = pthread_create(&threads[i], NULL, merge_sort_thread, arg);
        if (ret) {
            printf("Error creating thread %d\n", i);
            exit(-1);
        }
    }

    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    merge_subarrays(a, num_threads, 1);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);

    verify_sorted();
    return 0;
}