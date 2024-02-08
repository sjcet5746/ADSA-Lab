#include <stdio.h>
#include <stdlib.h>

// Structure to represent a job
struct Job {
    char id;
    int deadline;
    int profit;
};

// Function to compare jobs based on their profits
int compare(const void* a, const void* b) {
    return ((struct Job*)b)->profit - ((struct Job*)a)->profit;
}

// Function to find the maximum deadline
int findMaxDeadline(struct Job arr[], int n) {
    int max_deadline = -1;
    for (int i = 0; i < n; i++) {
        if (arr[i].deadline > max_deadline)
            max_deadline = arr[i].deadline;
    }
    return max_deadline;
}

// Function to schedule jobs to maximize profit
void jobSequencing(struct Job arr[], int n) {
    qsort(arr, n, sizeof(arr[0]), compare);

    int max_deadline = findMaxDeadline(arr, n);
    char result[max_deadline];
    int slot[max_deadline];

    for (int i = 0; i < max_deadline; i++)
        slot[i] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = arr[i].deadline - 1; j >= 0; j--) {
            if (slot[j] == -1) {
                slot[j] = i;
                result[j] = arr[i].id;
                break;
            }
        }
    }

    printf("The sequence of jobs: ");
    for (int i = 0; i < max_deadline; i++) {
        if (slot[i] != -1)
            printf("%c ", result[i]);
    }
}

int main() {
    struct Job arr[] = {
        {'a', 2, 100},
        {'b', 1, 19},
        {'c', 2, 27},
        {'d', 1, 25},
        {'e', 3, 15}
    };

    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Following is the maximum profit sequence of jobs:\n");
    jobSequencing(arr, n);

    return 0;
}
