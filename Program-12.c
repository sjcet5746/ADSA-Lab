#include <stdio.h>

#define MAX 100

int x[MAX];
int n, sum;
int w[MAX];
int count = 0;

void sumOfSubsets(int s, int k, int r) {
    int i;
    x[k] = 1;

    if (s + w[k] == sum) {
        printf("\nSubset %d: ", ++count);
        for (i = 0; i <= k; i++)
            if (x[i] == 1)
                printf("%d ", w[i]);
    } else if (s + w[k] + w[k + 1] <= sum)
        sumOfSubsets(s + w[k], k + 1, r - w[k]);

    if ((s + r - w[k] >= sum) && (s + w[k + 1] <= sum)) {
        x[k] = 0;
        sumOfSubsets(s, k + 1, r - w[k]);
    }
}

int main() {
    int i, j, temp, total = 0;

    printf("Enter number of elements: ");
    scanf("%d", &n);

    printf("Enter the elements:\n");
    for (i = 0; i < n; i++) {
        scanf("%d", &w[i]);
        total += w[i];
    }

    printf("Enter the required sum: ");
    scanf("%d", &sum);

    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (w[i] > w[j]) {
                temp = w[i];
                w[i] = w[j];
                w[j] = temp;
            }
        }
    }

    printf("\nSubsets with sum equal to %d are:\n", sum);
    sumOfSubsets(0, 0, total);

    if (count == 0)
        printf("\nNo such subsets found\n");

    return 0;
}
