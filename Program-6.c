#include <stdio.h>
#include <stdlib.h>

// Structure to represent an item
struct Item {
    int value;
    int weight;
};

// Function to compare items based on their value-to-weight ratio
int compare(const void *a, const void *b) {
    double ratio_a = ((double)((struct Item *)a)->value) / ((struct Item *)a)->weight;
    double ratio_b = ((double)((struct Item *)b)->value) / ((struct Item *)b)->weight;
    return (ratio_b - ratio_a) > 0 ? 1 : -1;
}

// Function to solve the knapsack problem using the greedy method
void knapsackGreedy(struct Item items[], int n, int capacity) {
    qsort(items, n, sizeof(items[0]), compare);

    int currentWeight = 0;
    double totalValue = 0.0;

    for (int i = 0; i < n; i++) {
        if (currentWeight + items[i].weight <= capacity) {
            currentWeight += items[i].weight;
            totalValue += items[i].value;
        } else {
            double remainingCapacity = capacity - currentWeight;
            totalValue += items[i].value * (remainingCapacity / items[i].weight);
            break;
        }
    }

    printf("Maximum value obtained from the knapsack: %.2f\n", totalValue);
}

int main() {
    int capacity = 50;
    struct Item items[] = {{60, 10}, {100, 20}, {120, 30}};

    int n = sizeof(items) / sizeof(items[0]);

    knapsackGreedy(items, n, capacity);

    return 0;
}
