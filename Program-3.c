#include <stdio.h>
#include <stdlib.h>

// Structure of a node in the Splay tree
struct node {
    int key;
    struct node *left, *right;
};

// Function to create a new node
struct node *newNode(int key) {
    struct node *temp = (struct node *)malloc(sizeof(struct node));
    temp->key = key;
    temp->left = temp->right = NULL;
    return temp;
}

// Function to perform single left rotation
struct node *leftRotate(struct node *x) {
    struct node *y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

// Function to perform single right rotation
struct node *rightRotate(struct node *x) {
    struct node *y = x->left;
    x->left = y->right;
    y->right = x;
    return y;
}

// Function to splay a given key in the tree rooted with root
struct node *splay(struct node *root, int key) {
    // Base cases: root is NULL or key is present at root
    if (root == NULL || root->key == key)
        return root;

    // Key lies in the left subtree
    if (root->key > key) {
        // Key is not in tree, return root
        if (root->left == NULL)
            return root;

        // Zig-Zig (Left Left)
        if (root->left->key > key) {
            // First recursively bring the key as root of left-left
            root->left->left = splay(root->left->left, key);

            // Do first rotation for root, second rotation is done after else
            root = rightRotate(root);
        } else if (root->left->key < key) { // Zig-Zag (Left Right)
            // First recursively bring the key as root of left-right
            root->left->right = splay(root->left->right, key);

            // Do first rotation for root->left
            if (root->left->right != NULL)
                root->left = leftRotate(root->left);
        }

        // Do second rotation for root
        return (root->left == NULL) ? root : rightRotate(root);
    } else { // Key lies in right subtree
        // Key is not in tree, return root
        if (root->right == NULL)
            return root;

        // Zig-Zag (Right Left)
        if (root->right->key > key) {
            // Bring the key as root of right-left
            root->right->left = splay(root->right->left, key);

            // Do first rotation for root->right
            if (root->right->left != NULL)
                root->right = rightRotate(root->right);
        } else if (root->right->key < key) { // Zag-Zag (Right Right)
            // Bring the key as root of right-right and do first rotation
            root->right->right = splay(root->right->right, key);
            root = leftRotate(root);
        }

        // Do second rotation for root
        return (root->right == NULL) ? root : leftRotate(root);
    }
}

// Function to insert a new key k in Splay Tree rooted with root
struct node *insert(struct node *root, int key) {
    // If tree is empty, return a new node
    if (root == NULL) return newNode(key);

    // Bring the closest leaf node to root
    root = splay(root, key);

    // If key is already present, return root
    if (root->key == key) return root;

    // Otherwise, allocate memory for new node
    struct node *newnode = newNode(key);

    // If root's key is greater, make root as right child of newnode
    // and copy the left child of root to newnode
    if (root->key > key) {
        newnode->right = root;
        newnode->left = root->left;
        root->left = NULL;
    }
    // If root's key is smaller, make root as left child of newnode
    // and copy the right child of root to newnode
    else {
        newnode->left = root;
        newnode->right = root->right;
        root->right = NULL;
    }

    // Finally, newnode becomes the root
    return newnode;
}

// A utility function to print preorder traversal of the tree.
// The function also prints height of every node
void preOrder(struct node *root) {
    if (root != NULL) {
        printf("%d ", root->key);
        preOrder(root->left);
        preOrder(root->right);
    }
}

int main() {
    struct node *root = NULL;

    root = insert(root, 100);
    root = insert(root, 50);
    root = insert(root, 200);
    root = insert(root, 40);
    root = insert(root, 30);
    root = insert(root, 20);

    printf("Preorder traversal of the modified Splay tree is: \n");
    preOrder(root);

    return 0;
}
