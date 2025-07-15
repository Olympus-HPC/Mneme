#ifndef HEAP_H
#define HEAP_H


#define LCHILD(x) 2 * x + 1
#define RCHILD(x) 2 * x + 2
#define PARENT(x) x / 2

#include "findex.h"

/* node struct */
typedef struct node {
    float data ;
    findex_t coord;
} node ;


/* min heap */
typedef struct minHeap {
    int size ;
    node *elem ;
} minHeap ;


minHeap initMinHeap(int size);
void swap(node *n1, node *n2);
void printNode(node n);
void heapify(minHeap *hp, int i);
void buildMinHeap(minHeap *hp, int *arr, int size);
void insertNode(minHeap *hp, float data, findex_t frag);
void deleteNode(minHeap *hp);
node popRoot(minHeap *hp);
int getMaxNode(minHeap *hp, int i);
void deleteMinHeap(minHeap *hp);
void inorderTraversal(minHeap *hp, int i);
void preorderTraversal(minHeap *hp, int i);
void postorderTraversal(minHeap *hp, int i);
void levelorderTraversal(minHeap *hp);

#endif
