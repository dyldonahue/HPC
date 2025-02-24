// Dylan Donahue
// HPC Homework 2 Problem 3 - graph coloring, 02/12/2025

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>
#include <stdbool.h>

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

typedef struct Vertex Vertex;
// color = 0 if uncolored
typedef struct Vertex {
    int vertex_id;
    Vertex* next;
    bool is_colored;
    int color;  
    int degree;
} Vertex;

// define graph
typedef struct {
    int num_vertices;
    Vertex** adj_list;
} Graph;

// init new vertex
Vertex *create_vertex(int id){

    // check if vertex already exists
    Vertex *new_vertex = (Vertex*)malloc(sizeof(Vertex));
    new_vertex->vertex_id = id;
    new_vertex->color = 0;
    new_vertex->is_colored = false;
    new_vertex->next = NULL;
    new_vertex->degree = 0;
    return new_vertex;
}

int vertex_exists(Vertex *start, int id){
    Vertex* temp = start;
        while (temp) {
            if (temp->vertex_id == id) {
                return 1;
            }
            temp = temp->next;
        }
        return 0; // Not found
}

void new_edge(Graph* graph, int src, int dest) {
    Vertex* new_vertex = create_vertex(dest);
    new_vertex->next = graph->adj_list[src];
    graph->adj_list[src] = new_vertex;
    
   
    new_vertex = create_vertex(src);
    new_vertex->next = graph->adj_list[dest];
    graph->adj_list[dest] = new_vertex;
    graph->adj_list[dest]->degree++;
}

void print_graph(Graph *graph){
    printf("Vertex:  Adjacency List\n");
    for (int i = 0; i < graph->num_vertices; i++) {
        Vertex* temp = graph->adj_list[i];
        printf("%d --->", i);
        while (temp) {
            printf(" %d ->", temp->vertex_id);
            temp = temp->next;
        }
        printf(" NULL\n");  
    }
}

// function to randomly generate graph
Graph *create_graph(int num_vertices, int num_edges){
    Graph *graph = (Graph*) malloc(sizeof(Graph));
    graph->num_vertices = num_vertices;
    graph->adj_list = (Vertex**) malloc(num_vertices * sizeof(Vertex*));

    for (int i = 0; i < num_vertices; i++){
        graph->adj_list[i] = NULL;

    }

    // ensure connected by first creating spanning tree
    for (int i = 1; i < num_vertices; i++){
        new_edge(graph, i, rand() % i);
    }

    // add remaining edges
    for (int i = num_vertices - 1; i < num_edges; i++){
        int src_vertex_id = rand() % num_vertices;
        int dst_vertex_id = rand() % num_vertices;
        new_edge(graph, src_vertex_id, dst_vertex_id);
    }
    
    return graph;
}

//function to check if coloring is valid
bool is_valid_coloring(Graph* graph){
    for (int i = 0; i < graph->num_vertices; i++){
        Vertex* temp = graph->adj_list[i];
        Vertex* neighbor = temp->next;
        while (neighbor != NULL){
            int neighbor_id = neighbor->vertex_id;
            if (neighbor_id != i && graph->adj_list[neighbor_id]->color == graph->adj_list[i]->color){
                printf("Conflict: Vertex %d and Vertex %d have the same color\n", i, neighbor_id);
                return false;
            }
            neighbor = neighbor->next;
        }
    }
    return true;
}


int main(int argc, char *argv[]){

    if (argc != 4) {
        printf("Usage: %s num_threads, num_vertices, num_edges\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int num_vertices = atoi(argv[2]);
    long num_edges = atol(argv[3]);

    if (num_edges > ((long)num_vertices*((long)num_vertices-1)/2)){
        printf("max # of edges is v * v-1 / 2");
        return 1;
    }

    if (num_threads > num_vertices){
        printf("num_threads must be less than num_vertices");
        return 1;
    }

    if (num_edges < num_vertices - 1){
        printf("graph must be connectable (edges >= vertices - 1)");
        return 1;
    }

    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);

    srand(time(NULL));
    
    Graph* graph = create_graph(num_vertices, num_edges);

    print_graph(graph);

    // print degrees
     printf("Vertex:  Degree\n");
     for (int i = 0; i < num_vertices; i++) {
         printf("%d ---> %d\n", i, graph->adj_list[i]->degree);
     }

    double start = CLOCK(); 


    // approach to coloring:
    // 1. assign each vertex to a thread
    // 2. each thread colors its vertices
    //      2a. if a vertex is uncolored, color it with the lowest color not used by its neighbors
    // 3. Do a second pass sequentially to handle conflicts tat arose from simultaneous coloring data races
    int num_colors = 0;
    #pragma omp parallel for
    for (int i = 0; i < num_vertices; i++){

        // max number of colors would be the number of vertices
        bool used_colors[num_vertices + 1]; 

        //reset for each vertex
        for (int j = 0; j <= num_vertices; j++)
        used_colors[j] = false; 

        Vertex* neighbor = graph->adj_list[i];

        while (neighbor != NULL) {
            int neighbor_id = neighbor->vertex_id;
            if (graph->adj_list[neighbor_id]->is_colored) {
                used_colors[graph->adj_list[neighbor_id]->color] = true;
            }

            neighbor = neighbor->next;
        }

        int curr_color = 1;
        while (used_colors[curr_color]) {
            curr_color++;
        }

        graph->adj_list[i]->color = curr_color;
        graph->adj_list[i]->is_colored = true;
        if (curr_color > num_colors) {
            num_colors = curr_color;
        }

        //printf("Thread %d: Vertex %d colored with color %d\n", start, i, curr_color);
    }


    // second pass to handle conflicts
    int conflicts = 0;
    for (int i = 0; i < num_vertices; i++) {
        Vertex* neighbor = graph->adj_list[i];
        while (neighbor != NULL) {
            int neighbor_id = neighbor->vertex_id;
            if (neighbor_id != i && graph->adj_list[i]->color == graph->adj_list[neighbor_id]->color) {
                graph->adj_list[i]->color++;
                conflicts++;
                neighbor = graph->adj_list[i];
            }
            else neighbor = neighbor->next;
        }
    }


    double end = CLOCK();
    printf("--------------------------------\n");
    printf("Time spent (ms): %f\n", end - start);

    printf("Number of conflicts that needed to be resolved sequentially: %d out of %d vertices\n", conflicts, num_vertices);

    //print colored graph
    // printf("Vertex:  Color\n");
    // for (int i = 0; i < num_vertices; i++) {
    //     printf("%d ---> %d\n", i, graph->adj_list[i]->color);
    // }

    if (is_valid_coloring(graph)){
        printf("Valid coloring\n");
    } else {
        printf("Invalid coloring\n");
    }

    printf("Number of colors used: %d\n", num_colors);

    return 0;
}
    



   
    



