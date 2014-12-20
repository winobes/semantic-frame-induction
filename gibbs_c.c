#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    VERB,
    SUBJ,
    OBJ,
    COUNT,
    FRAME,
    M 
};

long random_weighted(double *weights, int len) {
    double r = (double)rand() / (double)RAND_MAX;
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += weights[i];
    }
    double total = 0;
    for (int i = 0; i < len; i++) {
        total += weights[i] / sum; 
        if (r < total) {
            return i;
        }
    }
}


void gibbs(void *data_void, void *samples_void, 
        long N, long V, long W, int F, int T, 
        double alpha, double beta, int burnIn ) {
        
    srand(time(NULL));

    long *data = (long *) data_void;
    int *samples = (int *) samples_void;

    long *frame_count = calloc(F, sizeof(long));
    long *doc_count = calloc(V, sizeof(long));
    long **frame_count_v = calloc(F, sizeof(long*));
    long **frame_count_w = calloc(F, sizeof(long*));
    double *posterior = calloc(F, sizeof(double));

    for (int f = 0; f < F; f ++) {
        frame_count_v[f] = calloc(V, sizeof(long));
        frame_count_w[f] = calloc(W, sizeof(long));
    }

    for (int i = 0; i < N; i++) {
        long v = data[M*i + VERB];
        long s = data[M*i + SUBJ];
        long o = data[M*i + OBJ];
        long c = data[M*i + COUNT];
        long f = data[M*i + FRAME];
        frame_count[f] += c;
        doc_count[v] += c;
        frame_count_v[f][v] += c;
        frame_count_w[f][s] += c;
        frame_count_w[f][o] += c;
    }

    for (int t = 0; t < T; t++) {
        int changes = 0;
        printf("Iteration %d of %d. ", t+1, T);
        for (int i = 0; i < N; i++) {

            long v = data[M*i + VERB];
            long s = data[M*i + SUBJ];
            long o = data[M*i + OBJ];
            long c = data[M*i + COUNT];
            long f = data[M*i + FRAME];

            // modify counts to exclude data point i
            frame_count[f] -= c;
            frame_count_v[f][v] -= c;
            frame_count_w[f][s] -= c;
            frame_count_w[f][o] -= c;

            // calculate the posterior for frames on this data point
            for (int f = 0; f < F; f++) {
                double v_term  = ( (beta + frame_count_v[f][v]) 
                                 / (V + frame_count[f]) );
                double so_term = ( (2*beta + frame_count_w[f][s] + frame_count_w[f][o])
                                 / (W + frame_count[f]) );
                double f_term  = ( (alpha + frame_count_v[f][v])
                                 / (F*alpha + c) );
                posterior[f] = v_term * so_term * f_term;
            }

            // assign the new frame randomly
            int f_new = random_weighted(posterior, F);
            if (f != f_new) changes +=1;
            if (f_new >= F) {
               printf("Error: frame out of range.\n");
               return;
            }
            data[M*i + FRAME] = f_new;
            // record the sample
            if (t >= burnIn) {
                samples[F*i + f_new] += 1;
            }

            // modify counts to reflect new frame 
            frame_count[f_new] += c;
            frame_count_v[f_new][v] += c;
            frame_count_w[f_new][s] += c;
            frame_count_w[f_new][o] += c;
        }
        printf("Had %d of %lu VSOs change frame.\n", changes, N);
    }

    for (int f = 0; f < F; f++) {
        free(frame_count_v[f]);
        free(frame_count_w[f]);
    }

    free(frame_count);
    free(doc_count);
    free(frame_count_v);
    free(frame_count_w);
    free(posterior);

}
