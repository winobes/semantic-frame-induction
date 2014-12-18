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
            if (i >= len) { // check if this is the source of the segfault
                printf("frame = %d\n", i);
                printf("dist = ");
                for (int j = 0; j < len; j++) {
                    printf("%f ", weights[j]);
                }
                printf("\nsum = %f, total = %f, r = %f\n", sum, total, r);
            }
            return i;
        }
    }
}


void gibbs(void *data_void, void *sample_void, 
        long N, long V, long W, int F, int T, 
        double alpha, double beta, int burnIn ) {
        
    srand(time(NULL));

    long *data = (long *) data_void;
    int *sample = (int *) sample_void;

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
        printf("Iteration %d of %d.\n", t, T);
        for (int i = 0; i < N; i++) {

            long v = data[M*i + VERB];
            long s = data[M*i + SUBJ];
            long o = data[M*i + OBJ];
            long c = data[M*i + COUNT];
            long f = data[M*i + FRAME];
            /*printf("here %d %lu %lu %lu %lu %lu\n", i, v,s,o,c,f);*/

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
            f = random_weighted(posterior, F);
            data[M*i + FRAME] = f;
            // record the sample
            if (t >= burnIn) {
                sample[F*i + f] += 1;
            }

            // modify counts to reflect new frame 
            frame_count[f] += c;
            frame_count_v[f][v] += c;
            frame_count_w[f][s] += c;
            frame_count_w[f][o] += c;
        }
    }

    printf("here\n");
    for (int f = 0; f < F; f++) {
        free(frame_count_v[f]);
        free(frame_count_w[f]);
    }

    free(frame_count);
    free(doc_count);
    free(frame_count_v);
    free(frame_count_w);
    free(posterior);
    printf("here\n");

}
