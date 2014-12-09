#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//function sample interval (X_MIN, X_MIN+X_RANGE)
#define X_MIN -2.0
#define X_RANGE 5

//amplitude of coefficients \in (-COEFF, +COEFF)
#define COEFF_MAX 500

//simulate noise by sine [f(x)*sin(x)] or add random noise [f(x)+noise]
#define SINE
#undef NOISE

float noise()
{
    float rnd = rand() / (float)RAND_MAX - 0.5; //<-0.5, 0.5> 
    //return 1.4 * rnd - 0.7; //<-.7, .7> 20% noise
    return COEFF_MAX*rnd; //50%
}

float poly(float x, float *c, int D)
{
    float sum = 0;
    for(int i=0; i<=D; i++){
        sum += c[i]*pow(x,i);
    }

    return sum;
}


int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("Usage: %s <POLY_ORDER> <SAMPLE_POINS_CNT>\n", argv[0]);
		exit(1);
	}

    int D = atoi(argv[1]);
    int N = atoi(argv[2]);
    
    FILE *f = fopen("input_file.txt","w");
    if (!f) return -1;

    float x = X_MIN;
    float interval_width = X_RANGE;

    //init random polynomial coeffs
    struct timeval time; 
    gettimeofday(&time,NULL);

    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
//    srand(time(NULL)); // does not work if launched multiple times within same second


    float *c = malloc((D+1)*sizeof(float));
    if (c == NULL)
        exit(1);

    //init coeffs
    for(int i = 0; i <= D; i++){
        c[i] = (2*(rand()/(float)RAND_MAX) - 1) * COEFF_MAX;
        printf("c%d=%f\n",i,c[i]);
    }

    float increment = interval_width / N;
    for (int i = 0; i < N; i++)
    {
        x += increment;
#ifdef NOISE
        fprintf(f, "%f %f\n", x, poly(x, c, D) + noise());
#endif

#ifdef SINE
        fprintf(f, "%f %lf\n", x, poly(x, c, D)*sin(x));
#endif
    }    

    fclose(f);
    free(c);

    return 0;
}

