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

float poly(float x, float c3, float c2, float c1, float c0)
{
    return c3 * x * x * x + c2 * x * x + c1 * x + c0;
}


int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <SAMPLE_POINS>\n", argv[0]);
		exit(1);
	}

    int N = atoi(argv[1]);
    
    FILE *f = fopen("input_file.txt","w");
    if (!f) return -1;

    float x = X_MIN;
    float interval_width = X_RANGE;

    //init random polynomial coeffs
    struct timeval time; 
    gettimeofday(&time,NULL);

    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
//    srand(time(NULL)); // does not work if launched multiple times within same second


    float c3 = (2*(rand()/(float)RAND_MAX) - 1) * COEFF_MAX;
    float c2 = (2*(rand()/(float)RAND_MAX) - 1) * COEFF_MAX;
    float c1 = (2*(rand()/(float)RAND_MAX) - 1) * COEFF_MAX;
    float c0 = (2*(rand()/(float)RAND_MAX) - 1) * COEFF_MAX;
    printf("c0=%f, c1=%f, c2=%f, c3=%f\n",c0,c1,c2,c3);

    float increment = interval_width / N;
    for (int i = 0; i < N; i++)
    {
        x += increment;
#ifdef NOISE
        fprintf(f, "%f %f\n", x, poly(x, c3, c2, c1, c0) + noise());
#endif

#ifdef SINE
        fprintf(f, "%f %lf\n", x, poly(x, c3, c2, c1, c0)*sin(x));
#endif
    }    

    fclose(f);

    return 0;
}

