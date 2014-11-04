#include <stdio.h>
#include <stdlib.h>

float c3 = -2.;
float c2 = 4.;
float c1 = 3.;
float c0 = -5.;

float noise()
{
    float rnd = rand() / (float)RAND_MAX; //<0, 1> 
    //return 1.4 * rnd - 0.7; //<-.7, .7> 20% noise
    return 2.8 * rnd -1.4; //40%
}

float poly(float x)
{
    return c3 * x * x * x + c2 * x * x + c1 * x +c0;
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <N>\n", argv[0]);
		exit(1);
	}

    int N = atoi(argv[1]);
    
    FILE *f = fopen("input.txt","w");
    if (!f) return -1;

    float x = -2.0;
    float interval_width = 5;

    float increment = interval_width / N;
    for (int i = 0; i < N; i++)
    {
        x += increment;
        fprintf(f, "%f %f\n", x, poly((x)) + noise());
    }    

    fclose(f);

    return 0;
}

