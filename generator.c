#include <stdio.h>
#include <stdlib.h>

float c3 = -2.;
float c2 = 4.;
float c1 = 3.;
float c0 = -5.;

float noise()
{
    float rnd = rand()/(float)RAND_MAX; //<0, 1> 
    return  .5*(rnd - 0.5); //<-.25, .25>
}

float poly(float x){

    return (c3*x*x*x + c2*x*x + c1*x +c0);
}

int main(int argc, char **argv)
{

    int N = atoi(argv[1]);
    
    FILE *f = fopen("input.txt","w");
    if(f == NULL)
        return -1;


    //fprintf(f,"%f %f %f %f\n",c0,c1,c2,c3);

    float x = -1.0;
    for(int i=0; i<100; i++){
        x += 0.03;
        fprintf(f, "%f %f\n", x, poly((x))+noise());
    }    

    fclose(f);

    return 0;
}
