//Texas A&M University
//CSCE 755: Parallel Computing
//Author: Vishakha Thakurdwarkar
//UIN: 539005713

/*
Major Project: Computing hyperparameters used in GPR model
Compute predictions at test points using training points 
and observed values at those training poionts as input to the model 
with hyperparameters t and l. 
*/

//includes
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<time.h>
#include<omp.h>

//Function to obtain z matrix value for ftest prediction
float* get_zMatrix(int m, float* t, float*z, float* f, float** L, float** U)
{
	int i,j;
	int n = m*m;
	
	//Solving Kz=f
	
    #pragma omp parallel for collapse(1) default(shared) private(i,j)
    for(i = 0; i < n; i++)
    {
        t[i] = f[i];
		
        for(j = 0; j < i; j++)
        {
            t[i] -= L[i][j]*t[j];
        }
    }
	
    #pragma omp parallel for collapse(1) default(shared) private(i,j)
    for(i = n-1; i >= 0; i--)
    {
        z[i] = t[i];
    
		for(j = i+1; j < n; j++)
        {
            z[i] -= U[i][j]*z[j];
        }
		
        z[i] /= U[i][i];
    }
	
	return z;
}

//Function to calculate predicted value of ftest using Gaussian Process Regression 
float gpr(int m, float x, float y, float* f, float** K, float* hyp_param)
{
	int i,j,k;
    float ftest = 0;
    float sum;

    int n = m*m;
    
	//Initializing the required matrices and allocating memory
    float** L = (float**)calloc(n,sizeof(float*));
    for (i=0; i<n; i++)
         L[i] = (float *)calloc(n,sizeof(float));
    float** U = (float**)calloc(n,sizeof(float*));
    for (i=0; i<n; i++)
         U[i] = (float *)calloc(n,sizeof(float));
    float* z = (float*)calloc(n,sizeof(float));
    float* t = (float*)calloc(n,sizeof(float));
    float* kt = (float*)calloc(n,sizeof(float));

    //Computing LU factorization 
    for (i = 0; i < n; i++)
    {
        // Upper Triangular Matrix
        for ( j = i; j < n; j++)
        {
            sum=0;
			
            #pragma omp parallel for default(shared) private(k) reduction(+:sum)
            for (k = 0; k < i; k++) 
                sum += (L[i][k] * U[k][j]);
			
            U[i][j] = K[i][j] - sum;
        } 
		
        // Lower Triangular Matrix
        for (j = i; j < n; j++)
        { 
            if (i == j) 
                L[i][i] = 1;
			
            else
            {
                sum = 0; 
				
                #pragma omp parallel for default(shared) private(k) reduction(+:sum)
                for (k = 0; k < i; k++)
                    sum += (L[j][k] * U[k][i]);
				
                L[j][i] = (K[j][i] - sum) / U[i][i];
            }
        }
    }
    
	z = get_zMatrix(m, t, z, f, L, U);

    //Create k matrix
    for(i = 0; i < n; i++)
    {
        kt[i] = (1/sqrt(2*M_PI))*exp(-1*((pow((((int)(i/m)/(float)(m+1))-x),2)/(2*pow(hyp_param[0],2)))+(pow((((i%m)/(float)(m+1))-y),2)/(2*pow(hyp_param[1],2)))));
    }

    //Compute ftest value
    for(i = 0; i < n; i++)
    {
        ftest += kt[i]*z[i];
    }

	//Freeing the memory allocated for all the matrices
    for (i = 0; i < n; i++)
        free(L[i]);
    free(L);
    for (i = 0; i < n; i++)
        free(U[i]);
    free(U);
    free(z);
    free(t);
    free(kt);   

    return ftest;
}

//Function to create Matrix k
float** create_kMatrix(float* hyp_param, int m, float** K)
{
	int n = m*m;
	//Create K matrix
	
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			K[i][j] = (1/sqrt(2*M_PI))*exp(-1*((pow((((int)(i/m)/(float)(m+1))-((int)(j/m)/(float)(m+1))),2)/(2*pow(hyp_param[0],2)))+(pow((((i%m)/(float)(m+1))-((j%m)/(float)(m+1))),2)/(2*pow(hyp_param[1],2)))));
			
			if(i==j)
				K[i][j] += 0.01;
		}
	}
	
	return K;
}

int main(int argc, char* argv[])
{
	int i,j,k,n,m,a,b, dataset;
    float x,y,ftest = 0;
    float hyp_param[2],hyp_param_min[2]={__FLT_MAX__,__FLT_MAX__};
    float hyp_param_range[10];
	float hyp_param_mni[2];
    float mse, mse_min, mse_list[100];
    
	//Get the grid size from command line
    m = (int)atoi(argv[1]);
	dataset = (int)atoi(argv[2]);
    srand(time(0));

    for(i = 0;i < 10;i++)
        hyp_param_range[i]= 0.25+(0.025*i);

    n = m*m;
    int id[n];
	
    float* f = (float*)calloc(n,sizeof(float));
    float** K = (float**)calloc(n,sizeof(float*));
    for (i = 0; i < n; i++)
         K[i] = (float *)calloc(n,sizeof(float));

    hyp_param[0]=2/(float)m;
    hyp_param[1]=2/(float)m;
	
    //Create f matrix
    for(i = 0; i < n; i++)
    {
		if(dataset == 1)
		{
			f[i] = (1/sqrt(2*M_PI))*exp(-1*((pow((((int)(i/m)/(float)(m+1))-0.25),2)/(2*pow(hyp_param[0],2)))+(pow((((i%m)/(float)(m+1))-0.25),2)/(2*pow(hyp_param[1],2)))));
			
			f[i] += (((float)rand()/(float)RAND_MAX) - 0.5)*0.02;
			f[i] += (((int)(i/m)/(m+1))*0.2) + (((i%m)/(m+1))*0.1);
		}
		
		//Applying the code to new dataset
		else if(dataset == 2)
		{
			f[i] = 1 - exp((pow((((int)(i/m)/(m+1))-0.5),2)+pow((((i%m)/(m+1))-0.5),2))) + ((((float)rand()/(float)RAND_MAX)*0.1)-0.05);
		}
		
		else
		{
			printf("Please enter correct dataset format");
			break;
		}
    }

    //Generating the indices for training and testing
    for(i = 0;i < n;i++)
    {
        id[i] = i;
    }
	
	//Assigning values to the indices
    for (i = 0; i < n - 1; i++) 
    {
        j = i + rand() / (RAND_MAX / (n - i) + 1);
        k = id[j];
		
        id[j] = id[i];
        id[i] = k;
    }

    //Finding the hyperparameters
    #pragma omp parallel for collapse(2) default(shared) private(a,b,i,j,ftest,x,y,mse,hyp_param)
    for(a = 0;a < 10; a++)
    {
        for(b = 0;b < 10; b++)
        {
            hyp_param[0] = hyp_param_range[a];
            hyp_param[1] = hyp_param_range[b];
			
			K = create_kMatrix(hyp_param, m, K);
			
            mse = 0;
            for (i = 0; i < 0.9*n; i++)
            {
                x = (int)(id[i]/m)/(float)(m+1);
                y = (id[i]%m)/(float)(m+1);
				
                ftest = gpr(m,x,y,f,K,hyp_param);
                mse  += pow(f[id[i]]-ftest,2);
            }
			
            mse /= (int)(0.9*n);
            mse_list[a*10 + b] = mse;
			
            printf("MSE = %f L1 = %f L2 = %f\n",mse,hyp_param[0],hyp_param[1]);
        }
    }

	//Calculating minimun MSE and its respective hyper parameter values
    mse_min = mse_list[0];
	hyp_param_mni[0] = hyp_param[0];
	hyp_param_mni[1] = hyp_param[1];
	
    for(i = 1;i < 100;i++)
    {
        if(mse_list[i] < mse_min)
        {
            mse_min = mse_list[i];
			
            hyp_param_min[0] = hyp_param_range[(int)(i/10)];
            hyp_param_min[1] = hyp_param_range[i%10];
        }
    }
    
    printf("MSE min = %f L1 = %f L2 = %f\n",mse_min,hyp_param_mni[0],hyp_param_mni[1]);


    //Testing 
	
    //Create K matrix for test data
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            K[i][j] = (1/sqrt(2*M_PI))*exp(-1*((pow((((int)(i/m)/(float)(m+1))-((int)(j/m)/(float)(m+1))),2)/(2*pow(hyp_param_min[0],2)))+(pow((((i%m)/(float)(m+1))-((j%m)/(float)(m+1))),2)/(2*pow(hyp_param_min[1],2)))));
            
			if(i==j)
                K[i][j] += 0.01;
        }
    }
	
    mse = 0;
    k = 0;
	
	//Calulating MSE for test data
    #pragma omp parallel for default(shared) private(i,x,y,ftest,k) reduction(+:mse)
    for (i = (int)(0.9*n); i < n; i++)
    {
        x = (int)(id[i]/m)/(float)(m+1);
        y = (id[i]%m)/(float)(m+1);
		
        ftest = gpr(m,x,y,f,K,hyp_param_min);
        mse += pow(f[id[i]]-ftest,2);
    }
	
    mse /= (0.1*n);
    
    printf("MSE on test data = %f \n",mse);    

	//Freeing the allocated memory for matrices
    free(f);  
	
    for (i = 0; i < n; i++)
         free(K[i]);
	 
    free(K);

    return 0;
}