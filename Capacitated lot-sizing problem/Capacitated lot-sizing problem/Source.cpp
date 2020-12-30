#pragma warning(disable : 4996)


#include <ilcplex/ilocplex.h>
ILOSTLBEGIN
#include <iostream>
#include <string>
#include <random>
#include <cstdlib>


#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>



//Variables to time the solution
clock_t  start1, end1;
double CPLEXtime=0;


int main(int argc, char **argv){

	//Number of CLSP Instances that must be solved
	int NumOfInstance = 100000;
	//Capacity to demand ratio - c
	int capdem = 3;
	//Setup to holding cost ratio - f
	int sethold = 1000;
	//Number of stages in CLSP
	int NumOfStage = 90;
	//Variables used for generating problem parameters
	int range_from;
	int range_to;
	int dbar;
	int hbar;
	
	//Output file
	ofstream myfile;
	myfile.open("/data/resultsfromclspsolvercd3f1000t90n120k.txt");
	
	//Start with zero initial inventory
	IloInt  InitialInv = 0;

	//Loop through the number of instances to generate CLSP instances
	for (int instance = 0; instance < NumOfInstance; instance++)
	{
		printf("\n");
		printf("instance %d", instance);
		printf("\n");

		//Enviroment
		IloEnv env;


		IloIntArray p(env, NumOfStage);
		//Unit Production Cost random generation for each period
		range_from = 1;
		range_to = 5;
		std::random_device                  rand_dev;
		std::mt19937                        generator(rand_dev());
		std::uniform_int_distribution<int>  dist(range_from, range_to);

		for (int i = 0; i < NumOfStage; i++)
		{
			p[i] = dist(generator);
		}

		IloIntArray d(env, NumOfStage);
		//Demand random generation for each period
		range_from = 1;
		range_to = 600;

		std::uniform_int_distribution<int>  distri(range_from, range_to);

		for (int i = 0; i < NumOfStage; i++)
		{
			d[i] = distri(generator);
		}

		//Calculate dbar
		dbar = IloSum(d) / d.getSize();

		IloIntArray c(env, NumOfStage);
		//Capacity random generation for each period
		range_from = 0.9*capdem*dbar;
		range_to = 1.1*capdem*dbar;

		std::uniform_int_distribution<int>  distrib(range_from, range_to);

		for (int i = 0; i < NumOfStage; i++)
		{
			c[i] = distrib(generator);
		}


		IloIntArray h(env, NumOfStage);
		//Constant holding cost
		for (int i = 0; i < NumOfStage; i++)
		{
			h[i] = 1;
		}

		hbar = IloSum(h) / h.getSize();


		IloIntArray s(env, NumOfStage);
		//Setup cost random generation for each period
		range_from = 0.9*sethold*hbar;
		range_to = 1.1*sethold*hbar;

		std::uniform_int_distribution<int>  distribit(range_from, range_to);

		for (int i = 0; i < NumOfStage; i++)
		{
			s[i] = distribit(generator);
		}

		//Output the generated parameters
		myfile << capdem << ";";
		myfile << sethold << ";";
		for (int i = 0; i < NumOfStage; i++)
		{
			myfile << p[i] << ";";
		}
		for (int i = 0; i < NumOfStage; i++)
		{
			myfile << d[i] << ";";
		}for (int i = 0; i < NumOfStage; i++)
		{
			myfile << c[i] << ";";
		}
		for (int i = 0; i < NumOfStage; i++)
		{
			myfile << h[i] << ";";
		}
		for (int i = 0; i < NumOfStage; i++)
		{
			myfile << s[i] << ";";
		}

		//Define decision variable arrays
		IloNumVarArray x(env, NumOfStage), y(env, NumOfStage), inv(env, NumOfStage);
		for (int t = 0; t < NumOfStage; t++) {
			x[t] = IloNumVar(env);
			y[t] = IloNumVar(env, 0, 1, ILOINT);
			inv[t] = IloNumVar(env);
		}


		IloModel mod(env);

		//Objective function
		IloExpr Obj(env);
		for (int t = 0; t < NumOfStage; t++)
		{
			Obj += p[t] * x[t] + s[t] * y[t] + h[t] * inv[t];
		}
		mod.add(IloMinimize(env, Obj));
		Obj.end();

		//Add flow balance constraints
		IloExpr FlowBalance(env);
		for (int t = 0; t < NumOfStage; t++){

			//If first period, use initial inventory
			if (t == 0)
			{
				mod.add(InitialInv + x[t] - d[t] == inv[t]);

			}
			//Otherwise use inventory from previous period
			else
			{
				mod.add(inv[t - 1] + x[t] - d[t] == inv[t]);
			}
		}
		FlowBalance.end();


		//Production capacity constraint
		IloExpr ProdCap(env);
		for (int t = 0; t < NumOfStage; t++){
			mod.add(x[t] <= c[t] * y[t]);
		}
		ProdCap.end();

		//Set model
		IloCplex cplex(env);
		cplex.extract(mod);
		cplex.exportModel("clsp.lp");

		//SolveProblem
		start1 = clock();
		cplex.solve();
		end1 = clock();
		//Output solution time
		CPLEXtime = CPLEXtime + ((end1 - start1) / (double) CLOCKS_PER_SEC);
		myfile << ((end1 - start1) / (double)CLOCKS_PER_SEC) << ";";

		//Get solution status
		env.out() << "Solution status = " << cplex.getStatus() << endl;

		//If optimal solution is found output the decision variables
		if (cplex.getStatus() == IloAlgorithm::Optimal)
		{
			//Outut objective value
			env.out() << "Solution value = " << cplex.getObjValue() << endl;
			myfile << cplex.getObjValue() << ";";

			//Output the decision variables
			IloNumArray vals(env);
			IloNumArray vals1(env);
			IloNumArray vals2(env);
			cplex.getValues(vals, x);
			//env.out() << "x Values = " << vals << endl;
			for (int i = 0; i < NumOfStage; i++)
			{
				myfile << vals[i] << ";";
				//printf("%f ",vals[i]);
				
			}	
			printf("\n");

			cplex.getValues(vals1, y);
			//env.out() << "y Values = " << vals1 << endl;
			for (int i = 0; i < NumOfStage; i++)
			{
				myfile << vals1[i] << ";";
				//printf("%f ", vals1[i]);
				
			}
			printf("\n");

			myfile << InitialInv << ";";
			cplex.getValues(vals2, inv);
			//env.out() << "i Values = " << vals2 << endl;
			for (int i = 0; i < NumOfStage; i++)
			{
				myfile << vals2[i] << ";";
				//printf("%f ", vals[i]);
			}
			myfile << endl;
		}
		//If infeasible
		else
		{
			myfile << cplex.getStatus() << endl;
			
		}

		env.end();
	}

	//Print totsl runtime
	printf(" Time =  %.2f\n", CPLEXtime);
	myfile.close();
	system("PAUSE");
	return 0;
}



