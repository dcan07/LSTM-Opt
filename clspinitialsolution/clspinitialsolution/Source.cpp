#pragma warning(disable : 4996)


#include <ilcplex/ilocplex.h>
ILOSTLBEGIN
#include <iostream>
#include <string>
#include <random>
#include <cstdlib>

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

//Variables to time the solution
clock_t  start1, end1;
double CPLEXtime = 0;


int main(int argc, char **argv){

	printf(" Time =  %.2f\n", CPLEXtime);
	//Number of instances in the test set
	int testsetlength;
	testsetlength = 20000;
	printf("%d", testsetlength);
	//Capacity to demand ratio - c
	int capdem;
	//Setup to holding cost ratio - f
	int sethold;
	//Number of stages in CLSP
	int NumOfStage;

	//IMPORTANT INPUT
	//Choose to one of the  MipStart or USerCuts approach to use
	//string Option = "MS";
	string Option = "UC";

	//Input files that will be solved using predictions from LSTM
	//Could be single file
	string inputfiles[] = {
		"/results/c3f1000t90-percent100.txt",
		"/results/c5f1000t90-percent100.txt" };

	//Output files that the results will be written
	string outputfiles[] = {
		"/results/resultsc3f1000t90-percent100uc.txt",
		"/results/resultsc5f1000t90-percent100uc.txt" };

	//Number of stages for files
	//Note that inputfiles, outputfiles and NumOfStageArr should be of same length
	int NumOfStageArr[] = { 90, 90};

	//Loop for number of files
	for (unsigned int a = 0; a < sizeof(inputfiles) / sizeof(inputfiles[0]); a = a + 1)
	{
		//Get the number of stages
		NumOfStage = NumOfStageArr[a];
		printf("%d", NumOfStage);
		printf("\n");

		//Get inout and output files
		ifstream inputfile;
		inputfile.open(inputfiles[a]);
		ofstream output;
		output.open(outputfiles[a]);

		//Resolve each instance
		for (int instance = 0; instance < testsetlength; instance++)
		{
			printf("\n");
			printf("\n");
			printf("instance %d", instance);
			printf("\n");
			printf("\n");

			IloEnv env;

			inputfile >> capdem;
			inputfile >> sethold;
			printf("%d", capdem);
			printf("\n");
			printf("%d", sethold);
			printf("\n");

			IloIntArray p(env, NumOfStage);
			//Read Unit Production Cost from input file
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> p[i];

			}

			IloIntArray d(env, NumOfStage);
			//Read demand from input file
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> d[i];
			}

			IloIntArray c(env, NumOfStage);
			//Read capacity from input file
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> c[i];
			}

			IloIntArray h(env, NumOfStage);
			//Read holding cost from input file
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> h[i];
			}

			IloIntArray s(env, NumOfStage);
			//Read setup from input file
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> s[i];
			}

			//Read original solution time from input file
			double originalsoltime;
			inputfile >> originalsoltime;

			//Read objective value from input file
			IloNum  objvalue;
			inputfile >> objvalue;
			printf("%d", objvalue);
			printf("\n");


			//Lets absorbe x y and inv, from the original solution
			//We do not use them in any point in here
			IloNumArray givenx(env, NumOfStage);
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> givenx[i];
			}
			IloNumArray giveny(env, NumOfStage);
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> giveny[i];
			}
			//Sometimes it works with IloInt
			IloNum  InitialInv;
			inputfile >> InitialInv;
			printf("%d ", InitialInv);
			IloNumArray giveninv(env, NumOfStage);
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> giveninv[i];
			}



			//This is predicted y variable by LSTM
			IloNumArray predy(env, NumOfStage);
			for (int i = 0; i < NumOfStage; i++)
			{
				inputfile >> predy[i];
			}

			//Define decision variable arrays
			IloNumVarArray x(env, NumOfStage), inv(env, NumOfStage), y(env, NumOfStage);
			for (int t = 0; t < NumOfStage; t++) {
				x[t] = IloNumVar(env, 0);
				inv[t] = IloNumVar(env, 0);
				y[t] = IloNumVar(env, 0, 1, ILOINT);
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

			IloCplex cplex(env);
			cplex.extract(mod);
			cplex.exportModel("clsp.lp");


			if (Option=="MS")
			{
				//This part is addMIPstart
				IloNumVarArray startVar(env);
				IloNumArray startVal(env);
				for (int t = 0; t < NumOfStage; t++){
					if (predy[t] != -1){
						startVar.add(y[t]);
						startVal.add(predy[t]);
					}
				}
				cplex.addMIPStart(startVar, startVal, IloCplex::MIPStartSolveMIP);
				startVal.end();
				startVar.end();

				cplex.setParam(IloCplex::Param::MIP::Limits::RepairTries, 1000);
			}
			

			if (Option=="UC")
			{
				//This part is user cuts
				IloConstraintArray predictedconst(env);
				for (int t = 0; t < NumOfStage; t++){
					if (predy[t] != -1){
						predictedconst.add(y[t] == predy[t]);
					}
				}
				cplex.addUserCuts(predictedconst);
			}
			
			//SolveProblem
			start1 = clock();
			cplex.solve();
			end1 = clock();
			//Output solution time
			CPLEXtime = CPLEXtime + ((end1 - start1) / (double)CLOCKS_PER_SEC);
			double mlsoltime;
			mlsoltime = ((end1 - start1) / (double)CLOCKS_PER_SEC);

			output << originalsoltime << ";";
			output << mlsoltime << ";";

			output << objvalue << ";";
			// Get solution status
			env.out() << "Solution status = " << cplex.getStatus() << endl;
			//If optimal solution is found output the objective value
			if (cplex.getStatus() == IloAlgorithm::Optimal)
			{
				env.out() << "Solution value = " << cplex.getObjValue() << endl;
				output << cplex.getObjValue() << ";";
				output << endl;
			}
			//Otherwise output infeasible
			else
			{
				output << cplex.getStatus() << endl;

			}
			env.end();
		}
		printf(" Time =  %.2f\n", CPLEXtime);
		inputfile.close();
		output.close();

	}

	system("PAUSE");
	return 0;
}

