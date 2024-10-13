#include<iostream>
#include <math.h>
#include<vector>
#include <casadi/casadi.hpp>
#include <casadi/core/function_internal.hpp>
#include <chrono>

using namespace std;
using namespace casadi;

namespace FOCConst {
	const float PI = 3.1415926;
	const float Ts = 0.001;
	const int32_t Steps = 30;
	const float Dest = 100 * 2 * PI;
	const float TotalTime = 1.0;

	const float R = 1.02;
	const float Lqd = 0.00059;
	const float L_Inv = 1.0 / Lqd;
	const float Ke = (4.3 * 60) / (2000 * PI);
	const float R_Over_L = R / Lqd;
	const float K_Over_L = Ke / Lqd;
	const float m = 0.051;
	const float I = m * 0.0325 * 0.0325;
	const float Sqrt_3 = sqrtf(3);
};

extern "C" int solver(const double** arg, double** res, int* iw, double* w, int mem);
extern "C" const char* solver_name_in(int i);
extern "C" const char* solver_name_out(int i);
extern "C" int solver_n_in(void);
extern "C" int solver_n_out(void);
extern "C" int solver_work_bytes(casadi_int * sz_arg, casadi_int * sz_res, casadi_int * sz_iw, casadi_int * sz_w);


class FFocSolver {
private:
	casadi::Function f_odes, Solver;

	DM LbxList, UbxList;
	DM LbgList, UbgList;

	int Steps;

	const bool UseGenerateSolver;

	SX Integrate(const SX &In_X, const SX& In_U) {
		const float Inv_6 = 1.0 / 6.0;
		const float Ts = FOCConst::Ts;
		std::vector<SX> f1 = f_odes(std::vector<SX>{In_X, In_U });
		std::vector<SX>	f2 = f_odes(std::vector<SX>{In_X + f1[0] * (0.5 * Ts), In_U});
		std::vector<SX>	f3 = f_odes(std::vector<SX>{In_X + f2[0] * (0.5 * Ts), In_U});
		std::vector<SX>	f4 = f_odes(std::vector<SX>{In_X + f3[0] * Ts, In_U});
		return In_X + (f1[0] + 2 * f2[0] + 2 * f3[0] + f4[0]) * Ts * Inv_6;
	}

public:
	FFocSolver(int32_t steps, float W_theta_dot, float W_theta, float W_U_dot, bool IsGenerate):Steps(steps),UseGenerateSolver(IsGenerate) {
		DM A = DM({ {-FOCConst::R_Over_L, 0, -FOCConst::K_Over_L, 0, FOCConst::L_Inv, 0},
						{0, -FOCConst::R_Over_L, 0, 0, 0, FOCConst::L_Inv },
						{FOCConst::Sqrt_3 * FOCConst::Ke / FOCConst::I, 0, 0, 0, 0, 0},
						{0, 0, 1, 0, 0, 0},
						{0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0} });

		DM B = DM({ {0,0},
						{0,0},
						{0,0},
						{0,0},
						{1,0},
						{0,1} });

		//ODES
		MX Iq = MX::sym("iq");
		MX Id = MX::sym("id");

		MX ThetaDot = MX::sym("theta_dot");
		MX Theta = MX::sym("theta");

		MX Vq = MX::sym("vq");
		MX Vd = MX::sym("vd");

		MX Vq_dot = MX::sym("vq_dot");
		MX Vd_dot = MX::sym("vd_dot");

		MX X = vertcat(Iq, Id, ThetaDot, Theta, Vq, Vd);
		MX U = vertcat(Vq_dot, Vd_dot);

		//线性系统
		MX X_dot = mtimes(A, X) + mtimes(B, U);
		f_odes = Function("f", { X,U }, { X_dot });

		SX X_s = SX::sym("X_s", 6, Steps + 1);
		SX U_s = SX::sym("U_s", 2, Steps);
		SX P   = SX::sym("P", 6, 1 + (Steps + 1)); //初始 + 轨迹

		SX Q = SX::zeros(6,6);
		Q(2, 2) = W_theta_dot;
		Q(3, 3) = W_theta;
		Q(5, 5) = 1000;
		
		SX Obj = 0;
		SX G = X_s(Slice(), 0) - P(Slice(), 0);

		LbgList = DM::zeros(6, 1);
		UbgList = DM::zeros(6, 1);

		for (int s = 1; s < Steps + 1; s++) {
			SX st = X_s(Slice(), s) - P(Slice(), s);
			Obj = Obj + mtimes(mtimes(st.T(), Q), st) + U_s(0, s - 1) * U_s(0, s - 1) * W_U_dot;
			G = vertcat(G, X_s(Slice(), s) - Integrate(X_s(Slice(), s - 1), U_s(Slice(), s - 1)));
			G = vertcat(G, X_s(Slice(4, 5), s));

			LbgList = vertcat(LbgList, DM::zeros(6, 1), DM::ones(1, 1) * -24.0);
			UbgList = vertcat(UbgList, DM::zeros(6, 1), DM::ones(1, 1) * 24.0);
		}

		SX VarX = reshape(X_s, X_s.columns() * X_s.rows(), 1);
		SX VarU = reshape(U_s, U_s.columns() * U_s.rows(), 1);

		Dict opts;
		opts["ipopt.max_iter"] = 100;
		opts["ipopt.print_level"] = 0;
		opts["ipopt.sb"] = "yes";

		SXDict nlp;
		nlp["x"] = vertcat(VarX, VarU);
		nlp["f"] = Obj;
		nlp["p"] = P;
		nlp["g"] = G;

		Solver = nlpsol("solver", "ipopt", nlp, opts);
		DM lbx_x = DM::ones(6, Steps + 1) * -inf;
		DM lbx_u = DM::ones(2, Steps) * -1000;

		DM ubx_x = DM::ones(6, Steps + 1) * inf;
		DM ubx_u = DM::ones(2, Steps) * 1000;

		LbxList = horzcat(reshape(lbx_x, 1, lbx_x.columns() * lbx_x.rows()), reshape(lbx_u, 1, lbx_u.columns() * lbx_u.rows()));
		UbxList = horzcat(reshape(ubx_x, 1, ubx_x.columns() * ubx_x.rows()), reshape(ubx_u, 1, ubx_u.columns() * ubx_u.rows()));
	}

	void Generate() {
		if (UseGenerateSolver) {
			return;
		}
		Solver.generate("FOCSolver.c");
	}

	void StartLinearMoveGenerate(float Distance, float TotalTime, int DebugTicks) {
		int N = (int)(TotalTime / FOCConst::Ts);
		int Stride = Steps + 1;
		float Vel = Distance / TotalTime;

		std::vector<float> Timeline, PosX, SpeedX;

		DM RunningPos = DM::zeros(1, Stride);
		DM RunningVel = DM::ones(1, Stride) * Vel;

		Timeline.resize(N);
		PosX.resize(Stride);
		SpeedX.resize(Stride);
		for (int t = 0; t < N; t++) {
			Timeline[t] = t * FOCConst::Ts;
			if (t < Stride) {
				PosX[t] = Timeline[t] * Vel;
				SpeedX[t] = Vel;

				RunningPos(0, t) = PosX[t];
			}
		}

		DM X_List = DM::zeros(6, Steps + 1);
		DM U_List = DM::zeros(2, Steps);

		float CurrentTime = 0;

		std::vector<double*> ArgsIn;
		std::vector<double*> ResOut;
		std::vector<double> ResX;

		ResX.resize(6 * (Steps + 1) + 2 * Steps);
		ArgsIn.resize(solver_n_in());
		ResOut.resize(solver_n_out());

		std::vector<double> X0;
		std::vector<double> P;
		X0.resize(6 * (Steps + 1) + 2 * Steps);
		P.resize(6 * (Stride + 1));

		enum ESolverIn : unsigned char
		{
			lbx=0,
			ubx,
			lbg,
			ubg,
			x0,
			p,
			max
		};

		int SolverInIndex[ESolverIn::max];

		for (size_t i = 0; i < solver_n_in(); i++) {
			std::string SolverName(solver_name_in(i));
			if (SolverName == "lbx") {
				SolverInIndex[ESolverIn::lbx] = i;
			}
			else if (SolverName == "ubx") {
				SolverInIndex[ESolverIn::ubx] = i;
			}
			else if (SolverName == "lbg") {
				SolverInIndex[ESolverIn::lbg] = i;
			}
			else if (SolverName == "ubg") {
				SolverInIndex[ESolverIn::ubg] = i;
			}
			else if (SolverName == "x0") {
				SolverInIndex[ESolverIn::x0] = i;
			}
			else if (SolverName == "p") {
				SolverInIndex[ESolverIn::p] = i;
			}
		}

		int ResIndex = 0;
		for (size_t i = 0; i < solver_n_out(); i++) {
			std::string SolverName(solver_name_out(i));
			if (SolverName == "x") {
				ResIndex = i;
				break;
			}
		}

		int64_t dwSize, iwSize;
		solver_work_bytes(nullptr, nullptr, &iwSize, &dwSize);
		std::vector<int> iwMemory;
		std::vector<double> dwMemory;
		iwMemory.resize(iwSize / sizeof(int));
		dwMemory.resize(dwSize / sizeof(double));

		const int TotalTicks = DebugTicks == -1 ? N + 500 : DebugTicks;
		for (int tick = 0; tick < TotalTicks; tick++) {
			DM DestStateList = DM::zeros(2, Stride);
			DestStateList = vertcat(DestStateList, RunningVel, RunningPos);
			DestStateList = vertcat(DestStateList, DM::zeros(2, Stride));
			DestStateList = horzcat(reshape(X_List(Slice(), 0), 6, 1), DestStateList);


			DM X_arr = reshape(X_List, 1, 6 * (Steps + 1));
			DM U_arr = reshape(U_List, 1, 2 * Steps);

			memcpy(&X0[0], horzcat(X_arr, U_arr)->data(), X0.size() * sizeof(double));
			memcpy(&P[0], DestStateList->data(), P.size() * sizeof(double));

			ArgsIn[SolverInIndex[ESolverIn::lbx]] = LbxList->data();
			ArgsIn[SolverInIndex[ESolverIn::ubx]] = UbxList->data();
			ArgsIn[SolverInIndex[ESolverIn::lbg]] = LbgList->data();
			ArgsIn[SolverInIndex[ESolverIn::ubg]] = UbgList->data();
			ArgsIn[SolverInIndex[ESolverIn::x0]] = X0.data();
			ArgsIn[SolverInIndex[ESolverIn::p]] = P.data();

			ResOut[ResIndex] = &ResX[0];
			//std::map<std::string, DM> res = Solver(ArgsIn.data(), ;
			auto start = std::chrono::high_resolution_clock::now();
			::solver((const double**)ArgsIn.data(), ResOut.data(), (int*)(iwMemory.size() ? &iwMemory[0] : nullptr), (double*)&dwMemory[0], 0);
			std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
			//DM res_x = res["x"];
			DM U_Res = DM({ {ResX[6 * (Steps + 1)]},{ResX[6 * (Steps + 1) + 1]} }); //= reshape(res_x(Slice(6 * (Steps + 1), 6 * (Steps + 1) + 2 * Steps), 0), 2, Steps);
			DM Y = Integrate(X_List(Slice(), 0), U_Res(Slice(), 0));
			X_List(Slice(), 0) = reshape(Y, 6, 1);

			DM PridPos = Y(3) + FOCConst::Ts * Stride * Vel;

			if (PridPos->data()[0] >= Distance) {
				RunningPos(0, Slice()) = Distance;
				RunningVel(0, Slice()) = 0;
			}
			else {
				CurrentTime += FOCConst::Ts * tick;
				for (int t = 0; t < Stride; t++) {
					RunningPos(0, t) = Y(3) + FOCConst::Ts * (float)t * Vel;
				}
			}

			std::cout << Y << " " << U_Res << " " << duration.count() << "\n";
			if (Y(3)->data()[0] >= Distance) {
				std::cout << "reach distance \n";
				break;
			}
		}
	}

	void StartLinearMove(float Distance, float TotalTime, int DebugTicks) {
		if (UseGenerateSolver) {
			StartLinearMoveGenerate(Distance, TotalTime, DebugTicks);
			return;
		}
		int N = (int)(TotalTime / FOCConst::Ts);
		int Stride = Steps + 1;
		float Vel = Distance / TotalTime;

		std::vector<float> Timeline, PosX, SpeedX;

		DM RunningPos = DM::zeros(1, Stride);
		DM RunningVel = DM::ones(1, Stride) * Vel;

		Timeline.resize(N);
		PosX.resize(Stride);
		SpeedX.resize(Stride);
		for (int t = 0; t < N; t++) {
			Timeline[t] = t * FOCConst::Ts;
			if (t < Stride) {
				PosX[t] = Timeline[t] * Vel;
				SpeedX[t] = Vel;

				RunningPos(0, t) = PosX[t];
			}
		}

		DM X_List = DM::zeros(6, Steps + 1);
		DM U_List = DM::zeros(2, Steps);

		float CurrentTime = 0;

		StringVector SolverNamesIn = Solver.name_in();
		StringVector SolverNamesOut = Solver.name_out();

		std::vector<const double*> ArgsIn;
		std::vector<double*> ResOut;
		std::vector<double> ResX;

		ResX.resize(6*(Steps+1)+2*Steps);
		ArgsIn.resize(SolverNamesIn.size());
		ResOut.resize(SolverNamesOut.size());

		std::vector<double> X0;
		std::vector<double> P;
		X0.resize(6 * (Steps + 1) + 2 * Steps);
		P.resize(6 * (Stride + 1));

		for (size_t i = 0; i < SolverNamesIn.size(); i++) {
			if (SolverNamesIn[i] == "lbx") {
				ArgsIn[i] = LbxList->data();
			}
			else if (SolverNamesIn[i] == "ubx") {
				ArgsIn[i] = UbxList->data();
			}
			else if (SolverNamesIn[i] == "lbg") {
				ArgsIn[i] = LbgList->data();
			}
			else if (SolverNamesIn[i] == "ubg") {
				ArgsIn[i] = UbgList->data();
			}
			else if (SolverNamesIn[i] == "x0") {
				ArgsIn[i] = &X0[0];
			}
			else if (SolverNamesIn[i] == "p") {
				ArgsIn[i] = &P[0];
			}
		}

		for (size_t i = 0; i < SolverNamesOut.size(); i++) {
			if (SolverNamesOut[i] == "x") {
				ResOut[i] = &ResX[0];
				break;
			}
		}

		const int TotalTicks = DebugTicks == -1 ? N + 500 : DebugTicks;
		for (int tick = 0; tick < TotalTicks; tick++) {
			DM DestStateList = DM::zeros(2, Stride);
			DestStateList = vertcat(DestStateList, RunningVel, RunningPos);
			DestStateList = vertcat(DestStateList, DM::zeros(2,Stride));
			DestStateList = horzcat(reshape(X_List(Slice(), 0), 6, 1), DestStateList);


			DM X_arr = reshape(X_List, 1, 6 * (Steps + 1));
			DM U_arr = reshape(U_List, 1, 2 * Steps);

			//std::map<std::string, DM> arg;
			// Solve the NLP
			//arg["lbx"] = LbxList;
			//arg["ubx"] = UbxList;
			//arg["lbg"] = LbgList;
			//arg["ubg"] = UbgList;
			//arg["x0"] = horzcat(X_arr,U_arr);
			//arg["p"] = DestStateList;
			//ArgsIn[X0_Index] = horzcat(X_arr, U_arr)->data();
			//ArgsIn[P_Index] = DestStateList->data();

			memcpy(&X0[0], horzcat(X_arr, U_arr)->data(), X0.size() * sizeof(double));
			memcpy(&P[0], DestStateList->data(), P.size() * sizeof(double));

			//std::map<std::string, DM> res = Solver(ArgsIn.data(), ;
			auto start = std::chrono::high_resolution_clock::now();
			Solver(ArgsIn, ResOut);
			std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
			//DM res_x = res["x"];
			DM U_Res = DM({ {ResX[6 * (Steps + 1)]},{ResX[6 * (Steps + 1)+1]}}); //= reshape(res_x(Slice(6 * (Steps + 1), 6 * (Steps + 1) + 2 * Steps), 0), 2, Steps);
			DM Y = Integrate(X_List(Slice(), 0), U_Res(Slice(), 0));
			X_List(Slice(), 0) = reshape(Y, 6, 1);

			DM PridPos = Y(3) + FOCConst::Ts * Stride * Vel;

			if (PridPos->data()[0] >= Distance) {
				RunningPos(0, Slice()) = Distance;
				RunningVel(0, Slice()) = 0;
			}
			else {
				CurrentTime += FOCConst::Ts * tick;
				for (int t = 0; t < Stride; t++) {
					RunningPos(0, t) = Y(3) + FOCConst::Ts * (float)t * Vel;
				}
			}

			std::cout << Y << " " << U_Res << " " << duration.count() << "\n";
			if (Y(3)->data()[0] >= Distance) {
				std::cout << "reach distance";
				break;
			}
		}
	}
};

void RunFOC(bool UseGenerate, int ticks) {
	FFocSolver Solver(30, 70, 3000, 10, UseGenerate);
	Solver.Generate();
	Solver.StartLinearMove(100 * 2 * FOCConst::PI, 1.0, ticks);
}