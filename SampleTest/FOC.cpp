#include<iostream>
#include <math.h>
#include<vector>
#include <casadi/casadi.hpp>
#include <casadi/core/function_internal.hpp>

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


class FFocSolver {
private:
	MX Iq;
	MX Id;
	MX ThetaDot;
	MX Theta;
	MX Vq;
	MX Vd;
	MX Vq_dot;
	MX Vd_dot;

	casadi::SX Obj, G;
	casadi::MX X, U, X_dot;
	casadi::DM A,B;

	casadi::Function f_odes, Solver;
	casadi::SX X_s, U_s, P, VarX, VarU;

	DM LbxList, UbxList;
	DM LbgList, UbgList;

	int Steps;

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
	FFocSolver(int32_t steps, float W_theta_dot, float W_theta, float W_U_dot):Steps(steps) {
		A = DM({ {-FOCConst::R_Over_L, 0, -FOCConst::K_Over_L, 0, FOCConst::L_Inv, 0},
						{0, -FOCConst::R_Over_L, 0, 0, 0, FOCConst::L_Inv },
						{FOCConst::Sqrt_3 * FOCConst::Ke / FOCConst::I, 0, 0, 0, 0, 0},
						{0, 0, 1, 0, 0, 0},
						{0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0} });

		B = DM({ {0,0},
						{0,0},
						{0,0},
						{0,0},
						{1,0},
						{0,1} });

		//ODES
		Iq = MX::sym("iq");
		Id = MX::sym("id");

		ThetaDot = MX::sym("theta_dot");
		Theta = MX::sym("theta");

		Vq = MX::sym("vq");
		Vd = MX::sym("vd");

		Vq_dot = MX::sym("vq_dot");
		Vd_dot = MX::sym("vd_dot");

		X = vertcat(Iq, Id, ThetaDot, Theta, Vq, Vd);
		U = vertcat(Vq_dot, Vd_dot);

		//线性系统
		X_dot = mtimes(A, X) + mtimes(B, U);
		f_odes = Function("f", { X,U }, { X_dot });

		X_s = SX::sym("X_s", 6, Steps + 1);
		U_s = SX::sym("U_s", 2, Steps);
		P   = SX::sym("P", 6, 1 + (Steps + 1)); //初始 + 轨迹

		SX Q = SX::eye(6);
		Q(2, 2) = W_theta_dot;
		Q(3, 3) = W_theta;
		
		Obj = 0;
		G = X_s(Slice(), 0) - P(Slice(), 0);

		LbgList = DM::zeros(6, 1);
		UbgList = DM::zeros(6, 1);

		for (int s = 1; s < Steps + 1; s++) {
			SX st = X_s(Slice(), s) - P(Slice(), s);
			Obj = Obj + mtimes(mtimes(st.T(), Q), st) + U_s(0, s - 1) * U_s(0, s - 1) * W_U_dot;
			G = vertcat(G, X_s(Slice(), s) - Integrate(X_s(Slice(), s - 1), U_s(Slice(), s - 1)));
			//G = vertcat(G, Integrate(X_s(Slice(), s - 1), U_s(Slice(), s - 1)));
			G = vertcat(G, X_s(Slice(4, 6), s));

			LbgList = vertcat(LbgList, DM::zeros(6, 1), DM::ones(2, 1) * -24.0);
			UbgList = vertcat(UbgList, DM::zeros(6, 1), DM::ones(2, 1) * 24.0);
		}

		VarX = reshape(X_s, X_s.columns() * X_s.rows(), 1);
		VarU = reshape(U_s, U_s.columns() * U_s.rows(), 1);

		Dict opts;
		opts["ipopt.max_iter"] = 1000;
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

		Solver.generate("FOCSolver.cpp");
	}

	void StartLinearMove(float Distance, float TotalTime) {
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
		bool IsTouchDest = false;

		for (int tick = 0; tick < N+500; tick++) {
			DM DestStateList = DM::zeros(2, Stride);
			DestStateList = vertcat(DestStateList, RunningVel, RunningPos);
			DestStateList = vertcat(DestStateList, DM::zeros(2,Stride));
			DestStateList = horzcat(reshape(X_List(Slice(), 0), 6, 1), DestStateList);


			DM X_arr = reshape(X_List, 1, 6 * (Steps + 1));
			DM U_arr = reshape(U_List, 1, 2 * Steps);

			std::map<std::string, DM> arg;
			// Solve the NLP
			arg["lbx"] = LbxList;
			arg["ubx"] = UbxList;
			arg["lbg"] = LbgList;
			arg["ubg"] = UbgList;
			arg["x0"] = horzcat(X_arr,U_arr);
			arg["p"] = DestStateList;

			std::map<std::string, DM> res = Solver(arg);
			DM res_x = res["x"];
			DM U_Res = reshape(res_x(Slice(6 * (Steps + 1), 6 * (Steps + 1) + 2 * Steps), 0), 2, Steps);
			DM Y = Integrate(X_List(Slice(), 0), U_Res(Slice(), 0));
			X_List(Slice(), 0) = reshape(Y, 6, 1);



			if (IsTouchDest) {
				for (int t = 0; t < Stride; t++) {
					RunningPos(0, t) = Distance;
					RunningVel(0, t) = 0;
				}
			}
			else {
				CurrentTime += FOCConst::Ts * tick;
				for (int t = 0; t < Stride; t++) {
					RunningPos(0, t) = Y(3, 0) + FOCConst::Ts * (float)t * Vel;
					if (RunningPos(0, t)->data()[0] > Distance) {
						RunningPos(0, t) = Distance;
						RunningVel(0, t) = 0;
						IsTouchDest = true;
					}
				}
			}

			std::cout << Y;
		}
	}
};

void RunFOC() {
	FFocSolver Solver(30, 100, 2000, 10);

	Solver.StartLinearMove(100 * 2 * FOCConst::PI, 1.0);
}