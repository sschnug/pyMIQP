#include "MIQP.hpp"
#include "MyTMINLP.hpp"

#include <cmath>

#include "CoinPragma.hpp"
#include "CoinTime.hpp"
#include "CoinError.hpp"

#include "BonOsiTMINLPInterface.hpp"
#include "BonIpoptSolver.hpp"
#include "MyTMINLP.hpp"
#include "BonCbc.hpp"
#include "BonBonminSetup.hpp"

#include "BonOACutGenerator2.hpp"
#include "BonEcpCuts.hpp"
#include "BonOaNlpOptim.hpp"

#include <chrono>

using namespace Ipopt;
using namespace Bonmin;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

// QP Problem API

void MIQP::set_c(RowArray& c)
{
    this->problemqp.set_c(c);
}

void MIQP::set_Q(SpMat& Q)
{
    this->problemqp.set_Q(Q);
}

void MIQP::set_A(SpMat& A)
{
    this->problemqp.set_A(A);
}

void MIQP::set_glb(RowArray& glb)
{
    this->problemqp.set_glb(glb);
}

void MIQP::set_gub(RowArray& gub)
{
    this->problemqp.set_gub(gub);
}

void MIQP::set_xlb(RowArray& xlb)
{
    this->problemqp.set_xlb(xlb);
}

void MIQP::set_xub(RowArray& xub)
{
    this->problemqp.set_xub(xub);
}

void MIQP::set_var_types(std::vector<int>& var_types)
{
    this->problemqp.set_var_types(var_types);
}

void MIQP::set_initial_point(std::vector<double>& initial_point)
{
    this->problemqp.set_initial_point(initial_point);
}

void MIQP::solve(std::string algorithm)
{
    SmartPtr<MyTMINLP> tminlp = new MyTMINLP;
    tminlp->add_problem(this->problemqp);

    BonminSetup bonmin;
    bonmin.initializeOptionsAndJournalist();

    if (this->hessian_approximation)
      bonmin.options()->SetStringValue("hessian_approximation", "limited-memory");

    if (!this->verbose)
    {
      bonmin.options()->SetIntegerValue("bb_log_level", 0);
      bonmin.options()->SetIntegerValue("fp_log_level", 0);
      bonmin.options()->SetIntegerValue("lp_log_level", 0);
      bonmin.options()->SetIntegerValue("milp_log_level", 0);
      bonmin.options()->SetIntegerValue("nlp_log_level", 0);
      bonmin.options()->SetIntegerValue("oa_log_level", 0);
      bonmin.options()->SetStringValue("sb", "yes");         // Ipopt banner
    }

    bonmin.options()->SetStringValue("algorithm", algorithm);
    bonmin.initialize(GetRawPtr(tminlp));

    //Set up done, now let's branch and bound
    try {
      Bab bb;

      auto t0 = Time::now();

      bb(bonmin);//process parameter file using Ipopt and do branch and bound using Cbc

      auto t1 = Time::now();
      ms d = std::chrono::duration_cast<ms>(t1 - t0);
      this->sol_time  = d.count();
      this->sol_x = tminlp->get_sol_x();
      this->sol_obj = tminlp->get_sol_obj();
      this->sol_status = tminlp->get_sol_status();
    }
    catch(TNLPSolver::UnsolvedError *E) {
      //There has been a failure to solve a problem with Ipopt.
      std::cerr<<"Ipopt has failed to solve a problem"<<std::endl;
    }
    catch(OsiTMINLPInterface::SimpleError &E) {
      std::cerr<<E.className()<<"::"<<E.methodName()
  	     <<std::endl
  	     <<E.message()<<std::endl;
    }
    catch(CoinError &E) {
      std::cerr<<E.className()<<"::"<<E.methodName()
  	     <<std::endl
  	     <<E.message()<<std::endl;
    }
}

RowArray MIQP::get_sol_x()
{
    return this->sol_x;
}

double MIQP::get_sol_obj()
{
    return this->sol_obj;
}

int MIQP::get_sol_status()
{
    return this->sol_status;
}

long MIQP::get_sol_time()
{
    return this->sol_time;
}
