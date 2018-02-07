// (C) Copyright Carnegie Mellon University 2006
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors :
// P. Bonami, Carnegie Mellon University
//
// Date :  03/17/2006
//
// MODIFIED for pyMIQP
#include <cmath>
#include "MyTMINLP.hpp"
#include "BonAmplInterface.hpp"

void MyTMINLP::add_problem(ProblemQP& problem_qp)
{
    this->problem_qp = problem_qp;
}

Eigen::VectorXd MyTMINLP::get_sol_x()
{
    return this->sol_x;
}

double MyTMINLP::get_sol_obj()
{
    return this->sol_obj;
}

unsigned int MyTMINLP::get_sol_status()
{
    return this->sol_status;
}

bool MyTMINLP::get_variables_types(Index n, VariableType* var_types)
{
  std::vector<int> qp_var_types = this->problem_qp.get_variables_types();

  for(unsigned int i=0; i<qp_var_types.size(); ++i)
    var_types[i] = static_cast<VariableType>(qp_var_types[i]);

  return true;
}

bool MyTMINLP::get_variables_linearity(Index n, Ipopt::TNLP::LinearityType* var_types)
{
  Eigen::SparseMatrix<double> Q = this->problem_qp.get_Q();

  for(int i=0; i<n; ++i)
  {
      if(Q.innerVector(i).nonZeros() > 0)
        var_types[i] = Ipopt::TNLP::NON_LINEAR;
      else
        var_types[i] = Ipopt::TNLP::LINEAR;
  }

  return true;
}


bool MyTMINLP::get_constraints_linearity(Index m, Ipopt::TNLP::LinearityType* const_types)
{
  for(int i=0; i<m; ++i)
    const_types[i] = Ipopt::TNLP::LINEAR;

  return true;
}

bool MyTMINLP::get_nlp_info(Index& n, Index&m, Index& nnz_jac_g,
                       Index& nnz_h_lag, TNLP::IndexStyleEnum& index_style)
{
  n = static_cast<Ipopt::Index>(this->problem_qp.get_n());
  m = static_cast<Ipopt::Index>(this->problem_qp.get_m());

  nnz_jac_g = this->problem_qp.get_nnz_I().size();
  if (this->hessian_approximation)
  {
    nnz_h_lag = 0;
  }
  else
  {
    nnz_h_lag = this->problem_qp.get_H_nnz_I().size();
  }

  index_style = TNLP::C_STYLE;

  return true;
}

bool MyTMINLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                            Index m, Number* g_l, Number* g_u)
{
  Eigen::VectorXd glb, gub, xlb, xub;
  glb = this->problem_qp.get_glb();
  gub = this->problem_qp.get_gub();
  xlb = this->problem_qp.get_xlb();
  xub = this->problem_qp.get_xub();

  for(int i=0; i<m; i++)
  {
    if(std::isfinite(glb[i]))
      g_l[i] = glb[i];
    else
      g_l[i] = -DBL_MAX;

    if(std::isfinite(gub[i]))
      g_u[i] = gub[i];
    else
      g_u[i] = DBL_MAX;
  }

  for(int i=0; i<n; i++)
  {
    if(std::isfinite(xlb[i]))
      x_l[i] = xlb[i];
    else
      x_l[i] = -DBL_MAX;

    if(std::isfinite(xub[i]))
      x_u[i] = xub[i];
    else
      x_u[i] = DBL_MAX;
  }

  return true;
}

bool MyTMINLP::get_starting_point(Index n, bool init_x, Number* x,
                             bool init_z, Number* z_L, Number* z_U,
                             Index m, bool init_lambda,
                             Number* lambda)
{
  if(this->problem_qp.get_initial_point_status())
  {
    auto initial_point = this->problem_qp.get_initial_point();
    std::copy(initial_point.begin(), initial_point.end(), x);
  }
  else
  {
    for(int i=0; i<n; ++i)
      x[i] = 0;
  }

  return true;
}

bool MyTMINLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  Eigen::VectorXd eigen_x = Eigen::Map<const Eigen::VectorXd>(x, n, 1);  // IPOPT to Eigen
  double linear_component = this->problem_qp.get_c().dot(eigen_x);
  auto quad_0 = eigen_x.transpose() * this->problem_qp.get_Q();
  double quad_component = quad_0.dot(eigen_x);
  obj_value = linear_component + 0.5 * quad_component;

  return true;
}

bool MyTMINLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
  Eigen::VectorXd eigen_x = Eigen::Map<const Eigen::VectorXd>(x, n, 1);  // IPOPT to Eigen
  Eigen::VectorXd gradient = this->problem_qp.get_c();
  gradient += this->problem_qp.get_Q() * eigen_x;
  Eigen::Map<Eigen::VectorXd>(grad_f, gradient.rows()) = gradient;       // Eigen to IPOPT

  return true;
}

bool MyTMINLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  Eigen::VectorXd eigen_x = Eigen::Map<const Eigen::VectorXd>(x, n, 1);  // IPOPT to Eigen
  Eigen::VectorXd g_val = this->problem_qp.get_A() * eigen_x;
  Eigen::Map<Eigen::VectorXd>(g, g_val.rows()) = g_val;       // Eigen to IPOPT

  return true;
}

bool MyTMINLP::eval_jac_g(Index n, const Number* x, bool new_x,
                     Index m, Index nnz_jac, Index* iRow, Index *jCol,
                     Number* values)
{
  auto nnz_I = this->problem_qp.get_nnz_I();
  auto nnz_J = this->problem_qp.get_nnz_J();

  if((iRow != NULL) && (jCol != NULL))
  {
   for(int i=0; i<nnz_I.size(); ++i)
   {
     iRow[i] = static_cast<int>(nnz_I[i]);
   }

   for(int i=0; i<nnz_J.size(); ++i)
   {
     jCol[i] = static_cast<int>(nnz_J[i]);
   }
  }
  else
  {
    unsigned int bla = nnz_I.size();
    auto nnz_flat = this->problem_qp.get_nnz_flat();
    auto nnz_V = this->problem_qp.get_nnz_V();

    for(unsigned int i=0; i<bla; ++i)
    {
      values[nnz_flat[i]] = nnz_V[i];
    }
  }

  return true;
}


bool MyTMINLP::eval_h(Index n, const Number* x, bool new_x,
                 Number obj_factor, Index m, const Number* lambda,
                 bool new_lambda, Index nele_hess, Index* iRow,
                 Index* jCol, Number* values)
{
  if (!this->hessian_approximation)
  {
    auto H_nnz_I = this->problem_qp.get_H_nnz_I();
    auto H_nnz_J = this->problem_qp.get_H_nnz_J();

    if((iRow != NULL) && (jCol != NULL))
    {
      for(int i=0; i<H_nnz_I.size(); ++i)
      {
        iRow[i] = static_cast<int>(H_nnz_I[i]);
      }

      for(int i=0; i<H_nnz_J.size(); ++i)
      {
        jCol[i] = static_cast<int>(H_nnz_J[i]);
      }
    }
    else
    {
      unsigned int bla = H_nnz_I.size();
      auto H_nnz_flat = this->problem_qp.get_H_nnz_flat();
      auto H_nnz_V = this->problem_qp.get_H_nnz_V();

      for(unsigned int i=0; i<bla; ++i)
      {
        values[H_nnz_flat[i]] = obj_factor * H_nnz_V[i];
      }
    }
    return true;
  }
  else
  {
    return false;
  }
}

void MyTMINLP::finalize_solution(TMINLP::SolverReturn status,
                            Index n, const Number* x, Number obj_value)
{
  this->sol_obj = obj_value;
  this->sol_status = static_cast<int>(status);

  if(x != NULL)
  {
      Eigen::VectorXd eigen_x = Eigen::Map<const Eigen::VectorXd>(x, n, 1);  // IPOPT to Eigen
      this->sol_x = eigen_x;  // TODO copy vs. ref
  }
}
