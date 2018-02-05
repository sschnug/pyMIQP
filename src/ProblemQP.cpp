#include "ProblemQP.hpp"

// Helpers
void ProblemQP::calc_jac_g()
{
  // TODO Check behaviour if A is not set!
  unsigned int n = this->A.rows();

  unsigned int counter = 0;
  for (int k=0; k<this->A.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(this->A,k); it; ++it)
    {
      unsigned int row = it.row();
      unsigned int col = it.col();
      double val = it.value();

      // sparsity-structure
      this->nnz_I.push_back(row);
      this->nnz_J.push_back(col);

      // values
      unsigned int flat_index = counter;

      this->nnz_flat.push_back(flat_index);
      this->nnz_V.push_back(val);
      counter++;
    }
}

// Setters

void ProblemQP::set_c(RowArray& c)
{
    this->c = c;
}

void ProblemQP::set_Q(SpMat& Q)
{
    this->Q = Q;
}

void ProblemQP::set_A(SpMat& A)
{
    this->A = A;
    this->calc_jac_g();
}

void ProblemQP::set_glb(RowArray& glb)
{
    this->glb = glb;
}

void ProblemQP::set_gub(RowArray& gub)
{
    this->gub = gub;
}

void ProblemQP::set_xlb(RowArray& xlb)
{
    this->xlb = xlb;
}

void ProblemQP::set_xub(RowArray& xub)
{
    this->xub = xub;
}

void ProblemQP::set_var_types(std::vector<int>& var_types)
{
    this->var_types = var_types;
}

void ProblemQP::set_initial_point(std::vector<double>& initial_point)
{
    this->initial_point = initial_point;
    this->use_initial_point = true;
}

// Getters

std::vector<unsigned int>& ProblemQP::get_nnz_I()
{
    return this->nnz_I;
}

std::vector<unsigned int>& ProblemQP::get_nnz_J()
{
    return this->nnz_J;
}

std::vector<unsigned int>& ProblemQP::get_nnz_flat()
{
    return this->nnz_flat;
}

std::vector<double>& ProblemQP::get_nnz_V()
{
    return this->nnz_V;
}

int ProblemQP::get_n()
{
    return this->c.size();
}

int ProblemQP::get_m()
{
    return this->A.rows();
}

RowArray& ProblemQP::get_c()
{
    return this->c;
}

SpMat& ProblemQP::get_Q()
{
    return this->Q;
}

SpMat& ProblemQP::get_A()
{
    return this->A;
}

RowArray& ProblemQP::get_glb()
{
    return this->glb;
}

RowArray& ProblemQP::get_gub()
{
    return this->gub;
}

RowArray& ProblemQP::get_xlb()
{
    return this->xlb;
}

RowArray& ProblemQP::get_xub()
{
    return this->xub;
}

std::vector<int>& ProblemQP::get_variables_types()
{
    return this->var_types;
}

std::vector<double>& ProblemQP::get_initial_point()
{
    return this->initial_point;
}

bool ProblemQP::get_initial_point_status()
{
  return this->use_initial_point;
}
