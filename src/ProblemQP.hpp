#ifndef PROBLEMQP_HPP
#define PROBLEMQP_HPP
#include <Eigen/Eigen>

typedef Eigen::VectorXd RowArray;
typedef Eigen::SparseMatrix<double> SpMat;

class ProblemQP
{
    // QP data
    RowArray c;
    SpMat Q;
    SpMat A;
    RowArray glb;
    RowArray gub;
    RowArray xlb;
    RowArray xub;
    std::vector<int> var_types;
    std::vector<double> initial_point;

    // Flags
    bool use_initial_point = false;

    // Helper data: sparsity-structure of A = jacobian
    std::vector<unsigned int> nnz_I;
    std::vector<unsigned int> nnz_J;

    // Helper data: corresponding values of A as sparse-vec
    std::vector<unsigned int> nnz_flat;
    std::vector<double> nnz_V;

    void calc_jac_g();

public:
    void set_c(RowArray& c);
    void set_Q(SpMat& Q);
    void set_A(SpMat& A);
    void set_glb(RowArray& glb);
    void set_gub(RowArray& gub);
    void set_xlb(RowArray& xlb);
    void set_xub(RowArray& xub);
    void set_var_types(std::vector<int>& var_types);
    void set_initial_point(std::vector<double>& initial_point);

    std::vector<unsigned int>& get_nnz_I();
    std::vector<unsigned int>& get_nnz_J();
    std::vector<unsigned int>& get_nnz_flat();
    std::vector<double>& get_nnz_V();
    int get_n();
    int get_m();
    RowArray& get_c();
    SpMat& get_Q();
    SpMat& get_A();
    RowArray& get_glb();
    RowArray& get_gub();
    RowArray& get_xlb();
    RowArray& get_xub();
    std::vector<int>& get_variables_types();
    std::vector<double>& get_initial_point();
    bool get_initial_point_status();

};
#endif
